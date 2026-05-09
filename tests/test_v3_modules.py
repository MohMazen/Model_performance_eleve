"""
Tests unitaires pour les nouveaux modules v3.0 :
- early_warning.py
- clustering.py
- temporal.py
- recommendations.py
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import generer_donnees_synthetiques, nettoyer_donnees
from src.features import add_advanced_features, prenttoyer_horaires
from src.early_warning import EarlyWarningSystem, RISK_ZONES
from src.clustering import StudentProfiler
from src.temporal import TemporalAnalyzer, generer_donnees_multi_periodes
from src.recommendations import RecommendationEngine


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def df_feat():
    """DataFrame avec features calculées."""
    df = generer_donnees_synthetiques(n_eleves=60)
    df = nettoyer_donnees(df)
    df = prenttoyer_horaires(df)
    df = add_advanced_features(df)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Tests Early Warning System
# ══════════════════════════════════════════════════════════════════════════
class TestEarlyWarning:
    def test_compute_risk_score_returns_dict(self, df_feat):
        ews = EarlyWarningSystem()
        row = df_feat.iloc[0]
        result = ews.compute_risk_score(row, proba_echec=0.7)
        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "risk_zone" in result
        assert "triggered_rules" in result

    def test_risk_score_range(self, df_feat):
        ews = EarlyWarningSystem()
        for i in range(min(10, len(df_feat))):
            result = ews.compute_risk_score(df_feat.iloc[i], proba_echec=np.random.random())
            assert 0 <= result["risk_score"] <= 100

    def test_risk_zone_valid(self, df_feat):
        ews = EarlyWarningSystem()
        valid_zones = {"serein", "vigilance", "alerte", "critique"}
        result = ews.compute_risk_score(df_feat.iloc[0], proba_echec=0.9)
        assert result["risk_zone"] in valid_zones

    def test_high_proba_gives_high_score(self, df_feat):
        ews = EarlyWarningSystem()
        low = ews.compute_risk_score(df_feat.iloc[0], proba_echec=0.1)
        high = ews.compute_risk_score(df_feat.iloc[0], proba_echec=0.95)
        assert high["risk_score"] >= low["risk_score"]

    def test_generate_alerts_empty_below_threshold(self, df_feat):
        df_test = df_feat.copy()
        df_test["risk_score"] = 10  # Tous en dessous
        alerts = EarlyWarningSystem.generate_alerts(df_test, seuil_alerte=50)
        assert len(alerts) == 0

    def test_generate_alerts_returns_above_threshold(self, df_feat):
        df_test = df_feat.copy()
        df_test["risk_score"] = np.random.uniform(0, 100, len(df_test))
        alerts = EarlyWarningSystem.generate_alerts(df_test, seuil_alerte=50)
        assert all(alerts["risk_score"] >= 50)

    def test_custom_weights(self, df_feat):
        ews = EarlyWarningSystem(ml_weight=0.8, rules_weight=0.2)
        result = ews.compute_risk_score(df_feat.iloc[0], proba_echec=0.5)
        assert isinstance(result["risk_score"], float)


# ══════════════════════════════════════════════════════════════════════════
# Tests Clustering
# ══════════════════════════════════════════════════════════════════════════
class TestClustering:
    def test_fit_clusters_returns_labels(self, df_feat):
        profiler = StudentProfiler()
        labels, X_used = profiler.fit_clusters(df_feat, method='kmeans', n_clusters=3)
        assert len(labels) == len(df_feat)
        assert len(set(labels)) >= 2

    def test_describe_clusters(self, df_feat):
        profiler = StudentProfiler()
        labels, _ = profiler.fit_clusters(df_feat, method='kmeans', n_clusters=3)
        profiles = profiler.describe_clusters(df_feat, labels)
        assert len(profiles) >= 2
        for cid, p in profiles.items():
            assert "size" in p
            assert "dominant_features" in p
            assert p["size"] > 0

    def test_generate_cluster_names(self, df_feat):
        profiler = StudentProfiler()
        labels, _ = profiler.fit_clusters(df_feat, method='kmeans', n_clusters=3)
        profiles = profiler.describe_clusters(df_feat, labels)
        names = profiler.generate_cluster_names(profiles)
        assert len(names) == len(profiles)
        for name in names.values():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_reduce_dimensions_pca(self, df_feat):
        profiler = StudentProfiler()
        profiler.fit_clusters(df_feat, method='kmeans', n_clusters=3)
        coords = profiler.reduce_dimensions(df_feat, method='pca')
        assert coords.shape == (len(df_feat), 2)

    def test_find_optimal_k(self, df_feat):
        profiler = StudentProfiler()
        X_raw = profiler._select_features(df_feat).fillna(0)
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X_raw)
        k = profiler.find_optimal_k(X_scaled, k_range=(2, 5))
        assert 2 <= k <= 5


# ══════════════════════════════════════════════════════════════════════════
# Tests Temporal
# ══════════════════════════════════════════════════════════════════════════
class TestTemporal:
    def test_generate_multi_period_data(self):
        df_multi = generer_donnees_multi_periodes(n_eleves=30, n_periodes=3)
        assert "periode" in df_multi.columns
        assert "periode_order" in df_multi.columns
        assert df_multi["periode"].nunique() == 3

    def test_compute_trends(self):
        df_multi = generer_donnees_multi_periodes(n_eleves=30, n_periodes=3)
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        trends = ta.compute_trends(df_multi)
        assert "slope" in trends.columns
        assert "trend_label" in trends.columns
        assert len(trends) > 0

    def test_detect_dropout_risk(self):
        df_multi = generer_donnees_multi_periodes(n_eleves=30, n_periodes=3)
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        trends = ta.compute_trends(df_multi)
        at_risk = ta.detect_dropout_risk(trends, slope_threshold=-0.1)
        assert isinstance(at_risk, pd.DataFrame)

    def test_compute_delta_metrics(self):
        ta = TemporalAnalyzer()
        m1 = {"r2": 0.7, "mae": 1.5}
        m2 = {"r2": 0.8, "mae": 1.2}
        deltas = ta.compute_delta_metrics(m1, m2)
        assert deltas["r2"]["delta"] == pytest.approx(0.1, abs=0.01)
        assert deltas["mae"]["direction"] == "↘️"

    def test_get_period_summary(self):
        df_multi = generer_donnees_multi_periodes(n_eleves=30, n_periodes=3)
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        summary = ta.get_period_summary(df_multi)
        assert len(summary) == 3
        assert "moyenne" in summary.columns

    def test_merge_uploaded_periods(self):
        df1 = generer_donnees_synthetiques(20)
        df2 = generer_donnees_synthetiques(20)
        ta = TemporalAnalyzer()
        merged = ta.merge_uploaded_periods([(df1, "P1"), (df2, "P2")])
        assert merged["periode"].nunique() == 2
        assert len(merged) == 40


# ══════════════════════════════════════════════════════════════════════════
# Tests Recommendations
# ══════════════════════════════════════════════════════════════════════════
class TestRecommendations:
    def test_get_individual_recommendations(self, df_feat):
        engine = RecommendationEngine()
        student = df_feat.iloc[0]
        # Fake SHAP values
        features = ["heures_etude_soir", "heures_sommeil", "stress_personnel", "confiance_soi", "organisation"]
        shap_vals = np.array([-0.5, -0.3, 0.4, -0.2, -0.1])
        recos = engine.get_individual_recommendations(student, features, shap_vals, top_n=3)
        assert isinstance(recos, list)
        assert len(recos) <= 3
        for r in recos:
            assert "factor" in r
            assert "recommendation" in r

    def test_generate_action_plan(self):
        engine = RecommendationEngine()
        recos = [
            {"factor": "heures_etude_soir", "label": "Temps d'étude", "icon": "📚",
             "recommendation": "Augmenter", "current_value": 1.5, "suggested_value": 3.0,
             "shap_impact": -0.5, "unit": "h", "is_improvable": True, "abs_impact": 0.5, "direction": "increase"},
        ]
        plan = engine.generate_action_plan("Test Élève", recos, note_actuelle=8.5, note_predite=9.2)
        assert "Test Élève" in plan
        assert "8.50" in plan
        assert "Temps d'étude" in plan

    def test_empty_recommendations_for_good_student(self, df_feat):
        engine = RecommendationEngine()
        student = df_feat.iloc[0]
        features = ["age", "classe"]  # Non-actionable
        shap_vals = np.array([0.01, 0.02])
        recos = engine.get_individual_recommendations(student, features, shap_vals)
        assert isinstance(recos, list)
        assert len(recos) == 0
