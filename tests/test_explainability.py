"""
Tests unitaires pour src/explainability.py.

Couvre les trois fonctions exposées (generate_shap_analysis,
generate_shap_failure_analysis, get_individual_shap_values) et la sélection
des top facteurs (get_top_actionable_factors).
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use("Agg")

from src.data_utils import generer_donnees_synthetiques, nettoyer_donnees
from src.features import nettoyer_horaires, add_advanced_features
from src.config import COLS_TO_DROP, TARGET_REG, TARGET_CLF
from src.models import ModelManager
from src.explainability import (
    generate_shap_analysis,
    generate_shap_failure_analysis,
    get_individual_shap_values,
    get_top_actionable_factors,
)


@pytest.fixture(scope="module")
def trained_pipeline():
    """Entraîne un pipeline XGBoost minimal pour les tests SHAP."""
    df = generer_donnees_synthetiques(n_eleves=80)
    df = nettoyer_donnees(df)
    df = nettoyer_horaires(df)
    df = add_advanced_features(df)

    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=cols_drop + [TARGET_REG, TARGET_CLF])
    y_reg = df[TARGET_REG]

    mm = ModelManager()
    mm.prepare_pipeline(X)
    pipeline = mm.train_regression(X, y_reg)
    return pipeline, X, y_reg


class TestGenerateShapAnalysis:
    def test_returns_shap_values(self, trained_pipeline):
        pipeline, X, _ = trained_pipeline
        buf = io.BytesIO()
        result = generate_shap_analysis(pipeline, X.iloc[:20], buf=buf)
        assert result is not None
        assert buf.getbuffer().nbytes > 0

    def test_handles_empty_sample(self, trained_pipeline):
        """Un échantillon vide ne doit pas lever d'exception, juste retourner None ou vide."""
        pipeline, X, _ = trained_pipeline
        buf = io.BytesIO()
        result = generate_shap_analysis(pipeline, X.iloc[:0], buf=buf)
        # Tolère soit None soit un objet shap valide (bien que vide).
        assert result is None or hasattr(result, "values")


class TestGenerateShapFailureAnalysis:
    def test_returns_none_when_no_failure_legacy_mode(self, trained_pipeline):
        """Mode legacy (use_predictions=False) : filtrage sur y_true."""
        pipeline, X, _ = trained_pipeline
        # Cible artificielle entièrement >= seuil → aucun échec.
        y_all_pass = pd.Series([20.0] * len(X), index=X.index)
        buf = io.BytesIO()
        result = generate_shap_failure_analysis(pipeline, X.iloc[:20],
                                                y_all_pass.iloc[:20],
                                                seuil=10.0, buf=buf,
                                                use_predictions=False)
        assert result is None

    def test_returns_none_when_no_failure_predicted(self, trained_pipeline):
        """Mode par défaut : filtrage sur prédictions du modèle.
        Avec un seuil très élevé hors plage de prédictions (-100), aucun élève
        ne peut être 'prédit en échec', donc résultat None."""
        pipeline, X, _ = trained_pipeline
        buf = io.BytesIO()
        result = generate_shap_failure_analysis(pipeline, X.iloc[:20],
                                                seuil=-100.0, buf=buf,
                                                use_predictions=True)
        assert result is None

    def test_returns_shap_values_when_predictions_below_threshold(self, trained_pipeline):
        """Mode par défaut : seuil élevé → la majorité des élèves sont
        prédits 'en échec' → SHAP values calculées."""
        pipeline, X, _ = trained_pipeline
        buf = io.BytesIO()
        # Seuil très haut pour garantir que des prédictions tombent en dessous.
        result = generate_shap_failure_analysis(pipeline, X.iloc[:20],
                                                seuil=20.0, buf=buf,
                                                use_predictions=True)
        assert result is not None

    def test_legacy_mode_returns_shap_values_when_failures_exist(self, trained_pipeline):
        pipeline, X, y_reg = trained_pipeline
        # Forcer au moins quelques échecs en abaissant artificiellement la cible.
        y_failures = y_reg.copy()
        y_failures.iloc[:5] = 5.0
        buf = io.BytesIO()
        result = generate_shap_failure_analysis(pipeline, X.iloc[:20],
                                                y_failures.iloc[:20],
                                                seuil=10.0, buf=buf,
                                                use_predictions=False)
        assert result is not None


class TestGetIndividualShapValues:
    def test_returns_dict_with_expected_keys(self, trained_pipeline):
        pipeline, X, _ = trained_pipeline
        result = get_individual_shap_values(pipeline, X.iloc[:10], student_index=0)
        assert result is not None
        assert "feature_names" in result
        assert "shap_values" in result
        assert "base_value" in result
        assert len(result["feature_names"]) == len(result["shap_values"])

    def test_index_clamped_to_valid_range(self, trained_pipeline):
        """Un index hors limite est ramené au dernier élève."""
        pipeline, X, _ = trained_pipeline
        result = get_individual_shap_values(pipeline, X.iloc[:5], student_index=999)
        assert result is not None
        assert len(result["shap_values"]) > 0


class TestGetTopActionableFactors:
    def test_returns_top_n_sorted_by_abs_impact(self):
        shap_result = {
            "feature_names": ["a", "b", "c", "d"],
            "shap_values": np.array([0.1, -0.5, 0.3, -0.2]),
            "base_value": 10.0,
        }
        top = get_top_actionable_factors(shap_result, top_n=2)
        assert len(top) == 2
        # Le facteur avec |shap| le plus grand est "b" (0.5), puis "c" (0.3).
        assert top[0][0] == "b"
        assert top[1][0] == "c"

    def test_returns_empty_for_none_input(self):
        assert get_top_actionable_factors(None, top_n=5) == []

    def test_top_n_larger_than_features(self):
        shap_result = {
            "feature_names": ["a", "b"],
            "shap_values": np.array([0.1, 0.2]),
            "base_value": 0.0,
        }
        top = get_top_actionable_factors(shap_result, top_n=10)
        assert len(top) == 2
