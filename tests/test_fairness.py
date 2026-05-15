"""
Tests unitaires pour le module src/fairness.py.

Couvre les métriques par groupe, les écarts de parité démographique et
d'égalité des chances, ainsi que les niveaux d'alerte.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fairness import (
    audit_fairness,
    compute_group_metrics_clf,
    compute_group_metrics_reg,
    demographic_parity_difference,
    equalized_odds_difference,
    format_fairness_report,
    get_privacy_preserving_features,
    compare_privacy_performance,
    SENSITIVE_COLS,
)


@pytest.fixture
def df_balanced():
    """Dataset équilibré : taux de réussite identiques entre groupes → fairness OK."""
    rng = np.random.default_rng(0)
    n_per_group = 50
    rows = []
    for genre in ('M', 'F'):
        y_true = rng.binomial(1, 0.6, n_per_group)
        # Modèle parfait → y_pred = y_true
        for yt in y_true:
            rows.append({'genre': genre, 'y_true': int(yt), 'y_pred': int(yt)})
    return pd.DataFrame(rows)


@pytest.fixture
def df_biased():
    """Dataset biaisé : modèle prédit toujours 1 pour M, toujours 0 pour F."""
    rng = np.random.default_rng(0)
    rows = []
    for genre in ('M', 'F'):
        y_true = rng.binomial(1, 0.5, 50)
        y_pred = np.ones(50, dtype=int) if genre == 'M' else np.zeros(50, dtype=int)
        for yt, yp in zip(y_true, y_pred):
            rows.append({'genre': genre, 'y_true': int(yt), 'y_pred': int(yp)})
    return pd.DataFrame(rows)


class TestComputeGroupMetricsClf:
    def test_returns_one_row_per_group(self, df_balanced):
        m = compute_group_metrics_clf(df_balanced, 'genre', 'y_true', 'y_pred')
        assert set(m.index) == {'M', 'F'}

    def test_perfect_model_has_tpr_one_fpr_zero(self, df_balanced):
        m = compute_group_metrics_clf(df_balanced, 'genre', 'y_true', 'y_pred')
        assert (m['tpr'] == 1.0).all()
        assert (m['fpr'] == 0.0).all()
        assert (m['accuracy'] == 1.0).all()

    def test_biased_model_extreme_tpr_fpr(self, df_biased):
        m = compute_group_metrics_clf(df_biased, 'genre', 'y_true', 'y_pred')
        # Pour M : tout prédit 1 → TPR=1, FPR=1
        assert m.loc['M', 'tpr'] == 1.0
        assert m.loc['M', 'fpr'] == 1.0
        # Pour F : tout prédit 0 → TPR=0, FPR=0
        assert m.loc['F', 'tpr'] == 0.0
        assert m.loc['F', 'fpr'] == 0.0


class TestDemographicParity:
    def test_zero_when_balanced(self, df_balanced):
        m = compute_group_metrics_clf(df_balanced, 'genre', 'y_true', 'y_pred')
        # Avec un modèle parfait, le taux positif prédit dépend du vrai taux,
        # donc la parité dépend de la distribution réelle. Les groupes ont
        # été générés avec p=0.6 → écart faible (< 0.2).
        assert demographic_parity_difference(m) < 0.2

    def test_high_when_biased(self, df_biased):
        m = compute_group_metrics_clf(df_biased, 'genre', 'y_true', 'y_pred')
        # M prédit toujours 1, F toujours 0 → écart = 1.0
        assert demographic_parity_difference(m) == pytest.approx(1.0)


class TestEqualizedOdds:
    def test_zero_for_perfect_model(self, df_balanced):
        m = compute_group_metrics_clf(df_balanced, 'genre', 'y_true', 'y_pred')
        eo = equalized_odds_difference(m)
        assert eo['tpr_diff'] == 0.0
        assert eo['fpr_diff'] == 0.0

    def test_max_for_extreme_bias(self, df_biased):
        m = compute_group_metrics_clf(df_biased, 'genre', 'y_true', 'y_pred')
        eo = equalized_odds_difference(m)
        assert eo['tpr_diff'] == pytest.approx(1.0)
        assert eo['fpr_diff'] == pytest.approx(1.0)


class TestAuditFairness:
    def test_alert_level_ok_for_balanced(self, df_balanced):
        audit = audit_fairness(df_balanced, sensitive_col='genre',
                               y_true_clf='y_true', y_pred_clf='y_pred')
        assert audit['alert_level'] == 'OK'
        assert 'classification' in audit
        assert audit['group_sizes'] == {'M': 50, 'F': 50}

    def test_alert_level_critique_for_biased(self, df_biased):
        audit = audit_fairness(df_biased, sensitive_col='genre',
                               y_true_clf='y_true', y_pred_clf='y_pred')
        assert audit['alert_level'] == 'CRITIQUE'

    def test_skips_small_groups(self):
        df = pd.DataFrame({
            'genre': ['M'] * 50 + ['F'] * 50 + ['X'] * 2,
            'y_true': [0, 1] * 51,
            'y_pred': [0, 1] * 51,
        })
        audit = audit_fairness(df, sensitive_col='genre',
                               y_true_clf='y_true', y_pred_clf='y_pred',
                               min_group_size=5)
        assert audit['skipped'] == ['X']

    def test_raises_on_missing_sensitive_col(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with pytest.raises(ValueError):
            audit_fairness(df, sensitive_col='inexistant',
                           y_true_clf='a', y_pred_clf='b')


class TestRegressionAudit:
    def test_regression_metrics_per_group(self):
        df = pd.DataFrame({
            'classe': ['A'] * 20 + ['B'] * 20,
            'note_true': [10.0] * 20 + [12.0] * 20,
            'note_pred': [10.5] * 20 + [11.0] * 20,
        })
        audit = audit_fairness(df, sensitive_col='classe',
                               y_true_reg='note_true', y_pred_reg='note_pred')
        assert 'regression' in audit
        per = audit['regression']['per_group']
        assert per.loc['A', 'mae'] == pytest.approx(0.5)
        assert per.loc['B', 'mae'] == pytest.approx(1.0)
        # Biais : A surestime de +0.5, B sous-estime de -1.0
        assert per.loc['A', 'mean_bias'] == pytest.approx(0.5)
        assert per.loc['B', 'mean_bias'] == pytest.approx(-1.0)


class TestFormatReport:
    def test_report_contains_essential_info(self, df_biased):
        audit = audit_fairness(df_biased, sensitive_col='genre',
                               y_true_clf='y_true', y_pred_clf='y_pred')
        report = format_fairness_report(audit)
        assert 'Audit fairness' in report
        assert 'genre' in report
        assert 'CRITIQUE' in report
        assert 'Demographic parity' in report


class TestGetPrivacyPreservingFeatures:
    def test_drops_known_sensitive_columns(self):
        df = pd.DataFrame({'genre': ['M', 'F'], 'score': [10, 12], 'age': [17, 18]})
        df_out, dropped = get_privacy_preserving_features(df)
        assert 'genre' not in df_out.columns
        assert 'age' not in df_out.columns
        assert 'score' in df_out.columns
        assert 'genre' in dropped
        assert 'age' in dropped

    def test_returns_unchanged_when_no_sensitive_cols(self):
        df = pd.DataFrame({'score': [10, 12], 'heures': [5, 6]})
        df_out, dropped = get_privacy_preserving_features(df)
        assert list(df_out.columns) == ['score', 'heures']
        assert dropped == []

    def test_extra_sensitive_cols_are_dropped(self):
        df = pd.DataFrame({'score': [1, 2], 'classe': ['A', 'B']})
        df_out, dropped = get_privacy_preserving_features(df, extra_sensitive=['classe'])
        assert 'classe' not in df_out.columns
        assert 'classe' in dropped

    def test_original_df_not_modified(self):
        df = pd.DataFrame({'genre': ['M', 'F'], 'score': [10, 12]})
        cols_before = list(df.columns)
        get_privacy_preserving_features(df)
        assert list(df.columns) == cols_before

    def test_sensitive_cols_is_list_of_strings(self):
        assert isinstance(SENSITIVE_COLS, list)
        assert all(isinstance(c, str) for c in SENSITIVE_COLS)


class TestComparePrivacyPerformance:
    @pytest.fixture
    def simple_data(self):
        rng = np.random.default_rng(42)
        n = 120
        score = rng.uniform(0, 20, n)
        X = pd.DataFrame({
            'score': score,
            'assiduite': rng.uniform(0, 100, n),
            'genre': rng.integers(0, 2, n),
        })
        y = pd.Series((score > 10).astype(int))
        return X, y

    def test_returns_expected_keys(self, simple_data):
        X, y = simple_data
        result = compare_privacy_performance(X, y, cv=2)
        expected = [
            'f1_full', 'f1_privacy', 'delta_f1', 'delta_pct',
            'accuracy_full', 'accuracy_privacy', 'cols_removed',
            'n_cols_removed', 'privacy_cost_acceptable', 'reference',
        ]
        for key in expected:
            assert key in result, f"Clé manquante : {key}"

    def test_privacy_cost_acceptable_is_bool(self, simple_data):
        X, y = simple_data
        result = compare_privacy_performance(X, y, cv=2)
        assert isinstance(result['privacy_cost_acceptable'], bool)

    def test_genre_column_removed(self, simple_data):
        X, y = simple_data
        result = compare_privacy_performance(X, y, cv=2)
        assert 'genre' in result['cols_removed']

    def test_f1_values_in_valid_range(self, simple_data):
        X, y = simple_data
        result = compare_privacy_performance(X, y, cv=2)
        assert 0.0 <= result['f1_full'] <= 1.0
        assert 0.0 <= result['f1_privacy'] <= 1.0

    def test_delta_consistent(self, simple_data):
        X, y = simple_data
        result = compare_privacy_performance(X, y, cv=2)
        expected_delta = round(result['f1_privacy'] - result['f1_full'], 4)
        assert abs(result['delta_f1'] - expected_delta) < 1e-3
