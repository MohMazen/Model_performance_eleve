"""
Tests de non-régression pour la version vectorisée de TemporalAnalyzer.compute_trends.

Compare les résultats avec une implémentation de référence basée sur
scipy.stats.linregress en boucle, pour garantir l'équivalence numérique.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.temporal import TemporalAnalyzer


def _ref_compute_trends(df_multi: pd.DataFrame, id_col: str, target: str) -> pd.DataFrame:
    """Implémentation de référence (boucle scipy.stats.linregress)."""
    rows = []
    for sid, group in df_multi.groupby(id_col):
        g = group.sort_values("periode_order")
        values = g[target].dropna().values
        n = len(values)
        if n < 2:
            rows.append({id_col: sid, "slope": 0.0, "n_periods": n, "trend_label": "➡️ Stable",
                         "first_value": float(values[0]) if n else 0.0,
                         "last_value": float(values[-1]) if n else 0.0})
            continue
        x = np.arange(n)
        slope, intercept, r_value, p_value, _ = stats.linregress(x, values)
        if slope > 0.5 and p_value < 0.1:
            label = "↗️ Progression"
        elif slope < -0.5 and p_value < 0.1:
            label = "↘️ Régression"
        else:
            label = "➡️ Stable"
        rows.append({id_col: sid, "slope": round(slope, 3), "intercept": round(intercept, 2),
                     "r_value": round(r_value, 3), "p_value": round(p_value, 4),
                     "n_periods": n, "trend_label": label,
                     "first_value": round(float(values[0]), 2),
                     "last_value": round(float(values[-1]), 2)})
    return pd.DataFrame(rows)


@pytest.fixture
def df_multi_periods():
    """Génère 50 élèves x 4 périodes avec des trajectoires variées."""
    rng = np.random.default_rng(0)
    rows = []
    for student_id in range(50):
        # Tendance linéaire ± bruit, avec variation par élève.
        slope = rng.uniform(-2.0, 2.0)
        for p in range(4):
            note = 10 + slope * p + rng.normal(0, 0.5)
            rows.append({"nom": f"eleve_{student_id}", "periode_order": p,
                         "note_moyenne": float(np.clip(note, 0, 20))})
    return pd.DataFrame(rows)


class TestVectorizedEquivalence:
    def test_slope_matches_reference(self, df_multi_periods):
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        result = ta.compute_trends(df_multi_periods).set_index("nom").sort_index()
        ref = _ref_compute_trends(df_multi_periods, "nom", "note_moyenne") \
            .set_index("nom").sort_index()

        # Les pentes doivent coïncider à l'arrondi près (3 décimales).
        np.testing.assert_allclose(result["slope"].values, ref["slope"].values, atol=1e-3)

    def test_trend_label_matches_reference(self, df_multi_periods):
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        result = ta.compute_trends(df_multi_periods).set_index("nom").sort_index()
        ref = _ref_compute_trends(df_multi_periods, "nom", "note_moyenne") \
            .set_index("nom").sort_index()
        assert (result["trend_label"] == ref["trend_label"]).all()

    def test_first_last_delta(self, df_multi_periods):
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        result = ta.compute_trends(df_multi_periods).set_index("nom").sort_index()
        ref = _ref_compute_trends(df_multi_periods, "nom", "note_moyenne") \
            .set_index("nom").sort_index()
        np.testing.assert_allclose(result["first_value"].values, ref["first_value"].values, atol=1e-2)
        np.testing.assert_allclose(result["last_value"].values, ref["last_value"].values, atol=1e-2)


class TestEdgeCases:
    def test_single_period_returns_stable(self):
        df = pd.DataFrame({
            "nom": ["a", "b"], "periode_order": [0, 0], "note_moyenne": [10.0, 12.0]
        })
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        result = ta.compute_trends(df)
        assert (result["trend_label"] == "➡️ Stable").all()
        assert (result["slope"] == 0.0).all()

    def test_constant_values_yields_zero_slope(self):
        df = pd.DataFrame({
            "nom": ["a"] * 4, "periode_order": list(range(4)),
            "note_moyenne": [12.0, 12.0, 12.0, 12.0]
        })
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        result = ta.compute_trends(df)
        assert result.loc[0, "slope"] == 0.0
        assert result.loc[0, "trend_label"] == "➡️ Stable"

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        with pytest.raises(ValueError):
            ta.compute_trends(df)

    def test_handles_nan_in_target(self):
        df = pd.DataFrame({
            "nom": ["a", "a", "a", "a"],
            "periode_order": [0, 1, 2, 3],
            "note_moyenne": [10.0, np.nan, 12.0, 14.0],
        })
        ta = TemporalAnalyzer(id_col="nom", target="note_moyenne")
        result = ta.compute_trends(df)
        # 3 valeurs valides, pente positive attendue.
        assert result.loc[0, "n_periods"] == 3
        assert result.loc[0, "slope"] > 0
