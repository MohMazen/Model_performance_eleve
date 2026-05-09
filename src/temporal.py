"""
Analyse Temporelle et Suivi Longitudinal.
Tracking de l'évolution des élèves sur plusieurs périodes.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyse longitudinale des performances élèves."""

    def __init__(self, id_col: str = "nom", target: str = "note_moyenne") -> None:
        self.id_col = id_col
        self.target = target

    def load_multi_period(self, paths_with_labels: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Charge et fusionne des CSVs multi-périodes.

        Parameters
        ----------
        paths_with_labels : list of (path, period_label) tuples
            Ex: [("data/t1.csv", "Trimestre 1"), ("data/t2.csv", "Trimestre 2")]
        """
        frames = []
        for path, label in paths_with_labels:
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
                df.columns = [c.lower() for c in df.columns]
                df["periode"] = label
                df["periode_order"] = len(frames)
                frames.append(df)
                logger.info(f"Période '{label}' chargée : {len(df)} lignes depuis {path}")
            except Exception as e:
                logger.error(f"Erreur de chargement {path} : {e}")
        if not frames:
            raise ValueError("Aucun fichier n'a pu être chargé.")
        return pd.concat(frames, ignore_index=True)

    def merge_uploaded_periods(self, dataframes: List[Tuple[pd.DataFrame, str]]) -> pd.DataFrame:
        """Fusionne des DataFrames déjà chargés avec leurs labels de période."""
        frames = []
        for i, (df, label) in enumerate(dataframes):
            df_copy = df.copy()
            df_copy.columns = [c.lower() for c in df_copy.columns]
            df_copy["periode"] = label
            df_copy["periode_order"] = i
            frames.append(df_copy)
        return pd.concat(frames, ignore_index=True)

    def compute_trends(self, df_multi: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule la pente de tendance par élève.

        Returns DataFrame avec colonnes : id_col, slope, intercept, r_value, p_value, trend_label
        """
        logger.info("Calcul des tendances de performance par élève…")
        if self.id_col not in df_multi.columns or self.target not in df_multi.columns:
            raise ValueError(f"Colonnes requises manquantes : {self.id_col}, {self.target}")

        results = []
        for student_id, group in df_multi.groupby(self.id_col):
            group_sorted = group.sort_values("periode_order")
            values = group_sorted[self.target].dropna().values
            if len(values) < 2:
                results.append({self.id_col: student_id, "slope": 0.0, "intercept": values[0] if len(values) else 0,
                                "r_value": 0.0, "p_value": 1.0, "n_periods": len(values), "trend_label": "➡️ Stable"})
                continue
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, _ = stats.linregress(x, values)
            if slope > 0.5 and p_value < 0.1:
                trend = "↗️ Progression"
            elif slope < -0.5 and p_value < 0.1:
                trend = "↘️ Régression"
            else:
                trend = "➡️ Stable"
            results.append({self.id_col: student_id, "slope": round(slope, 3), "intercept": round(intercept, 2),
                            "r_value": round(r_value, 3), "p_value": round(p_value, 4),
                            "n_periods": len(values), "trend_label": trend,
                            "last_value": round(values[-1], 2), "first_value": round(values[0], 2),
                            "delta": round(values[-1] - values[0], 2)})

        df_trends = pd.DataFrame(results)
        logger.info(f"Tendances calculées pour {len(df_trends)} élèves.")
        return df_trends

    def detect_dropout_risk(self, trends_df: pd.DataFrame, slope_threshold: float = -1.0) -> pd.DataFrame:
        """Identifie les élèves en décrochage (pente significativement négative)."""
        at_risk = trends_df[
            (trends_df["slope"] <= slope_threshold) & (trends_df["p_value"] < 0.15)
        ].copy()
        at_risk = at_risk.sort_values("slope", ascending=True)
        logger.info(f"{len(at_risk)} élèves identifiés en risque de décrochage (seuil pente ≤ {slope_threshold}).")
        return at_risk

    def compute_delta_metrics(self, metrics_t1: Dict[str, float], metrics_t2: Dict[str, float]) -> Dict[str, Dict]:
        """Calcule les variations de métriques entre deux périodes."""
        deltas = {}
        for key in set(list(metrics_t1.keys()) + list(metrics_t2.keys())):
            v1 = metrics_t1.get(key, 0)
            v2 = metrics_t2.get(key, 0)
            delta = v2 - v1
            pct = (delta / v1 * 100) if v1 != 0 else 0
            direction = "↗️" if delta > 0 else ("↘️" if delta < 0 else "➡️")
            deltas[key] = {"t1": round(v1, 4), "t2": round(v2, 4), "delta": round(delta, 4),
                           "pct_change": round(pct, 1), "direction": direction}
        return deltas

    def get_period_summary(self, df_multi: pd.DataFrame) -> pd.DataFrame:
        """Résumé statistique par période."""
        if self.target not in df_multi.columns:
            return pd.DataFrame()
        summary = df_multi.groupby("periode").agg(
            nb_eleves=(self.target, "count"),
            moyenne=(self.target, "mean"),
            ecart_type=(self.target, "std"),
            min_note=(self.target, "min"),
            max_note=(self.target, "max"),
            taux_reussite=(self.target, lambda x: (x >= 10).mean() * 100),
        ).round(2)
        return summary.sort_index()


def generer_donnees_multi_periodes(n_eleves: int = 100, n_periodes: int = 3) -> pd.DataFrame:
    """Génère des données synthétiques multi-périodes pour tester le suivi longitudinal."""
    from src.data_utils import generer_donnees_synthetiques
    frames = []
    period_labels = [f"Trimestre {i+1}" for i in range(n_periodes)]
    for i, label in enumerate(period_labels):
        np.random.seed(42 + i)
        df = generer_donnees_synthetiques(n_eleves)
        # Simuler une évolution : légère amélioration/dégradation progressive
        if 'note_moyenne' in df.columns:
            noise = np.random.normal(i * 0.3, 0.5, len(df))
            df['note_moyenne'] = np.clip(df['note_moyenne'] + noise, 0, 20)
        df["periode"] = label
        df["periode_order"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
