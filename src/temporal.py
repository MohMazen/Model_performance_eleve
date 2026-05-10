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
        Calcule la pente de tendance par élève (implémentation vectorisée).

        Utilise la formule analytique de la régression linéaire OLS appliquée
        en groupby pour éviter une boucle Python par élève. Le test de Student
        sur la pente fournit la p-value sans appeler scipy.stats.linregress
        à chaque itération.

        Returns DataFrame avec colonnes : id_col, slope, intercept, r_value,
        p_value, n_periods, trend_label, first_value, last_value, delta.
        """
        from scipy.stats import t as student_t

        logger.info("Calcul des tendances de performance par élève (vectorisé)…")
        if self.id_col not in df_multi.columns or self.target not in df_multi.columns:
            raise ValueError(f"Colonnes requises manquantes : {self.id_col}, {self.target}")

        # Préparation : tri stable par élève puis par période, retrait des NaN.
        df = (
            df_multi[[self.id_col, "periode_order", self.target]]
            .dropna(subset=[self.target])
            .sort_values([self.id_col, "periode_order"])
            .copy()
        )
        # Index 0..n-1 par élève (axe x de la régression).
        df["x"] = df.groupby(self.id_col).cumcount()

        # Statistiques par élève via aggregations vectorisées.
        g = df.groupby(self.id_col, sort=False)
        n = g["x"].size().rename("n")
        sum_x = g["x"].sum().rename("sx")
        sum_y = g[self.target].sum().rename("sy")
        sum_xy = g.apply(lambda d: float((d["x"] * d[self.target]).sum())).rename("sxy")
        sum_xx = g.apply(lambda d: float((d["x"] ** 2).sum())).rename("sxx")
        sum_yy = g.apply(lambda d: float((d[self.target] ** 2).sum())).rename("syy")
        first_value = g[self.target].first().rename("first_value")
        last_value = g[self.target].last().rename("last_value")

        stats_df = pd.concat([n, sum_x, sum_y, sum_xy, sum_xx, sum_yy,
                              first_value, last_value], axis=1)

        # Pente OLS : (n·Σxy − Σx·Σy) / (n·Σxx − (Σx)²)
        denom_x = stats_df["n"] * stats_df["sxx"] - stats_df["sx"] ** 2
        numer = stats_df["n"] * stats_df["sxy"] - stats_df["sx"] * stats_df["sy"]
        slope = np.where(denom_x > 0, numer / denom_x, 0.0)
        intercept = (stats_df["sy"] - slope * stats_df["sx"]) / stats_df["n"]

        # Coefficient de corrélation r.
        denom_y = stats_df["n"] * stats_df["syy"] - stats_df["sy"] ** 2
        denom_r = np.sqrt(denom_x * denom_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_value = np.where(denom_r > 0, numer / denom_r, 0.0)
        r_value = np.clip(r_value, -1.0, 1.0)

        # p-value : test bilatéral sur la pente (n-2 degrés de liberté).
        # On force p=1.0 quand n<3 ou |r|=1 (cas dégénéré sans variance résiduelle).
        df_resid = (stats_df["n"] - 2).clip(lower=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = r_value * np.sqrt(df_resid / np.maximum(1 - r_value ** 2, 1e-12))
            p_value = 2 * (1 - student_t.cdf(np.abs(t_stat), df_resid))
        p_value = np.where(stats_df["n"] >= 3, p_value, 1.0)
        p_value = np.where(np.isfinite(p_value), p_value, 1.0)

        # Étiquette de tendance.
        progression = (slope > 0.5) & (p_value < 0.1)
        regression = (slope < -0.5) & (p_value < 0.1)
        trend_label = np.where(progression, "↗️ Progression",
                       np.where(regression, "↘️ Régression", "➡️ Stable"))

        # Cas n < 2 : tendance forcée à "Stable", slope=0.
        too_short = stats_df["n"] < 2
        slope = np.where(too_short, 0.0, slope)
        intercept = np.where(too_short, stats_df["first_value"].fillna(0).values, intercept)
        r_value = np.where(too_short, 0.0, r_value)
        p_value = np.where(too_short, 1.0, p_value)
        trend_label = np.where(too_short, "➡️ Stable", trend_label)

        df_trends = pd.DataFrame({
            self.id_col: stats_df.index,
            "slope": np.round(slope, 3),
            "intercept": np.round(intercept, 2),
            "r_value": np.round(r_value, 3),
            "p_value": np.round(p_value, 4),
            "n_periods": stats_df["n"].astype(int).values,
            "trend_label": trend_label,
            "first_value": np.round(stats_df["first_value"].values, 2),
            "last_value": np.round(stats_df["last_value"].values, 2),
            "delta": np.round((stats_df["last_value"] - stats_df["first_value"]).values, 2),
        }).reset_index(drop=True)

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
