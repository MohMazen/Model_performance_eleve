"""
Système d'Alerte Précoce (Early Warning System).
Détection proactive des élèves à risque d'échec via scoring composite.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Règles métier éducatives ──────────────────────────────────────────────
BUSINESS_RULES: List[Dict[str, Any]] = [
    {
        "name": "sommeil_insuffisant",
        "label": "Sommeil insuffisant (< 6h)",
        "column": "heures_sommeil",
        "condition": lambda v: v < 6,
        "weight": 15,
        "recommendation": "Viser au minimum 7-8h de sommeil par nuit.",
    },
    {
        "name": "stress_eleve",
        "label": "Stress personnel élevé (≥ 3/4)",
        "column": "stress_personnel",
        "condition": lambda v: v >= 3,
        "weight": 12,
        "recommendation": "Proposer un accompagnement psychologique ou des ateliers de gestion du stress.",
    },
    {
        "name": "ecrans_excessifs",
        "label": "Temps d'écrans excessif (> 4h/j)",
        "column": "temps_ecrans_total",
        "condition": lambda v: v > 4,
        "weight": 10,
        "recommendation": "Limiter le temps d'écran récréatif à 2h maximum par jour.",
    },
    {
        "name": "etude_faible",
        "label": "Temps d'étude très faible (< 1h)",
        "column": "heures_etude_soir",
        "condition": lambda v: v < 1,
        "weight": 12,
        "recommendation": "Mettre en place un créneau d'étude quotidien de 1h30 minimum.",
    },
    {
        "name": "motivation_basse",
        "label": "Indice de motivation faible (< 4)",
        "column": "indice_motivation",
        "condition": lambda v: v < 4,
        "weight": 10,
        "recommendation": "Renforcer l'engagement via des projets pédagogiques motivants.",
    },
    {
        "name": "equilibre_faible",
        "label": "Déséquilibre vie / études",
        "column": "score_equilibre",
        "condition": lambda v: v < 0.5,
        "weight": 8,
        "recommendation": "Rééquilibrer le planning entre repos, sport et travail scolaire.",
    },
    {
        "name": "confiance_basse",
        "label": "Confiance en soi faible (< 4)",
        "column": "confiance_soi",
        "condition": lambda v: v < 4,
        "weight": 8,
        "recommendation": "Valoriser les réussites de l'élève et encourager la prise d'initiative.",
    },
    {
        "name": "pas_de_sport",
        "label": "Aucune activité sportive",
        "column": "activite_sportive",
        "condition": lambda v: str(v).lower() in ("non", "0", "false", ""),
        "weight": 5,
        "recommendation": "Encourager la pratique d'une activité physique régulière.",
    },
]

# Zones de risque
RISK_ZONES = [
    (0, 25, "🟢 Serein", "serein"),
    (25, 50, "🟡 Vigilance", "vigilance"),
    (50, 75, "🟠 Alerte", "alerte"),
    (75, 101, "🔴 Critique", "critique"),
]


class EarlyWarningSystem:
    """Moteur d'alertes précoces multi-critères."""

    def __init__(
        self,
        ml_weight: float = 0.60,
        rules_weight: float = 0.40,
        rules: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.ml_weight = ml_weight
        self.rules_weight = rules_weight
        self.rules = rules if rules is not None else BUSINESS_RULES

    # ── Score ML ──────────────────────────────────────────────────────────
    @staticmethod
    def _ml_risk_score(proba_echec: float) -> float:
        """Convertit la probabilité d'échec ML [0-1] en score [0-100]."""
        return float(np.clip(proba_echec * 100, 0, 100))

    # ── Score Règles Métier ───────────────────────────────────────────────
    def _rules_risk_score(self, row: pd.Series) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Évalue les règles métier sur une ligne du DataFrame.
        Retourne (score_normalise_0_100, liste_des_facteurs_déclenchés).
        """
        triggered: List[Dict[str, Any]] = []
        total_weight = sum(r["weight"] for r in self.rules)

        for rule in self.rules:
            col = rule["column"]
            if col not in row.index:
                continue
            try:
                value = row[col]
                if pd.isna(value):
                    continue
                if rule["condition"](value):
                    triggered.append(
                        {
                            "name": rule["name"],
                            "label": rule["label"],
                            "value": value,
                            "weight": rule["weight"],
                            "recommendation": rule["recommendation"],
                        }
                    )
            except (TypeError, ValueError):
                continue

        score_brut = sum(f["weight"] for f in triggered)
        score_norm = (score_brut / total_weight) * 100 if total_weight > 0 else 0
        return float(np.clip(score_norm, 0, 100)), triggered

    # ── Score Composite ───────────────────────────────────────────────────
    def compute_risk_score(
        self, row: pd.Series, proba_echec: float
    ) -> Dict[str, Any]:
        """
        Calcule le score de risque composite pour un élève.

        Returns
        -------
        dict avec clés : risk_score, risk_zone, risk_zone_label,
                         ml_score, rules_score, triggered_rules
        """
        ml_score = self._ml_risk_score(proba_echec)
        rules_score, triggered = self._rules_risk_score(row)

        composite = self.ml_weight * ml_score + self.rules_weight * rules_score
        composite = float(np.clip(composite, 0, 100))

        zone_label, zone_key = "Inconnu", "inconnu"
        for low, high, label, key in RISK_ZONES:
            if low <= composite < high:
                zone_label, zone_key = label, key
                break

        return {
            "risk_score": round(composite, 1),
            "risk_zone": zone_key,
            "risk_zone_label": zone_label,
            "ml_score": round(ml_score, 1),
            "rules_score": round(rules_score, 1),
            "triggered_rules": triggered,
        }

    # ── Classification de la cohorte ──────────────────────────────────────
    def classify_cohort(
        self,
        df: pd.DataFrame,
        model_clf: Any,
        feature_columns: List[str],
        cols_to_drop: Optional[List[str]] = None,
        target_reg: str = "note_moyenne",
        target_clf: str = "reussite",
    ) -> pd.DataFrame:
        """
        Ajoute les colonnes de risque à l'ensemble du DataFrame.
        """
        logger.info("Classification de la cohorte par niveau de risque…")

        # Préparer X
        cols_drop = cols_to_drop or []
        X = df.drop(
            columns=[c for c in cols_drop if c in df.columns]
            + [c for c in [target_reg, target_clf] if c in df.columns],
            errors="ignore",
        )
        # S'assurer qu'on utilise les bonnes colonnes
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]

        # Probabilités d'échec
        try:
            probas = model_clf.predict_proba(X)
            classes = list(model_clf.classes_)
            idx_echec = classes.index(0) if 0 in classes else 0
            proba_echec = probas[:, idx_echec]
        except Exception as e:
            logger.warning(f"Impossible d'obtenir predict_proba : {e}. Fallback sur predict.")
            preds = model_clf.predict(X)
            proba_echec = np.where(preds == 0, 0.8, 0.2)

        # Calcul ligne par ligne
        results = []
        for i, (idx, row) in enumerate(df.iterrows()):
            result = self.compute_risk_score(row, proba_echec[i])
            result["index"] = idx
            results.append(result)

        df_risk = pd.DataFrame(results).set_index("index")
        # Colonnes de risque à ajouter
        for col in ["risk_score", "risk_zone", "risk_zone_label", "ml_score", "rules_score"]:
            df[col] = df_risk[col]

        # Stocker les règles déclenchées en tant que liste sérialisée
        df["triggered_rules"] = df_risk["triggered_rules"]

        logger.info(
            f"Cohorte classifiée : "
            f"{(df['risk_zone'] == 'critique').sum()} critiques, "
            f"{(df['risk_zone'] == 'alerte').sum()} alertes, "
            f"{(df['risk_zone'] == 'vigilance').sum()} vigilances, "
            f"{(df['risk_zone'] == 'serein').sum()} sereins."
        )
        return df

    # ── Génération d'alertes ──────────────────────────────────────────────
    @staticmethod
    def generate_alerts(
        df: pd.DataFrame, seuil_alerte: float = 50.0
    ) -> pd.DataFrame:
        """
        Filtre les élèves au-dessus du seuil de risque et retourne
        un DataFrame trié par risk_score décroissant.
        """
        if "risk_score" not in df.columns:
            logger.warning("Aucune colonne 'risk_score' trouvée. Lancez classify_cohort d'abord.")
            return pd.DataFrame()

        alertes = df[df["risk_score"] >= seuil_alerte].copy()
        alertes = alertes.sort_values("risk_score", ascending=False)
        logger.info(f"{len(alertes)} alertes générées (seuil ≥ {seuil_alerte}).")
        return alertes
