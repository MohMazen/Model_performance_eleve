"""
Système de Recommandations Personnalisées par IA.
Transforme les analyses SHAP en actions concrètes et actionnables.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

ACTIONABLE_FACTORS: Dict[str, Dict[str, Any]] = {
    "heures_etude_soir": {"label": "Temps d'étude quotidien", "direction": "increase", "unit": "h",
        "recommendation": "Augmenter progressivement le temps d'étude de 30 min/jour.", "max_realistic_change": 2.0, "icon": "📚"},
    "heures_sommeil": {"label": "Durée de sommeil", "direction": "increase", "unit": "h",
        "recommendation": "Viser 8h de sommeil minimum avec une heure de coucher régulière.", "max_realistic_change": 2.0, "icon": "😴"},
    "stress_personnel": {"label": "Stress personnel", "direction": "decrease", "unit": "/4",
        "recommendation": "Exercices de relaxation et suivi régulier.", "max_realistic_change": -2, "icon": "🧘"},
    "heures_jeux_video": {"label": "Temps de jeux vidéo", "direction": "decrease", "unit": "h",
        "recommendation": "Réduire le temps de jeu et le remplacer par du sport.", "max_realistic_change": -2.0, "icon": "🎮"},
    "heures_reseaux_sociaux": {"label": "Réseaux sociaux", "direction": "decrease", "unit": "h",
        "recommendation": "Limiter à 1h/jour, désactiver les notifications.", "max_realistic_change": -2.0, "icon": "📱"},
    "heures_streaming": {"label": "Temps de streaming", "direction": "decrease", "unit": "h",
        "recommendation": "Limiter le streaming à 1h/jour.", "max_realistic_change": -2.0, "icon": "📺"},
    "perseverance": {"label": "Persévérance", "direction": "increase", "unit": "/5",
        "recommendation": "Fixer des objectifs intermédiaires et célébrer les progrès.", "max_realistic_change": 2, "icon": "💪"},
    "organisation": {"label": "Organisation", "direction": "increase", "unit": "/10",
        "recommendation": "Utiliser un planning hebdomadaire.", "max_realistic_change": 3, "icon": "📋"},
    "confiance_soi": {"label": "Confiance en soi", "direction": "increase", "unit": "/10",
        "recommendation": "Valoriser les réussites et encourager les initiatives.", "max_realistic_change": 3, "icon": "⭐"},
    "estime_soi": {"label": "Estime de soi", "direction": "increase", "unit": "/10",
        "recommendation": "Travail sur l'image de soi positive.", "max_realistic_change": 3, "icon": "🌟"},
    "qualite_sommeil": {"label": "Qualité du sommeil", "direction": "increase", "unit": "/10",
        "recommendation": "Chambre sombre, pas d'écran 1h avant, horaires réguliers.", "max_realistic_change": 3, "icon": "🌙"},
    "heures_activite_physique": {"label": "Activité physique", "direction": "increase", "unit": "h",
        "recommendation": "Pratiquer au moins 1h d'activité physique par jour.", "max_realistic_change": 2.0, "icon": "🏃"},
}

NON_ACTIONABLE = {"age", "classe", "etablissement", "duree_trajet_ar_min", "nom", "prenom", "adresse", "nb_repas"}

# Disclaimer obligatoire : les valeurs SHAP mesurent des corrélations dans les
# données d'entraînement, pas des relations causales vérifiées. Modifier un
# facteur ne garantit pas le changement de résultat prédit.
CORRELATION_DISCLAIMER = (
    "⚠️ **Avertissement méthodologique** : les recommandations ci-dessous sont "
    "fondées sur des corrélations statistiques (valeurs SHAP), non sur des "
    "relations causales établies. Agir sur un facteur n'entraîne pas "
    "mécaniquement une amélioration des résultats scolaires. Ces pistes doivent "
    "être interprétées par un professionnel de l'éducation avant toute action."
)


class RecommendationEngine:
    """Moteur de recommandations personnalisées basé sur SHAP."""

    def __init__(self, actionable_factors: Optional[Dict] = None) -> None:
        self.actionable_factors = actionable_factors or ACTIONABLE_FACTORS

    def get_individual_recommendations(self, student_row: pd.Series, shap_feature_names: List[str],
                                        shap_values: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """Top-N recommandations actionnables pour un élève."""
        recommendations = []
        for i, fname in enumerate(shap_feature_names):
            base_name = fname.split("_oui")[0].split("_non")[0]
            if base_name not in self.actionable_factors or base_name in NON_ACTIONABLE:
                continue
            info = self.actionable_factors[base_name]
            shap_impact = float(shap_values[i])
            is_improvable = ((info["direction"] == "increase" and shap_impact < 0)
                             or (info["direction"] == "decrease" and shap_impact > 0))
            if not is_improvable and abs(shap_impact) < 0.1:
                continue
            current_value = float(student_row[base_name]) if base_name in student_row.index else None
            suggested = round(current_value + info["max_realistic_change"], 1) if current_value is not None else None
            recommendations.append({
                "factor": base_name, "label": info["label"], "icon": info["icon"], "unit": info.get("unit", ""),
                "current_value": current_value, "shap_impact": round(shap_impact, 3), "abs_impact": abs(shap_impact),
                "direction": info["direction"], "recommendation": info["recommendation"],
                "suggested_value": suggested, "is_improvable": is_improvable,
            })
        recommendations.sort(key=lambda r: (not r["is_improvable"], -r["abs_impact"]))
        return recommendations[:top_n]

    def simulate_intervention(self, student_row: pd.Series, factor: str, new_value: float,
                               model: Any, feature_columns: List[str],
                               cols_to_drop: Optional[List[str]] = None,
                               target_reg: str = "note_moyenne", target_clf: str = "reussite") -> Dict[str, Any]:
        """Simule l'effet d'un changement de facteur sur la prédiction."""
        from src.features import add_advanced_features
        row_orig = student_row.to_frame().T.copy()
        row_mod = row_orig.copy()
        if factor in row_mod.columns:
            row_mod[factor] = new_value
        row_mod = add_advanced_features(row_mod)

        def _prep(row_df):
            X = row_df.drop(columns=[c for c in (cols_to_drop or []) if c in row_df.columns]
                            + [c for c in [target_reg, target_clf] if c in row_df.columns], errors="ignore")
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            return X[feature_columns]

        pred_orig = float(model.predict(_prep(row_orig))[0])
        pred_new = float(model.predict(_prep(row_mod))[0])
        return {"factor": factor, "factor_label": self.actionable_factors.get(factor, {}).get("label", factor),
                "old_value": float(student_row.get(factor, 0)), "new_value": new_value,
                "original_prediction": round(pred_orig, 2), "new_prediction": round(pred_new, 2),
                "delta": round(pred_new - pred_orig, 2)}

    def generate_action_plan(self, student_name: str, recommendations: List[Dict],
                              note_actuelle: Optional[float] = None, note_predite: Optional[float] = None) -> str:
        """Génère une fiche individuelle Markdown."""
        lines = [f"# 📋 Plan d'Action — {student_name}", f"*{datetime.now().strftime('%d/%m/%Y')}*", ""]
        if note_actuelle is not None:
            lines.append(f"**Note actuelle** : {note_actuelle:.2f}/20")
        if note_predite is not None:
            lines.append(f"**Note prédite** : {note_predite:.2f}/20")
        lines += ["", "---", "",
                  "> ⚠️ Ces recommandations sont fondées sur des corrélations statistiques "
                  "(valeurs SHAP), non sur des relations causales. Elles doivent être "
                  "interprétées par un professionnel de l'éducation avant toute action.",
                  "", "## Recommandations"]
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"### {i}. {rec.get('icon', '📌')} {rec['label']}")
            lines.append(f"**Action** : {rec['recommendation']}")
            if rec.get("current_value") is not None and rec.get("suggested_value") is not None:
                lines.append(f"- Actuel : `{rec['current_value']}{rec.get('unit', '')}` → Objectif : `{rec['suggested_value']}{rec.get('unit', '')}`")
            lines.append(f"- Corrélation SHAP : `{rec.get('shap_impact', 0):+.3f}` *(corrélation, pas causalité)*")
            lines.append("")
        lines.append("---\n*EduStats v3.0*")
        return "\n".join(lines)
