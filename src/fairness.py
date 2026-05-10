"""
Module d'audit fairness — détection de biais par groupe sensible.

Implémente sans dépendance externe (pas de fairlearn) :
- Demographic parity : taux de réussite prédit par groupe
- Equalized odds    : TPR (true positive rate) et FPR (false positive rate) par groupe
- Group-wise quality : accuracy, F1, MAE par groupe

Utilisation typique :
    >>> df_audit = X_test.copy()
    >>> df_audit['y_true'] = y_test
    >>> df_audit['y_pred'] = model.predict(X_test)
    >>> audit = audit_fairness(df_audit, sensitive_col='genre',
    ...                        y_true_clf='y_true', y_pred_clf='y_pred')
    >>> print(format_fairness_report(audit))

Pour un déploiement scolaire, ce module doit être exécuté à chaque
réentraînement et les écarts entre groupes doivent être documentés.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom > 0 else 0.0


def compute_group_metrics_clf(df: pd.DataFrame, sensitive_col: str,
                              y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """
    Calcule par groupe : taille, taux positif réel, taux positif prédit,
    TPR, FPR, accuracy, F1.
    """
    rows = []
    for group_value, sub in df.groupby(sensitive_col):
        y_true = sub[y_true_col].astype(int).values
        y_pred = sub[y_pred_col].astype(int).values
        n = len(sub)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        precision = _safe_div(tp, tp + fp)
        recall = tpr
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append({
            'group': group_value,
            'n': n,
            'positive_rate_true': _safe_div(int((y_true == 1).sum()), n),
            'positive_rate_pred': _safe_div(int((y_pred == 1).sum()), n),
            'tpr': tpr,
            'fpr': fpr,
            'accuracy': _safe_div(tp + tn, n),
            'f1': f1,
        })
    return pd.DataFrame(rows).set_index('group').round(4)


def compute_group_metrics_reg(df: pd.DataFrame, sensitive_col: str,
                              y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """Métriques régression par groupe : MAE, biais moyen (mean residual)."""
    rows = []
    for group_value, sub in df.groupby(sensitive_col):
        y_true = sub[y_true_col].astype(float).values
        y_pred = sub[y_pred_col].astype(float).values
        residuals = y_pred - y_true
        rows.append({
            'group': group_value,
            'n': len(sub),
            'mae': float(np.mean(np.abs(residuals))) if len(residuals) else 0.0,
            'mean_bias': float(np.mean(residuals)) if len(residuals) else 0.0,
            'mean_pred': float(np.mean(y_pred)) if len(y_pred) else 0.0,
            'mean_true': float(np.mean(y_true)) if len(y_true) else 0.0,
        })
    return pd.DataFrame(rows).set_index('group').round(4)


def demographic_parity_difference(df_metrics: pd.DataFrame) -> float:
    """
    Écart maximal de taux de prédiction positive entre groupes.
    0 = parité parfaite ; > 0.1 = écart significatif à investiguer.
    """
    if 'positive_rate_pred' not in df_metrics.columns or len(df_metrics) < 2:
        return 0.0
    rates = df_metrics['positive_rate_pred']
    return float(rates.max() - rates.min())


def equalized_odds_difference(df_metrics: pd.DataFrame) -> Dict[str, float]:
    """Écart maximal de TPR et FPR entre groupes."""
    if not {'tpr', 'fpr'}.issubset(df_metrics.columns) or len(df_metrics) < 2:
        return {'tpr_diff': 0.0, 'fpr_diff': 0.0}
    return {
        'tpr_diff': float(df_metrics['tpr'].max() - df_metrics['tpr'].min()),
        'fpr_diff': float(df_metrics['fpr'].max() - df_metrics['fpr'].min()),
    }


def audit_fairness(df: pd.DataFrame, sensitive_col: str,
                   y_true_clf: Optional[str] = None,
                   y_pred_clf: Optional[str] = None,
                   y_true_reg: Optional[str] = None,
                   y_pred_reg: Optional[str] = None,
                   min_group_size: int = 5) -> Dict[str, Any]:
    """
    Audit complet pour une variable sensible.

    Parameters
    ----------
    df : DataFrame contenant la colonne sensible et les prédictions/vérités.
    sensitive_col : nom de la colonne sensible (genre, classe, etablissement…).
    y_true_clf, y_pred_clf : colonnes binaires si audit classification.
    y_true_reg, y_pred_reg : colonnes numériques si audit régression.
    min_group_size : groupes plus petits sont ignorés et listés dans 'skipped'.

    Returns
    -------
    Dict avec clés : sensitive_col, group_sizes, classification (optionnel),
    regression (optionnel), summary, skipped.
    """
    if sensitive_col not in df.columns:
        raise ValueError(f"Colonne sensible '{sensitive_col}' absente du DataFrame.")

    group_sizes = df[sensitive_col].value_counts().to_dict()
    skipped = [g for g, n in group_sizes.items() if n < min_group_size]
    df_filtered = df[~df[sensitive_col].isin(skipped)] if skipped else df

    result: Dict[str, Any] = {
        'sensitive_col': sensitive_col,
        'group_sizes': group_sizes,
        'skipped': skipped,
        'min_group_size': min_group_size,
    }

    if y_true_clf and y_pred_clf and y_true_clf in df.columns and y_pred_clf in df.columns:
        clf_metrics = compute_group_metrics_clf(df_filtered, sensitive_col, y_true_clf, y_pred_clf)
        result['classification'] = {
            'per_group': clf_metrics,
            'demographic_parity_diff': demographic_parity_difference(clf_metrics),
            'equalized_odds_diff': equalized_odds_difference(clf_metrics),
        }

    if y_true_reg and y_pred_reg and y_true_reg in df.columns and y_pred_reg in df.columns:
        reg_metrics = compute_group_metrics_reg(df_filtered, sensitive_col, y_true_reg, y_pred_reg)
        result['regression'] = {
            'per_group': reg_metrics,
            'mae_diff': float(reg_metrics['mae'].max() - reg_metrics['mae'].min()),
            'bias_diff': float(reg_metrics['mean_bias'].max() - reg_metrics['mean_bias'].min()),
        }

    # Niveau d'alerte global sur la base de seuils communs (Aequitas-like).
    alert_level = 'OK'
    if 'classification' in result:
        dp = result['classification']['demographic_parity_diff']
        eo = result['classification']['equalized_odds_diff']
        if dp > 0.2 or eo['tpr_diff'] > 0.2 or eo['fpr_diff'] > 0.2:
            alert_level = 'CRITIQUE'
        elif dp > 0.1 or eo['tpr_diff'] > 0.1 or eo['fpr_diff'] > 0.1:
            alert_level = 'ATTENTION'
    if 'regression' in result and result['regression']['bias_diff'] > 1.0:
        alert_level = 'ATTENTION' if alert_level == 'OK' else alert_level
    result['alert_level'] = alert_level

    return result


def format_fairness_report(audit: Dict[str, Any]) -> str:
    """Formate un audit pour affichage console / log."""
    lines: List[str] = []
    lines.append(f"=== Audit fairness : {audit['sensitive_col']} ===")
    lines.append(f"Niveau d'alerte : {audit['alert_level']}")
    lines.append(f"Tailles des groupes : {audit['group_sizes']}")
    if audit['skipped']:
        lines.append(f"Groupes ignorés (< {audit['min_group_size']} élèves) : {audit['skipped']}")

    if 'classification' in audit:
        c = audit['classification']
        lines.append("")
        lines.append("-- Classification --")
        lines.append(f"Demographic parity diff : {c['demographic_parity_diff']:.3f}")
        lines.append(f"Equalized odds — TPR diff : {c['equalized_odds_diff']['tpr_diff']:.3f}, "
                     f"FPR diff : {c['equalized_odds_diff']['fpr_diff']:.3f}")
        lines.append("Détail par groupe :")
        lines.append(c['per_group'].to_string())

    if 'regression' in audit:
        r = audit['regression']
        lines.append("")
        lines.append("-- Régression --")
        lines.append(f"MAE diff : {r['mae_diff']:.3f}, Biais moyen diff : {r['bias_diff']:.3f}")
        lines.append("Détail par groupe :")
        lines.append(r['per_group'].to_string())

    return "\n".join(lines)
