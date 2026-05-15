"""
Page Streamlit dédiée à l'audit fairness (équité).

Permet aux directions d'établissement de vérifier que le modèle ne traite
pas différemment certains groupes (genre, classe, établissement…).
"""
import os
import sys
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import COLS_TO_DROP, TARGET_CLF, TARGET_REG
from src.fairness import (
    audit_fairness, format_fairness_report,
    compare_privacy_performance, get_privacy_preserving_features, SENSITIVE_COLS,
)
from app.utils_st import _get

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("⚖️ Audit Fairness")
st.markdown(
    "Vérifie que le modèle ne défavorise pas systématiquement un groupe "
    "(genre, classe, établissement). Trois écarts sont mesurés :"
)
st.markdown(
    "- **Parité démographique** : différence des taux de réussite prédits entre groupes\n"
    "- **Égalité des chances (TPR diff)** : différence du taux de vrais positifs\n"
    "- **Égalité de traitement (FPR diff)** : différence du taux de faux positifs\n\n"
    "Seuils d'alerte : **OK** < 0.10 ≤ **ATTENTION** < 0.20 ≤ **CRITIQUE**"
)

df_feat = _get("df_feat")
models = _get("models")

if df_feat is None or models is None:
    st.warning("⚠️ Entraînez d'abord un modèle (Page 3 — Modélisation).")
    st.stop()

# Détection des colonnes sensibles disponibles.
sensitive_candidates = [c for c in ('genre', 'sexe', 'classe', 'etablissement', 'niveau')
                        if c in df_feat.columns]
if not sensitive_candidates:
    st.error(
        "Aucune colonne sensible détectée (genre/sexe/classe/etablissement). "
        "L'audit fairness nécessite au moins une variable de groupe."
    )
    st.stop()

sensitive_col = st.selectbox(
    "Variable sensible à auditer",
    options=sensitive_candidates,
    help="Le modèle sera évalué groupe par groupe sur cette variable.",
)

min_group_size = st.slider("Taille minimale d'un groupe pour être inclus",
                           min_value=1, max_value=30, value=5)

if st.button("Lancer l'audit", type="primary"):
    with st.spinner("Calcul des métriques par groupe…"):
        # Reconstruire X (sans cibles ni colonnes d'identité non-prédictives).
        cols_drop = [c for c in COLS_TO_DROP if c in df_feat.columns]
        # Garde la colonne sensible même si elle est dans COLS_TO_DROP, on en a besoin.
        cols_drop = [c for c in cols_drop if c != sensitive_col]
        targets = [c for c in (TARGET_REG, TARGET_CLF) if c in df_feat.columns]
        X = df_feat.drop(columns=cols_drop + targets, errors='ignore')

        model_clf = models.get('clf') if isinstance(models, dict) else getattr(models, 'best_model_clf', None)
        model_reg = models.get('reg') if isinstance(models, dict) else getattr(models, 'best_model_reg', None)

        df_audit = X.copy()
        if TARGET_CLF in df_feat.columns:
            df_audit[TARGET_CLF] = df_feat[TARGET_CLF].values
        if TARGET_REG in df_feat.columns:
            df_audit[TARGET_REG] = df_feat[TARGET_REG].values

        # Prédictions (drop la colonne sensible avant predict si elle n'était pas dans X au train).
        X_for_pred = X.drop(columns=[sensitive_col], errors='ignore')
        try:
            if model_clf is not None and TARGET_CLF in df_audit.columns:
                df_audit['y_pred_clf'] = model_clf.predict(X_for_pred)
            if model_reg is not None and TARGET_REG in df_audit.columns:
                df_audit['y_pred_reg'] = model_reg.predict(X_for_pred)
        except Exception as e:
            st.error(f"Impossible de prédire avec ce modèle : {e}")
            st.stop()

        audit = audit_fairness(
            df_audit, sensitive_col=sensitive_col,
            y_true_clf=TARGET_CLF if 'y_pred_clf' in df_audit.columns else None,
            y_pred_clf='y_pred_clf' if 'y_pred_clf' in df_audit.columns else None,
            y_true_reg=TARGET_REG if 'y_pred_reg' in df_audit.columns else None,
            y_pred_reg='y_pred_reg' if 'y_pred_reg' in df_audit.columns else None,
            min_group_size=min_group_size,
        )

    # ── Affichage ──────────────────────────────────────────────────────
    alert_color = {'OK': '🟢', 'ATTENTION': '🟡', 'CRITIQUE': '🔴'}.get(audit['alert_level'], '⚪')
    st.metric("Niveau d'alerte global", f"{alert_color} {audit['alert_level']}")

    if audit['skipped']:
        st.info(f"Groupes ignorés (< {min_group_size} élèves) : {audit['skipped']}")

    if 'classification' in audit:
        st.subheader("Classification (réussite/échec)")
        c = audit['classification']
        col1, col2, col3 = st.columns(3)
        col1.metric("Parité démographique", f"{c['demographic_parity_diff']:.3f}")
        col2.metric("Écart TPR (vrais positifs)", f"{c['equalized_odds_diff']['tpr_diff']:.3f}")
        col3.metric("Écart FPR (faux positifs)", f"{c['equalized_odds_diff']['fpr_diff']:.3f}")

        st.dataframe(c['per_group'], use_container_width=True)

        per = c['per_group'].reset_index()
        fig = px.bar(
            per.melt(id_vars='group',
                     value_vars=['positive_rate_pred', 'positive_rate_true'],
                     var_name='Type', value_name='Taux'),
            x='group', y='Taux', color='Type', barmode='group',
            title="Taux de réussite prédit vs réel par groupe",
        )
        st.plotly_chart(fig, use_container_width=True)

    if 'regression' in audit:
        st.subheader("Régression (note prédite)")
        r = audit['regression']
        col1, col2 = st.columns(2)
        col1.metric("Écart MAE", f"{r['mae_diff']:.3f}")
        col2.metric("Écart de biais moyen", f"{r['bias_diff']:.3f}")
        st.dataframe(r['per_group'], use_container_width=True)

    with st.expander("Rapport texte complet"):
        st.code(format_fairness_report(audit), language='text')

# ── Mode Privacy-First ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔒 Mode Privacy-First")
st.markdown(
    "Inspiré du **Case 3** de Muresan et al. (2026) : un modèle entraîné *sans* "
    "données démographiques sensibles (genre, âge, situation de handicap…) perd "
    "typiquement **< 1.9 % de F1** par rapport à un modèle intégrant toutes les features.\n\n"
    "Ce test compare les performances via une validation croisée 5-fold avec "
    "régression logistique standardisée."
)

cols_sens_present = [c for c in SENSITIVE_COLS if c in df_feat.columns]
if not cols_sens_present:
    st.info(
        "Aucune colonne sensible détectée dans les données actuelles "
        f"(colonnes surveillées : {', '.join(SENSITIVE_COLS[:6])}…). "
        "Si des données réelles contenant ces colonnes sont chargées, "
        "le test s'activera automatiquement."
    )
else:
    st.markdown(f"**Colonnes sensibles détectées** : `{', '.join(cols_sens_present)}`")

    if st.button("🔍 Lancer la comparaison privacy-first", key="btn_privacy"):
        with st.spinner("Validation croisée en cours…"):
            cols_drop = [c for c in COLS_TO_DROP if c in df_feat.columns]
            targets = [c for c in (TARGET_REG, TARGET_CLF) if c in df_feat.columns]
            X_full = df_feat.drop(columns=cols_drop + targets, errors='ignore')
            y_clf = df_feat[TARGET_CLF] if TARGET_CLF in df_feat.columns else None

        if y_clf is None or len(y_clf.unique()) < 2:
            st.warning("Cible de classification manquante ou constante — test impossible.")
        else:
            try:
                result = compare_privacy_performance(X_full, y_clf, cv=5)

                colA, colB, colC = st.columns(3)
                colA.metric("F1 (toutes features)", f"{result['f1_full']:.3f}")
                colB.metric(
                    "F1 (sans données sensibles)",
                    f"{result['f1_privacy']:.3f}",
                    delta=f"{result['delta_f1']:+.3f}",
                )
                colC.metric(
                    "Variation relative",
                    f"{result['delta_pct']:+.1f} %",
                    delta_color="inverse",
                )

                if result['privacy_cost_acceptable']:
                    st.success(
                        f"✅ Variation de F1 acceptable ({result['delta_pct']:+.1f} % < 5 %). "
                        "Le mode privacy-first est recommandé pour ce jeu de données."
                    )
                else:
                    st.warning(
                        f"⚠️ Perte de F1 significative ({result['delta_pct']:+.1f} %). "
                        "L'utilisation des colonnes sensibles est justifiée par la performance."
                    )

                removed_str = (
                    ', '.join(f'`{c}`' for c in result['cols_removed'])
                    if result['cols_removed'] else '_aucune_'
                )
                st.caption(f"Colonnes retirées : {removed_str} — {result['reference']}")

                with st.expander("Détails complets"):
                    st.json({
                        "f1_full": result['f1_full'],
                        "f1_privacy": result['f1_privacy'],
                        "accuracy_full": result['accuracy_full'],
                        "accuracy_privacy": result['accuracy_privacy'],
                        "colonnes_supprimees": result['cols_removed'],
                        "reference": result['reference'],
                    })
            except Exception as e:
                st.error(f"Erreur lors de la comparaison : {e}")
