
import os
import sys
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import COLS_TO_DROP, TARGET_CLF, TARGET_REG
from src.early_warning import EarlyWarningSystem, RISK_ZONES
from app.utils_st import _get, _set

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("🚨 Alertes Précoces")
st.markdown("Détection proactive des élèves à risque d'échec scolaire.")

model_clf = _get("model_clf")
df_feat = _get("df_feat")
feature_columns = _get("feature_columns")

if model_clf is None or df_feat is None or feature_columns is None:
    st.warning("⚠️ Entraînez d'abord les modèles (Page 3).")
    st.stop()

# Paramètres
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    seuil_alerte = st.slider("Seuil d'alerte (score de risque)", 20, 80, 50, 5)
with col_p2:
    ml_weight = st.slider("Poids ML (%)", 30, 80, 60, 5) / 100
with col_p3:
    rules_weight = 1.0 - ml_weight
    st.metric("Poids Règles Métier", f"{rules_weight*100:.0f}%")

if st.button("🔍 Analyser les risques", type="primary"):
    with st.spinner("Calcul des scores de risque…"):
        ews = EarlyWarningSystem(ml_weight=ml_weight, rules_weight=rules_weight)
        target_reg = _get("target_reg", TARGET_REG)
        df_risk = ews.classify_cohort(
            df_feat.copy(), model_clf, feature_columns,
            cols_to_drop=COLS_TO_DROP, target_reg=target_reg, target_clf=TARGET_CLF
        )
        _set("df_risk", df_risk)
        st.success("✅ Analyse de risque terminée.")

df_risk = _get("df_risk")
if df_risk is None:
    st.info("Cliquez sur 'Analyser les risques' pour démarrer.")
    st.stop()

# Vue d'ensemble
st.markdown("---")
st.subheader("📊 Vue d'ensemble")

zone_counts = df_risk["risk_zone"].value_counts()
zone_map = {"serein": "🟢 Serein", "vigilance": "🟡 Vigilance", "alerte": "🟠 Alerte", "critique": "🔴 Critique"}
zone_colors = {"🟢 Serein": "#4caf50", "🟡 Vigilance": "#ffc107", "🟠 Alerte": "#ff9800", "🔴 Critique": "#f44336"}

col1, col2, col3, col4 = st.columns(4)
for i, (zone_key, label) in enumerate(zone_map.items()):
    count = int(zone_counts.get(zone_key, 0))
    with [col1, col2, col3, col4][i]:
        st.metric(label, count)

# Pie chart
fig_pie = px.pie(
    names=[zone_map.get(z, z) for z in zone_counts.index],
    values=zone_counts.values,
    color=[zone_map.get(z, z) for z in zone_counts.index],
    color_discrete_map=zone_colors,
    title="Répartition par zone de risque",
    hole=0.4,
)
st.plotly_chart(fig_pie, use_container_width=True)

# Alertes
st.markdown("---")
st.subheader(f"⚠️ Élèves en alerte (score ≥ {seuil_alerte})")

alertes = EarlyWarningSystem.generate_alerts(df_risk, seuil_alerte)

if alertes.empty:
    st.success(f"🎉 Aucun élève au-dessus du seuil de risque ({seuil_alerte}).")
else:
    st.warning(f"**{len(alertes)} élève(s)** nécessitent une attention particulière.")

    # Colonnes d'affichage
    display_cols = []
    for c in ["nom", "prenom", "classe", "risk_score", "risk_zone_label", "ml_score", "rules_score"]:
        if c in alertes.columns:
            display_cols.append(c)

    target_reg = _get("target_reg", TARGET_REG)
    if target_reg in alertes.columns:
        display_cols.append(target_reg)

    st.dataframe(
        alertes[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Score Risque", min_value=0, max_value=100, format="%.0f"),
        },
    )

    # Bar chart horizontal
    chart_df = alertes.head(20).copy()
    name_col = "nom" if "nom" in chart_df.columns else chart_df.index.astype(str)
    if "nom" in chart_df.columns:
        chart_df["label"] = chart_df["nom"]
    else:
        chart_df["label"] = [f"Élève {i}" for i in range(len(chart_df))]

    fig_bar = px.bar(
        chart_df, x="risk_score", y="label", orientation="h",
        color="risk_zone_label",
        color_discrete_map={v: zone_colors.get(v, "#999") for v in zone_map.values()},
        title="Top 20 — Score de risque",
        labels={"risk_score": "Score de risque", "label": "Élève"},
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    # Détail par élève
    st.markdown("---")
    st.subheader("🔎 Détail par élève")
    eleve_options = alertes["nom"].tolist() if "nom" in alertes.columns else [f"Élève {i}" for i in range(len(alertes))]
    selected_eleve = st.selectbox("Sélectionnez un élève", eleve_options)

    if selected_eleve:
        if "nom" in alertes.columns:
            eleve_row = alertes[alertes["nom"] == selected_eleve].iloc[0]
        else:
            idx = eleve_options.index(selected_eleve)
            eleve_row = alertes.iloc[idx]

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("Score de risque", f"{eleve_row['risk_score']:.0f}/100")
            st.metric("Zone", eleve_row["risk_zone_label"])
            st.metric("Score ML", f"{eleve_row['ml_score']:.0f}")
            st.metric("Score Règles", f"{eleve_row['rules_score']:.0f}")

        with col_d2:
            triggered = eleve_row.get("triggered_rules", [])
            if isinstance(triggered, list) and triggered:
                st.markdown("**Facteurs de risque déclenchés :**")
                for rule in triggered:
                    st.error(f"⚠️ **{rule['label']}** (poids: {rule['weight']})")
                    st.caption(f"→ {rule['recommendation']}")
            else:
                st.info("Aucune règle métier déclenchée — le risque provient principalement du modèle ML.")

    # Export
    st.markdown("---")
    csv = alertes[display_cols].to_csv(index=False, sep=';', encoding='utf-8-sig')
    st.download_button("⬇️ Exporter les alertes (CSV)", csv, "alertes_eleves.csv", "text/csv")
