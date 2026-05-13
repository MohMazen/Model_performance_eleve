
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

from src.recommendations import RecommendationEngine, CORRELATION_DISCLAIMER
from src.explainability import get_individual_shap_values
from src.config import COLS_TO_DROP, TARGET_CLF, TARGET_REG
from app.utils_st import _get, _set

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("💡 Recommandations Personnalisées")
st.markdown("Transformez l'analyse IA en **actions concrètes** pour chaque élève.")

model_reg = _get("model_reg")
df_feat = _get("df_feat")
feature_columns = _get("feature_columns")
X_test = _get("X_test")

if model_reg is None or df_feat is None or feature_columns is None:
    st.warning("⚠️ Entraînez d'abord les modèles (Page 3).")
    st.stop()

tab_reco, tab_sim = st.tabs(["📋 Recommandations", "🔮 Simulateur What-If"])

with tab_reco:
    st.subheader("Recommandations individuelles basées sur SHAP")
    st.info(CORRELATION_DISCLAIMER)

    # Sélection de l'élève
    eleve_options = df_feat["nom"].tolist() if "nom" in df_feat.columns else [f"Élève {i}" for i in range(len(df_feat))]
    sel_idx = st.selectbox("Sélectionnez un élève", range(len(eleve_options[:100])),
                           format_func=lambda i: eleve_options[i], key="reco_eleve")

    top_n = st.slider("Nombre de recommandations", 3, 10, 5)

    if st.button("🔍 Générer les recommandations", type="primary"):
        with st.spinner("Analyse SHAP individuelle…"):
            try:
                student_row = df_feat.iloc[sel_idx]
                sample = X_test if X_test is not None and len(X_test) > 0 else df_feat.drop(
                    columns=[c for c in COLS_TO_DROP + [TARGET_REG, TARGET_CLF] if c in df_feat.columns], errors="ignore"
                )[feature_columns].head(50)

                shap_result = get_individual_shap_values(model_reg, sample, student_index=min(sel_idx, len(sample) - 1))

                if shap_result is None:
                    st.error("Impossible de calculer les valeurs SHAP pour cet élève.")
                    st.stop()

                engine = RecommendationEngine()
                recos = engine.get_individual_recommendations(
                    student_row, shap_result["feature_names"], shap_result["shap_values"], top_n=top_n
                )
                _set("current_recos", recos)
                _set("current_student_idx", sel_idx)
                _set("current_shap_result", shap_result)

                if not recos:
                    st.info("Aucune recommandation actionnable identifiée pour cet élève.")
                else:
                    st.success(f"✅ {len(recos)} recommandation(s) générée(s).")
            except Exception as e:
                st.error(f"Erreur : {e}")
                logger.error(f"Erreur recommandations : {e}", exc_info=True)

    recos = _get("current_recos")
    if recos:
        student_row = df_feat.iloc[_get("current_student_idx", 0)]
        target_reg = _get("target_reg", TARGET_REG)
        note_actuelle = student_row.get(target_reg)

        st.markdown("---")

        # Affichage en cartes
        for i, rec in enumerate(recos):
            improvable = rec.get("is_improvable", False)
            border_color = "#4caf50" if improvable else "#ff9800"
            with st.container():
                col_icon, col_content = st.columns([1, 8])
                with col_icon:
                    st.markdown(f"## {rec['icon']}")
                with col_content:
                    priority = "🔴 Prioritaire" if improvable else "🟡 Secondaire"
                    st.markdown(f"**{i+1}. {rec['label']}** — {priority}")
                    st.markdown(f"_{rec['recommendation']}_")

                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        if rec.get("current_value") is not None:
                            st.metric("Valeur actuelle", f"{rec['current_value']:.1f}{rec.get('unit', '')}")
                    with col_c2:
                        if rec.get("suggested_value") is not None:
                            st.metric("Objectif", f"{rec['suggested_value']:.1f}{rec.get('unit', '')}")
                    with col_c3:
                        st.metric("Corr. SHAP", f"{rec['shap_impact']:+.3f}",
                              help="Corrélation SHAP avec la note prédite. Ne reflète pas une relation causale.")
                st.markdown("---")

        # Graphique d'impact
        if recos:
            fig_impact = go.Figure()
            fig_impact.add_trace(go.Bar(
                y=[r["label"] for r in recos],
                x=[r["shap_impact"] for r in recos],
                orientation="h",
                marker_color=[("#f44336" if r["shap_impact"] < 0 else "#4caf50") for r in recos],
                text=[f"{r['shap_impact']:+.3f}" for r in recos],
                textposition="outside",
            ))
            fig_impact.update_layout(
                title="Corrélation SHAP des facteurs actionnables (corrélation ≠ causalité)",
                xaxis_title="Corrélation SHAP avec la note prédite",
                yaxis_title="",
                height=300 + len(recos) * 30,
            )
            st.plotly_chart(fig_impact, use_container_width=True)

        # Export du plan d'action
        engine = RecommendationEngine()
        student_name = str(student_row.get("nom", f"Élève {_get('current_student_idx', 0)}"))
        plan_md = engine.generate_action_plan(
            student_name, recos, note_actuelle=note_actuelle,
            note_predite=float(model_reg.predict(
                df_feat.iloc[_get("current_student_idx", 0):_get("current_student_idx", 0)+1].drop(
                    columns=[c for c in COLS_TO_DROP + [TARGET_REG, TARGET_CLF] if c in df_feat.columns], errors="ignore"
                )[feature_columns]
            )[0]) if feature_columns else None
        )
        st.download_button("📥 Télécharger le plan d'action (.md)", plan_md,
                           f"plan_action_{student_name}.md", "text/markdown")

with tab_sim:
    st.subheader("🔮 Simulateur What-If")
    st.markdown("Modifiez un paramètre et observez l'impact sur la prédiction en temps réel.")
    st.warning(
        "**Limite du simulateur** : les variations affichées reflètent la réponse "
        "du modèle statistique, pas un effet causal réel. Un élève qui dormirait "
        "davantage ne verrait pas nécessairement sa note progresser du même montant."
    )

    eleve_options_sim = df_feat["nom"].tolist() if "nom" in df_feat.columns else [f"Élève {i}" for i in range(len(df_feat))]
    sel_idx_sim = st.selectbox("Élève", range(len(eleve_options_sim[:100])),
                               format_func=lambda i: eleve_options_sim[i], key="sim_eleve")

    student_row_sim = df_feat.iloc[sel_idx_sim]

    # Liste des facteurs actionnables présents dans les données
    engine = RecommendationEngine()
    available_factors = [f for f in engine.actionable_factors if f in student_row_sim.index]

    if not available_factors:
        st.info("Aucun facteur actionnable disponible pour cet élève.")
    else:
        factor_labels = {f: engine.actionable_factors[f]["label"] for f in available_factors}
        sel_factor = st.selectbox("Facteur à modifier", available_factors,
                                   format_func=lambda f: f"{engine.actionable_factors[f]['icon']} {factor_labels[f]}",
                                   key="sim_factor")

        current_val = float(student_row_sim.get(sel_factor, 0))
        info = engine.actionable_factors[sel_factor]

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Valeur actuelle", f"{current_val:.1f}{info.get('unit', '')}")
        with col_s2:
            if info["direction"] == "increase":
                new_val = st.slider(f"Nouvelle valeur ({info.get('unit', '')})",
                                    float(current_val), float(current_val + abs(info["max_realistic_change"]) * 1.5),
                                    float(current_val), 0.1, key="sim_slider")
            else:
                new_val = st.slider(f"Nouvelle valeur ({info.get('unit', '')})",
                                    max(0.0, float(current_val + info["max_realistic_change"] * 1.5)),
                                    float(current_val), float(current_val), 0.1, key="sim_slider")

        if st.button("🔮 Simuler", key="btn_simulate"):
            with st.spinner("Simulation…"):
                try:
                    result = engine.simulate_intervention(
                        student_row_sim, sel_factor, new_val, model_reg,
                        feature_columns, cols_to_drop=COLS_TO_DROP,
                        target_reg=_get("target_reg", TARGET_REG), target_clf=TARGET_CLF
                    )
                    _set("sim_result", result)
                except Exception as e:
                    st.error(f"Erreur de simulation : {e}")

        sim_result = _get("sim_result")
        if sim_result:
            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Note prédite (avant)", f"{sim_result['original_prediction']:.2f}/20")
            with col_r2:
                st.metric("Note prédite (après)", f"{sim_result['new_prediction']:.2f}/20",
                          delta=f"{sim_result['delta']:+.2f}")
            with col_r3:
                color = "🟢" if sim_result["delta"] > 0 else "🔴" if sim_result["delta"] < 0 else "🟡"
                st.metric(f"{color} Variation", f"{sim_result['delta']:+.2f} points")

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sim_result["new_prediction"],
                delta={"reference": sim_result["original_prediction"], "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                title={"text": f"Impact de {sim_result['factor_label']}"},
                gauge={
                    "axis": {"range": [0, 20]},
                    "bar": {"color": "#1976d2"},
                    "steps": [
                        {"range": [0, 10], "color": "#ffcdd2"},
                        {"range": [10, 14], "color": "#fff9c4"},
                        {"range": [14, 20], "color": "#c8e6c9"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 10},
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
