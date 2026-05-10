
import io
import os
import sys
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

logger = logging.getLogger(__name__)

# Permettre l'import depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score
)
from sklearn.model_selection import train_test_split

from src.config import COLS_TO_DROP, TARGET_CLF, TARGET_REG, MODEL_FILE, SEUIL_REUSSITE
from src.data_utils import charger_donnees, generer_donnees_synthetiques, nettoyer_donnees, valider_schema
from src.explainability import generate_shap_analysis, generate_shap_failure_analysis
from src.features import add_advanced_features, nettoyer_horaires, get_column_mapping
from src.models import ModelManager
from src.reporting import generer_rapport_markdown, generer_rapport_pdf
from app.utils_st import _get, _set

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("📝 Rapport")

df_feat = _get("df_feat")
metrics_reg = _get("metrics_reg")
metrics_clf = _get("metrics_clf")
metrics_nn_reg = _get("metrics_nn_reg")
metrics_nn_clf = _get("metrics_nn_clf")
metrics_svm_reg = _get("metrics_svm_reg")
metrics_svm_clf = _get("metrics_svm_clf")
selected_features = _get("selected_features")

if df_feat is None or metrics_reg is None:
    st.warning("⚠️ Entraînez d'abord les modèles (Page 3).")
    st.stop()

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("📄 Générer le rapport Markdown"):
        model_name = _get("model_name")
        target_reg = _get("target_reg", TARGET_REG)
        threshold = _get("seuil_reussite", SEUIL_REUSSITE)
        rapport = generer_rapport_markdown(df_feat, metrics_reg, metrics_clf, path=None,
                                           metrics_nn_reg=metrics_nn_reg,
                                           metrics_nn_clf=metrics_nn_clf,
                                           metrics_svm_reg=metrics_svm_reg,
                                           metrics_svm_clf=metrics_svm_clf,
                                           selected_features=selected_features,
                                           model_name=model_name,
                                           target_col=target_reg, threshold=threshold)
        _set("rapport_md", rapport)
        st.success("✅ Rapport Markdown généré.")

with col_btn2:
    if st.button("📕 Générer le rapport PDF"):
        model_name = _get("model_name")
        target_reg = _get("target_reg", TARGET_REG)
        threshold = _get("seuil_reussite", SEUIL_REUSSITE)
        with st.spinner("Génération du PDF…"):
            pdf_bytes = generer_rapport_pdf(
                df_feat, metrics_reg, metrics_clf, path=None,
                metrics_nn_reg=metrics_nn_reg, metrics_nn_clf=metrics_nn_clf,
                metrics_svm_reg=metrics_svm_reg, metrics_svm_clf=metrics_svm_clf,
                model_name=model_name, target_col=target_reg, threshold=threshold,
            )
            if pdf_bytes:
                _set("rapport_pdf", pdf_bytes)
                st.success("✅ Rapport PDF généré.")
            else:
                st.error("❌ Échec de la génération PDF. Vérifiez que `reportlab` est installé.")

rapport_md = _get("rapport_md")
rapport_pdf = _get("rapport_pdf")

if rapport_md is not None:
    target_reg = _get("target_reg", TARGET_REG)
    threshold = _get("seuil_reussite", SEUIL_REUSSITE)

    st.markdown("---")
    st.markdown(rapport_md)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="⬇️ Télécharger le rapport (.md)",
            data=rapport_md,
            file_name="rapport_analyse_scolaire.md",
            mime="text/markdown"
        )
    with col_dl2:
        if rapport_pdf:
            st.download_button(
                label="⬇️ Télécharger le rapport (.pdf)",
                data=rapport_pdf,
                file_name="rapport_analyse_scolaire.pdf",
                mime="application/pdf"
            )

    if target_reg in df_feat.columns:
        st.markdown("---")
        st.subheader(f"Distribution de {target_reg}")
        fig = px.histogram(
            df_feat,
            x=target_reg,
            color_discrete_sequence=['#ff7f0e'],
            nbins=25,
            title=f"Distribution de {target_reg}",
            labels={target_reg: target_reg, 'count': "Nombre d'élèves"},
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Seuil réussite ({threshold})")
        st.plotly_chart(fig, use_container_width=True)

