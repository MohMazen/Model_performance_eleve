
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
from src.features import add_advanced_features, prenttoyer_horaires, get_column_mapping
from src.models import ModelManager
from src.reporting import generer_rapport_markdown
from app.utils_st import _get, _set

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("📊 Explicabilité (SHAP)")

model_reg = _get("model_reg")
X_test = _get("X_test")

if model_reg is None or X_test is None:
    st.warning("⚠️ Entraînez d'abord les modèles (Page 3).")
    st.stop()

if st.button("🔍 Lancer l'analyse SHAP"):
    with st.spinner("Calcul des valeurs SHAP… (peut prendre un moment)"):
        try:
            # 1. Facteurs de réussite
            buf_succ = io.BytesIO()
            sample_size = min(50, len(X_test))
            res_succ = generate_shap_analysis(model_reg, X_test.iloc[:sample_size], buf=buf_succ)
            buf_succ.seek(0)
            
            # 2. Facteurs d'échec
            buf_fail = io.BytesIO()
            target_reg = _get("target_reg", TARGET_REG)
            # On utilise les notes réelles du test pour filtrer les échecs dans l'analyse SHAP
            df_feat = _get("df_feat")
            if target_reg in df_feat.columns:
                y_test_real = df_feat.loc[X_test.index, target_reg]
                res_fail = generate_shap_failure_analysis(model_reg, X_test.iloc[:sample_size], y_test_real.iloc[:sample_size], buf=buf_fail)
            else:
                res_fail = None
            buf_fail.seek(0)

            if res_succ is not None:
                _set("shap_buf_succ", buf_succ)
                _set("shap_buf_fail", buf_fail if res_fail is not None else None)
                _set("shap_error", None)
                st.success("✅ Analyse SHAP terminée.")
            else:
                st.warning("⚠️ L'analyse SHAP n'a pas pu produire de graphique.")
        except Exception as e:
            _set("shap_error", str(e))
            st.error(f"Erreur SHAP : {e}")

shap_buf_succ = _get("shap_buf_succ")
shap_buf_fail = _get("shap_buf_fail")
shap_error = _get("shap_error")

if shap_buf_succ is not None:
    col_s, col_f = st.columns(2)
    
    with col_s:
        st.subheader("🔵 Facteurs de Réussite")
        st.image(shap_buf_succ, use_container_width=True)
        st.caption("Variables favorisant une note élevée.")

    with col_f:
        st.subheader("🔴 Facteurs d'Échec")
        if shap_buf_fail:
            st.image(shap_buf_fail, use_container_width=True)
            st.caption("Variables contribuant à une note faible (<10).")
        else:
            st.info("Aucun élève en situation d'échec dans cet échantillon pour identifier des facteurs spécifiques.")
elif shap_error:
    st.info(f"💡 {shap_error}")


# ---------------------------------------------------------------------------
# PAGE 6 : Rapport
# ---------------------------------------------------------------------------
