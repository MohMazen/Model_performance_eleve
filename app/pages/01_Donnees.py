
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
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v2.1")

st.title("📂 Données")

col_gen, col_upload = st.columns(2)

with col_gen:
    st.subheader("Générer des données synthétiques")
    n_eleves = st.number_input("Nombre d'élèves", min_value=50, max_value=2000, value=300, step=50)
    
    classes_dispo = [
        "Sixième", "Cinquième", "Quatrième", "Troisième", 
        "Seconde", "Première", "Terminale"
    ]
    classes_sel = st.multiselect(
        "Classes à inclure", 
        options=classes_dispo,
        default=["Quatrième", "Troisième"]
    )
    
    if st.button("🔄 Générer", key="btn_gen"):
        if not classes_sel:
            st.error("⚠️ Veuillez sélectionner au moins une classe.")
        else:
            with st.spinner("Génération en cours…"):
                df = generer_donnees_synthetiques(int(n_eleves), classes_selectionnees=classes_sel)
            _set("df_raw", df)
            _set("data_source", "synthetic")
            _set("df_clean", None)
            _set("df_feat", None)
            _set("models", None)
            st.success(f"✅ {len(df)} élèves générés ({', '.join(classes_sel)}).")

with col_upload:
    st.subheader("Charger un fichier CSV")
    uploaded = st.file_uploader("Fichier CSV (séparateur ',' ou ';')", type=["csv"])
    
    # On ne reset que si un NOUVEAU fichier est chargé
    last_uploaded_name = _get("last_uploaded_name")
    
    if uploaded is not None and uploaded.name != last_uploaded_name:
        try:
            # pandas sep=None détecte automatiquement , ou ;
            df_up = pd.read_csv(uploaded, sep=None, engine='python', encoding='utf-8-sig')
            df_up.columns = [c.lower() for c in df_up.columns]
            _set("df_raw", df_up)
            _set("data_source", "uploaded")
            _set("df_clean", None)
            _set("df_feat", None)
            _set("models", None)
            _set("last_uploaded_name", uploaded.name)
            st.success(f"✅ Fichier chargé : {df_up.shape[0]} lignes, {df_up.shape[1]} colonnes.")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

df = _get("df_raw")
if df is not None:
    st.markdown("---")
    st.subheader("Aperçu des données")
    st.dataframe(df.head(20), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(), width='stretch')
    with col2:
        st.subheader("Valeurs manquantes")
        na_df = df.isnull().sum().rename("NaN").reset_index()
        na_df.columns = ["Colonne", "Valeurs manquantes"]
        na_df = na_df[na_df["Valeurs manquantes"] > 0]
        if na_df.empty:
            st.info("Aucune valeur manquante.")
        else:
            st.dataframe(na_df, width='stretch')

    st.markdown("---")
    st.subheader("Visualisations des données")

    cols_a_exclure = ['nom', 'prenom', 'prénom', 'prenoms', 'prénoms', 'Nom', 'Prenom', 'Adresse', 'id', 'mail']
    all_cols = [c for c in df.columns if str(c).lower() not in [x.lower() for x in cols_a_exclure]]

    if all_cols:
        palette = px.colors.qualitative.Prism
        for i in range(0, len(all_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(all_cols[i:i + 2]):
                idx = i + j
                couleur = palette[idx % len(palette)]
                with cols[j]:
                    unique_vals = df[col_name].nunique()
                    # Ligne pour les dates
                    if pd.api.types.is_datetime64_any_dtype(df[col_name]) or 'date' in str(col_name).lower():
                        vc = df[col_name].value_counts().sort_index().reset_index(name="count")
                        fig = px.line(
                            vc, x=col_name, y="count",
                            title=f"Évolution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"},
                            color_discrete_sequence=[couleur]
                        )
                    # Histogramme pour les valeurs numériques continues ou avec beaucoup de valeurs uniques
                    elif pd.api.types.is_numeric_dtype(df[col_name]) and unique_vals > 10:
                        fig = px.histogram(
                            df, x=col_name,
                            color_discrete_sequence=[couleur],
                            title=f"Distribution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"}
                        )
                        fig.update_traces(marker_line_width=1, marker_line_color="white")
                    # Camembert (Pie) pour les catégories avec peu de valeurs uniques
                    elif unique_vals <= 10:
                        vc = df[col_name].value_counts().reset_index(name="count")
                        fig = px.pie(
                            vc, names=col_name, values="count",
                            title=f"Répartition de {col_name}",
                            hole=0.3
                        )
                    # Bar chart pour les catégories avec plus de 10 valeurs
                    else:
                        vc = df[col_name].value_counts().reset_index(name="count")
                        if col_name in ['heure_lever', 'heure_coucher']:
                            vc = vc.sort_values(by=col_name)
                        else:
                            vc = vc.sort_values(by="count", ascending=False).head(20)
                            
                        fig = px.bar(
                            vc, x=col_name, y="count",
                            color="count",
                            color_continuous_scale='Plasma',
                            title=f"Distribution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"}
                        )
                    st.plotly_chart(fig, width='stretch')
else:
    st.info("Générez des données synthétiques ou chargez un fichier CSV pour commencer.")


# ---------------------------------------------------------------------------
# PAGE 2 : Preprocessing
# ---------------------------------------------------------------------------
