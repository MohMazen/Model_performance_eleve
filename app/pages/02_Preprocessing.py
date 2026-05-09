
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

st.title("🔧 Preprocessing")

df = _get("df_raw")
if df is None:
    st.warning("⚠️ Chargez d'abord des données (Page 1).")
    st.stop()

st.write(f"Données brutes : **{df.shape[0]} lignes × {df.shape[1]} colonnes**")

col1, col2 = st.columns(2)

data_source = _get("data_source", "synthetic")

if data_source == "uploaded":
    st.markdown("### 🎯 Définition de la Cible (Target)")
    df_cols_target = _get("df_clean") if _get("df_clean") is not None else df
    num_cols_tgt = df_cols_target.select_dtypes(include=[np.number]).columns.tolist()
    default_target = TARGET_REG if TARGET_REG in num_cols_tgt else (num_cols_tgt[0] if num_cols_tgt else None)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_reg = st.selectbox("Sélectionnez la variable cible (Régression)", num_cols_tgt, index=num_cols_tgt.index(default_target) if default_target in num_cols_tgt else 0)
    with col_t2:
        threshold = st.number_input("Seuil de réussite (Classification)", value=10.0, step=0.5)
    
    _set("target_reg", target_reg)
    _set("seuil_reussite", threshold)
else:
    _set("target_reg", TARGET_REG)
    _set("seuil_reussite", SEUIL_REUSSITE)

# UI Mapping des colonnes (pour données uploadées)
if data_source == "uploaded":
    st.markdown("---")
    with st.expander("🔗 Mapping des colonnes (Automatique → Manuel)", expanded=True):
        st.info("L'IA tente de détecter vos colonnes automatiquement. Vérifiez et ajustez si nécessaire.")
        mapping = _get("column_mapping")
        if mapping is None:
            mapping = get_column_mapping(df.columns)
            _set("column_mapping", mapping)
        
        from src.features import KEYWORDS
        new_mapping = {}
        cols_avail = ["-- Non présent --"] + df.columns.tolist()
        
        matches_cols = st.columns(3)
        for i, (concept, description) in enumerate([
            ('note_moyenne', 'Cible (Note Moyenne)'),
            ('sommeil', 'Heures de sommeil'),
            ('etude', 'Heures d\'étude'),
            ('sport', 'Activité sportive (oui/non)'),
            ('jeux_video', 'Heures Jeux Vidéo'),
            ('reseaux', 'Heures Réseaux Sociaux'),
            ('streaming', 'Heures Streaming'),
            ('stress', 'Niveau de Stress'),
            ('heure_coucher', 'Heure de Coucher'),
            ('heure_lever', 'Heure de Lever')
        ]):
            with matches_cols[i % 3]:
                current_val = mapping.get(concept, "-- Non présent --")
                if current_val not in cols_avail: current_val = "-- Non présent --"
                sel = st.selectbox(f"📍 {description}", cols_avail, index=cols_avail.index(current_val), key=f"map_{concept}")
                if sel != "-- Non présent --":
                    new_mapping[concept] = sel
        
        if st.button("💾 Enregistrer le mapping"):
            _set("column_mapping", new_mapping)
            st.success("✅ Mapping mis à jour.")

st.markdown("---")
st.markdown("### Traitements")

with col1:
    if st.button("🧹 Nettoyer les données"):
        df_clean = nettoyer_donnees(df)
        _set("df_clean", df_clean)
        _set("df_feat", None)
        st.success("✅ Nettoyage effectué.")

with col2:
    if st.button("⚙️ Feature Engineering"):
        df_clean_tmp = _get("df_clean")
        df_base = df_clean_tmp if df_clean_tmp is not None else df
        
        mapping = _get("column_mapping")
        if data_source == "uploaded" and mapping is None:
            mapping = get_column_mapping(df_base.columns)
            _set("column_mapping", mapping)

        with st.spinner("Application des transformations..."):
            df_h = prenttoyer_horaires(df_base, mapping=mapping)
            df_feat = add_advanced_features(df_h, mapping=mapping)
            
            # S'assurer que le TARGET_CLF est présent même si note_moyenne absente du mapping
            if data_source == "uploaded":
                target_reg = _get("target_reg")
                threshold = _get("seuil_reussite")
                if TARGET_CLF not in df_feat.columns and target_reg and target_reg in df_feat.columns:
                    df_feat[TARGET_CLF] = (df_feat[target_reg] >= threshold).astype(int)
            
            _set("df_feat", df_feat)
            st.success("✅ Feature Engineering terminé avec succès.")

df_clean = _get("df_clean")
df_feat = _get("df_feat")

if df_clean is not None or df_feat is not None:
    st.markdown("---")
    tab_clean, tab_feat = st.tabs(["Après nettoyage", "Après Feature Engineering"])

    with tab_clean:
        if df_clean is not None:
            na_avant = df.isnull().sum().sum()
            na_apres = df_clean.isnull().sum().sum()
            st.metric("Valeurs manquantes avant", na_avant)
            st.metric("Valeurs manquantes après", na_apres)
            st.dataframe(df_clean.head(10), width='stretch')
        else:
            st.info("Cliquez sur 'Nettoyer les données'.")

    with tab_feat:
        if df_feat is not None:
            new_cols = [c for c in df_feat.columns if c not in df.columns]
            st.write(f"**{len(new_cols)} nouvelles colonnes créées :** {', '.join(new_cols)}")
            
            st.markdown(r"""
            **Signification des variables ajoutées :**
            - **`score_equilibre`** : Ratio entre le repos/détente (sommeil, sport) et la charge (devoirs, écrans). Un score élevé indique un meilleur équilibre de vie.
            - **`stress_total`** : Utilise directement le niveau de stress personnel ressenti.
            - **`perseverance`** : Capacité de l'élève à maintenir ses efforts face aux difficultés.
            - **`motivation_travail`** : Synergie entre la motivation et les heures de devoirs (indicateur d'engagement).
            - **`heure_coucher_num` / `heure_lever_num`** : Conversion des horaires en heures décimales.
            - **`reussite`** : Variable cible binaire créée pour la classification (1 si $\ge$ 10, 0 sinon).
            """)

            st.dataframe(df_feat[new_cols].head(10), width='stretch')
        else:
            st.info("Cliquez sur 'Feature Engineering'.")


# ---------------------------------------------------------------------------
# PAGE 3 : Modélisation
# ---------------------------------------------------------------------------
