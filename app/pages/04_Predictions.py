
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
from src.reporting import generer_rapport_markdown
from app.utils_st import _get, _set

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("🔮 Prédictions")

model_reg = _get("model_reg")
model_clf = _get("model_clf")
feature_columns = _get("feature_columns")
df_feat = _get("df_feat")
mm = _get("mm")

if model_reg is None or model_clf is None:
    st.warning("⚠️ Aucun modèle n'est chargé en mémoire. Veuillez en entraîner un (Page 3) ou en charger un ci-dessous.")
    
st.markdown("---")
st.subheader("📁 Charger un modèle existant")

# Lister les fichiers .joblib dans outputs/
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
available_models = [f for f in os.listdir(output_dir) if f.endswith(".joblib")]

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    selected_model_file = st.selectbox("Sélectionnez un modèle", available_models if available_models else ["Aucun modèle trouvé"])

with col_btn:
    if st.button("🔌 Charger") and available_models:
        with st.spinner("Chargement..."):
            try:
                full_path = os.path.join(output_dir, selected_model_file)
                new_mm = ModelManager()
                if new_mm.load_models(path=full_path):
                    _set("mm", new_mm)
                    _set("model_reg", new_mm.best_overall_reg if new_mm.best_overall_reg else new_mm.best_model_reg)
                    _set("model_clf", new_mm.best_overall_clf if new_mm.best_overall_clf else new_mm.best_model_clf)
                    _set("model_nn_reg", new_mm.best_model_nn_reg)
                    _set("model_nn_clf", new_mm.best_model_nn_clf)
                    # BUG#5 fix : restauration des feature_columns persistées dans le joblib
                    if new_mm.feature_columns is not None:
                        _set("feature_columns", new_mm.feature_columns)
                    st.success(f"✅ Modèle {selected_model_file} chargé.")
                    st.rerun()
                else:
                    st.error("Échec du chargement.")
            except Exception as e:
                st.error(f"Erreur : {e}")

if _get("model_reg") is None:
    st.stop()

# BUG#2 fix : feature_columns doit être disponible pour les prédictions sur données uploadées
feature_columns = _get("feature_columns")

tab_ind, tab_all = st.tabs(["👤 Prédiction Individuelle", "📋 Prédictions par Élève"])

with tab_ind:
    st.subheader("Saisir les paramètres d'un élève")

    data_source = _get("data_source", "synthetic")
    input_data = {}

    if data_source == "synthetic":
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data['heures_etude_soir'] = st.slider("Heures d'étude / soir", 0.0, 10.0, 3.0, 0.5)
            input_data['interet_maths'] = st.slider("Intérêt pour les Maths (0-10)", 0, 10, 7)
            input_data['heures_sommeil'] = st.slider("Heures de sommeil", 4.0, 11.0, 8.0, 0.5)
            input_data['stress_personnel'] = st.slider("Niveau de stress personnel (0-4)", 0, 4, 1)
            input_data['perseverance'] = st.slider("Niveau de persévérance (1-5)", 1, 5, 3)

        with col2:
            input_data['heures_jeux_video'] = st.slider("Heures jeux vidéo / jour", 0.0, 8.0, 1.0, 0.5)
            input_data['confiance_soi'] = st.slider("Confiance en soi (1-10)", 1, 10, 7)
            input_data['estime_soi'] = st.slider("Estime de soi (1-10)", 1, 10, 7)

        with col3:
            input_data['activite_sportive'] = st.selectbox("Activité sportive", ["oui", "non"])
            input_data['classe'] = st.selectbox("Classe", ["6eme", "5eme", "4eme", "3eme", "2nde", "1ere", "terminale"])
            input_data['type_etab'] = st.selectbox("Type d'établissement", ["Public", "Privé"])

    else:
        # BUG#2 fix : feature_columns est requis pour le mode "données uploadées"
        if feature_columns is None:
            st.error("⚠️ Les colonnes du modèle sont introuvables. Veuillez d'abord entraîner un modèle (Page 3) ou recharger un modèle sauvegardé ci-dessus.")
            st.stop()
        if df_feat is None:
            st.error("⚠️ Les données doivent être prétraitées (Page 2) avant la prédiction.")
            st.stop()
        st.info("Saisie dynamique des paramètres pour le modèle.")
        cols = st.columns(3)
        for i, col in enumerate(feature_columns):
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(df_feat[col]):
                    val = float(df_feat[col].median()) if not df_feat[col].isnull().all() else 0.0
                    input_data[col] = st.number_input(col, value=val, key=f"inp_{col}")
                else:
                    options = [x for x in df_feat[col].dropna().unique() if str(x) != ""]
                    if not options: options = ["Inconnu"]
                    input_data[col] = st.selectbox(col, options=options, key=f"inp_{col}")

    if st.button("🔮 Prédire"):
        # Construire un DataFrame à partir d'un échantillon pour avoir toutes les colonnes
        if df_feat is not None:
            input_row = df_feat.iloc[0:1].copy()
        else:
            st.error("Les données doivent être prétraitées avant la prédiction.")
            st.stop()

        # Mise à jour des valeurs avec la saisie utilisateur
        for k, v in input_data.items():
            input_row[k] = v
        
        if data_source == "synthetic":
            # Recalculer les features dérivées via la fonction centrale
            input_row = add_advanced_features(input_row)

        target_reg = _get("target_reg", TARGET_REG)
        # Conserver uniquement les colonnes attendues par le modèle
        cols_drop = [c for c in COLS_TO_DROP if c in input_row.columns]
        targets = [c for c in [target_reg, TARGET_CLF] if c in input_row.columns]
        X_input = input_row.drop(columns=cols_drop + targets, errors='ignore')
        
        missing_cols = set(feature_columns) - set(X_input.columns)
        for mcol in missing_cols:
            X_input[mcol] = 0
            
        X_input = X_input[feature_columns]

        note_pred = model_reg.predict(X_input)[0]
        
        classes = list(model_clf.classes_)
        prob_preds = model_clf.predict_proba(X_input)[0]
        if 1 in classes:
            proba_reussite = prob_preds[classes.index(1)] * 100
        else:
            proba_reussite = 0.0

        st.markdown("---")
        col_r, col_c = st.columns(2)
        with col_r:
            st.subheader("Note Moyenne Prédite")
            color = "🟢" if note_pred >= 14 else "🟠" if note_pred >= 10 else "🔴"
            st.metric(f"{color} Note prédite", f"{note_pred:.2f} / 20")

        with col_c:
            st.subheader("Probabilité de Réussite")
            st.metric("Probabilité", f"{proba_reussite:.1f}%")
            st.progress(int(proba_reussite))

        # Prédictions par matière
        if mm and mm.subject_models:
            st.markdown("---")
            st.subheader("Détails par matière")
            cols_sub = st.columns(len(mm.subject_models))
            for i, (sub_name, sub_model) in enumerate(mm.subject_models.items()):
                sub_pred = sub_model.predict(X_input)[0]
                with cols_sub[i]:
                    st.metric(sub_name.replace('note_', '').capitalize(), f"{sub_pred:.2f} / 20")

with tab_all:
    st.subheader("Prédictions pour tous les élèves")
    if st.button("📊 Générer les prédictions globales"):
        with st.spinner("Calcul en cours…"):
            try:
                cols_drop = [c for c in COLS_TO_DROP if c in df_feat.columns]
                target_reg = _get("target_reg", TARGET_REG)
                X_all = df_feat.drop(columns=cols_drop + [target_reg, TARGET_CLF], errors='ignore')
                X_all = X_all[feature_columns]

                # Identification dynamique des colonnes d'identité pour éviter les erreurs de casse
                ident_cols = [c for c in df_feat.columns if str(c).lower() in ['nom', 'prenom', 'prénom', 'id', 'mail']]
                cols_to_select = ident_cols + [target_reg] if target_reg in df_feat.columns else ident_cols
                df_preds = df_feat[cols_to_select].copy()
                df_preds['Note Prédite (Moy)'] = model_reg.predict(X_all)
                
                if mm and mm.subject_models:
                    for sub_name, sub_model in mm.subject_models.items():
                        col_label = f"Prédit_{sub_name.replace('note_', '')}"
                        df_preds[col_label] = sub_model.predict(X_all)
                
                if target_reg in df_preds.columns:
                    df_preds['Écart'] = df_preds['Note Prédite (Moy)'] - df_preds[target_reg]
                
                st.dataframe(df_preds, use_container_width=True)
                
                csv = df_preds.to_csv(index=False, sep=';', encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ Télécharger les prédictions (CSV)",
                    data=csv,
                    file_name="predictions_eleves.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")


# ---------------------------------------------------------------------------
# PAGE 5 : Explicabilité (SHAP)
# ---------------------------------------------------------------------------
