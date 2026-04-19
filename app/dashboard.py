"""
Dashboard Streamlit complet - Analyse Prédictive des Performances Scolaires.
6 pages : Données, Preprocessing, Modélisation, Prédictions, Explicabilité, Rapport.
"""
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduStats – Analyse Scolaire",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = [
    "📂 Données",
    "🔧 Preprocessing",
    "🤖 Modélisation",
    "🔮 Prédictions",
    "📊 Explicabilité (SHAP)",
    "📝 Rapport",
]

st.sidebar.title("🎓 EduStats")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v2.0")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(key, default=None):
    return st.session_state.get(key, default)


def _set(key, value):
    st.session_state[key] = value


# ---------------------------------------------------------------------------
# PAGE 1 : Données
# ---------------------------------------------------------------------------
if page == PAGES[0]:
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
elif page == PAGES[1]:
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
elif page == PAGES[2]:
    st.title("🤖 Modélisation")

    df_feat = _get("df_feat")
    if df_feat is None:
        st.warning("⚠️ Effectuez d'abord le Preprocessing (Page 2).")
        st.stop()

    # Saisie du nom du modèle via une liste déroulante multi-sélection
    model_options = ["XGBoost", "Random Forest", "Réseau de Neurones (MLP)", "SVM"]
    selected_models = st.multiselect(
        "Modèles à identifier (Nom du modèle)",
        options=model_options,
        default=_get("selected_models", []),
        help="Sélectionnez les modèles qui seront mentionnés dans le rapport et les résultats."
    )
    _set("selected_models", selected_models)
    model_name = ", ".join(selected_models) if selected_models else ""
    _set("model_name", model_name)

    data_source = _get("data_source", "synthetic")
    target_reg = _get("target_reg")
    threshold = _get("seuil_reussite")
    from src.config import TARGET_CLF

    if st.button("🚀 Entraîner les modèles"):
        with st.spinner("Entraînement en cours… (peut prendre quelques minutes)"):
            try:
                if data_source == "synthetic":
                    cols_drop = [c for c in COLS_TO_DROP if c in df_feat.columns]
                else:
                    # Pour les données uploadées, on ne drop que les identifiants classiques
                    common_ids = ['id', 'nom', 'prenom', 'prénom', 'adresse', 'mail']
                    cols_drop = [c for c in df_feat.columns if c.lower() in common_ids]
                
                # S'assurer de drop les deux cibles
                X = df_feat.drop(columns=[c for c in cols_drop if c in df_feat.columns] + [target_reg, TARGET_CLF], errors='ignore')
                y_reg = df_feat[target_reg]
                y_clf = df_feat[TARGET_CLF]

                # Sécurité : Vérifier qu'il y a bien au moins 2 classes pour la classification
                if len(np.unique(y_clf)) < 2:
                    st.error("⚠️ **Impossible d'entraîner la classification** : La variable cible ne contient qu'une seule classe. Cela signifie que le seuil de réussite choisi fait que tous les élèves réussissent (ou échouent). Veuillez ajuster le seuil.")
                    st.stop()

                X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
                    X, y_reg, y_clf, test_size=0.2, random_state=42
                )
                
                # Double vérification après le split
                if len(np.unique(yc_train)) < 2:
                    st.error("⚠️ Le jeu d'entraînement ne contient qu'une seule classe après séparation. L'échantillon est trop déséquilibré. Ajustez le seuil.")
                    st.stop()

                mm = ModelManager()
                mm.prepare_pipeline(X_train)
                model_reg = mm.train_regression(X_train, yr_train)
                
                # Entraînement des modèles par matière
                from src.config import GRADE_COLUMNS
                for subject in GRADE_COLUMNS:
                    if subject in df_feat.columns:
                        mm.train_regression(X_train, df_feat.loc[X_train.index, subject], subject_name=subject)

                model_clf = mm.train_classification(X_train, yc_train)
                model_nn_reg = mm.train_nn_regression(X_train, yr_train)
                model_nn_clf = mm.train_nn_classification(X_train, yc_train)
                model_svm_reg = mm.train_svm_regression(X_train, yr_train)
                model_svm_clf = mm.train_svm_classification(X_train, yc_train)

                yr_pred = model_reg.predict(X_test)
                yc_pred = model_clf.predict(X_test)
                yr_nn_pred = model_nn_reg.predict(X_test)
                yc_nn_pred = model_nn_clf.predict(X_test)
                yr_svm_pred = model_svm_reg.predict(X_test)
                yc_svm_pred = model_svm_clf.predict(X_test)

                metrics_reg = {
                    'r2': r2_score(yr_test, yr_pred),
                    'mae': mean_absolute_error(yr_test, yr_pred),
                    'rmse': np.sqrt(mean_squared_error(yr_test, yr_pred))
                }
                metrics_clf = {
                    'accuracy': accuracy_score(yc_test, yc_pred) * 100,
                    'f1': f1_score(yc_test, yc_pred, zero_division=0),
                    'precision': precision_score(yc_test, yc_pred, zero_division=0),
                    'recall': recall_score(yc_test, yc_pred, zero_division=0)
                }
                metrics_nn_reg = {
                    'r2': r2_score(yr_test, yr_nn_pred),
                    'mae': mean_absolute_error(yr_test, yr_nn_pred),
                    'rmse': np.sqrt(mean_squared_error(yr_test, yr_nn_pred))
                }
                metrics_nn_clf = {
                    'accuracy': accuracy_score(yc_test, yc_nn_pred) * 100,
                    'f1': f1_score(yc_test, yc_nn_pred, zero_division=0),
                    'precision': precision_score(yc_test, yc_nn_pred, zero_division=0),
                    'recall': recall_score(yc_test, yc_nn_pred, zero_division=0)
                }
                metrics_svm_reg = {
                    'r2': r2_score(yr_test, yr_svm_pred),
                    'mae': mean_absolute_error(yr_test, yr_svm_pred),
                    'rmse': np.sqrt(mean_squared_error(yr_test, yr_svm_pred))
                }
                metrics_svm_clf = {
                    'accuracy': accuracy_score(yc_test, yc_svm_pred) * 100,
                    'f1': f1_score(yc_test, yc_svm_pred, zero_division=0),
                    'precision': precision_score(yc_test, yc_svm_pred, zero_division=0),
                    'recall': recall_score(yc_test, yc_svm_pred, zero_division=0)
                }
                cm = confusion_matrix(yc_test, yc_pred, labels=[0, 1])

                _set("mm", mm)
                _set("model_reg", model_reg)
                _set("model_clf", model_clf)
                _set("model_nn_reg", model_nn_reg)
                _set("model_nn_clf", model_nn_clf)
                _set("model_svm_reg", model_svm_reg)
                _set("model_svm_clf", model_svm_clf)
                _set("X_test", X_test)
                _set("metrics_reg", metrics_reg)
                _set("metrics_clf", metrics_clf)
                _set("metrics_nn_reg", metrics_nn_reg)
                _set("metrics_nn_clf", metrics_nn_clf)
                _set("metrics_svm_reg", metrics_svm_reg)
                _set("metrics_svm_clf", metrics_svm_clf)
                _set("confusion_matrix", cm)
                _set("feature_columns", list(X_train.columns))

                # Sélection du meilleur modèle global
                # Régression
                best_reg_config = {"score": metrics_reg['r2'], "model": model_reg, "type": "XGBoost"}
                if metrics_nn_reg['r2'] > best_reg_config["score"]:
                    best_reg_config = {"score": metrics_nn_reg['r2'], "model": model_nn_reg, "type": "Réseau de Neurones"}
                if metrics_svm_reg['r2'] > best_reg_config["score"]:
                    best_reg_config = {"score": metrics_svm_reg['r2'], "model": model_svm_reg, "type": "SVM"}
                
                mm.best_overall_reg = best_reg_config["model"]
                _set("best_reg_type", best_reg_config["type"])

                # Classification
                best_clf_config = {"score": metrics_clf['accuracy'], "model": model_clf, "type": "Random Forest"}
                if metrics_nn_clf['accuracy'] > best_clf_config["score"]:
                    best_clf_config = {"score": metrics_nn_clf['accuracy'], "model": model_nn_clf, "type": "Réseau de Neurones"}
                if metrics_svm_clf['accuracy'] > best_clf_config["score"]:
                    best_clf_config = {"score": metrics_svm_clf['accuracy'], "model": model_svm_clf, "type": "SVM"}

                mm.best_overall_clf = best_clf_config["model"]
                _set("best_clf_type", best_clf_config["type"])
                
                # Mettre à jour les modèles actifs avec les meilleurs
                _set("model_reg", mm.best_overall_reg)
                _set("model_clf", mm.best_overall_clf)

                # Extraction des variables sélectionnées (pour affichage)
                # On récupère le support du sélecteur du meilleur modèle de régression
                try:
                    pipeline = mm.best_overall_reg
                    preprocessor = pipeline.named_steps['pre']
                    selector = pipeline.named_steps['select']
                    
                    # Noms des colonnes après preprocessing
                    cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
                    num_names = preprocessor.transformers_[0][2]
                    all_names = list(num_names) + list(cat_names)
                    
                    # Filtrage par le sélecteur
                    selected_mask = selector.get_support()
                    selected_features = [name for name, selected in zip(all_names, selected_mask) if selected]
                    excluded_features = [name for name, selected in zip(all_names, selected_mask) if not selected]
                    
                    _set("selected_features", selected_features)
                    _set("excluded_features", excluded_features)
                except Exception as e:
                    logger.warning(f"Impossible d'extraire les variables sélectionnées : {e}")

                st.success(f"✅ Modèles entraînés avec succès. Meilleurs : { _get('best_reg_type') } (Rég) et { _get('best_clf_type') } (Clf).")
            except Exception as e:
                st.error(f"Erreur lors de l'entraînement : {e}")

    metrics_reg = _get("metrics_reg")
    metrics_clf = _get("metrics_clf")
    metrics_nn_reg = _get("metrics_nn_reg")
    metrics_nn_clf = _get("metrics_nn_clf")
    metrics_svm_reg = _get("metrics_svm_reg")
    metrics_svm_clf = _get("metrics_svm_clf")
    cm = _get("confusion_matrix")

    if metrics_reg is not None:
        st.markdown("---")
        
        with st.expander("ℹ️ Comprendre les métriques d'évaluation"):
            col_inf1, col_inf2 = st.columns(2)
            with col_inf1:
                st.markdown(r"""
                **Régression (Prédire la note) :**
                - **R²** : Score entre $-\infty$ et 1. Plus il est proche de 1, plus le modèle explique bien les variations de notes.
                - **MAE** : Écart moyen (en points) entre la note réelle et la note prédite.
                - **RMSE** : Écart-type des erreurs de prédiction (pénalise les gros écarts).
                """)
            with col_inf2:
                st.markdown("""
                **Classification (Prédire la réussite) :**
                - **Accuracy** : Pourcentage global de prédictions correctes.
                - **F1-Score** : Équilibre entre précision et rappel (idéal pour les classes déséquilibrées).
                - **Precision** : Fiabilité de l'annonce d'une réussite.
                - **Recall** : Capacité à détecter tous les élèves en réussite.
                """)

        # Affichage des variables sélectionnées par l'IA
        selected_features = _get("selected_features")
        excluded_features = _get("excluded_features")
        if selected_features:
            with st.expander("🔍 Sélection automatique des variables (Feature Selection)"):
                st.write(f"L'IA a automatiquement filtré les données pour ne garder que les facteurs ayant un impact réel.")
                st.write(f"**{len(selected_features)} variables conservées.**")
                
                col_feat1, col_feat2 = st.columns(2)
                with col_feat1:
                    st.success("**✅ Variables sélectionnées :**")
                    # Afficher par petits groupes pour la lisibilité
                    st.write(", ".join(selected_features[:30]) + ("..." if len(selected_features) > 30 else ""))
                
                with col_feat2:
                    if excluded_features:
                        st.error("**❌ Variables exclues (bruit/non-corrélatives) :**")
                        st.write(", ".join(excluded_features[:30]) + ("..." if len(excluded_features) > 30 else ""))
                    else:
                        st.info("Toutes les variables ont été jugées pertinentes.")

        col1, col2, col3, col4 = st.columns(4)

        model_suffix = f" ({model_name})" if model_name else ""

        with col1:
            st.subheader(f"📈 Régression{model_suffix}")
            st.metric("R²", f"{metrics_reg['r2']:.3f}")
            st.metric("MAE", f"{metrics_reg['mae']:.3f}")
            st.metric("RMSE", f"{metrics_reg['rmse']:.3f}")

        with col2:
            st.subheader(f"🎯 Classification{model_suffix}")
            st.metric("Accuracy", f"{metrics_clf['accuracy']:.1f}%")
            st.metric("F1-Score", f"{metrics_clf['f1']:.3f}")
            st.metric("Precision", f"{metrics_clf['precision']:.3f}")
            st.metric("Recall", f"{metrics_clf['recall']:.3f}")

        with col3:
            st.subheader(f"🧠 Réseau de Neurones{model_suffix}")
            if metrics_nn_reg is not None:
                st.markdown("**Régression (MLP)**")
                st.metric("R²", f"{metrics_nn_reg['r2']:.3f}")
                st.metric("MAE", f"{metrics_nn_reg['mae']:.3f}")
                st.metric("RMSE", f"{metrics_nn_reg['rmse']:.3f}")
            if metrics_nn_clf is not None:
                st.markdown("**Classification (MLP)**")
                st.metric("Accuracy ", f"{metrics_nn_clf['accuracy']:.1f}%")
                st.metric("F1-Score ", f"{metrics_nn_clf['f1']:.3f}")

        with col4:
            st.subheader(f"🛡️ SVM{model_suffix}")
            if metrics_svm_reg is not None:
                st.markdown("**Régression (SVR)**")
                st.metric("R² ", f"{metrics_svm_reg['r2']:.3f}")
                st.metric("MAE ", f"{metrics_svm_reg['mae']:.3f}")
                st.metric("RMSE ", f"{metrics_svm_reg['rmse']:.3f}")
            if metrics_svm_clf is not None:
                st.markdown("**Classification (SVC)**")
                st.metric("Accuracy  ", f"{metrics_svm_clf['accuracy']:.1f}%")
                st.metric("F1-Score  ", f"{metrics_svm_clf['f1']:.3f}")

        if cm is not None:
            st.subheader("Matrice de confusion")
            fig_cm = ff.create_annotated_heatmap(
                z=cm.tolist(),
                x=["Prédit Échec", "Prédit Réussite"],
                y=["Réel Échec", "Réel Réussite"],
                colorscale="Blues",
                showscale=True
            )
            fig_cm.update_layout(title="Matrice de confusion")
            st.plotly_chart(fig_cm, width='stretch')
            st.info("""
            **Comment lire cette matrice ?**
            - **Diagonale (bleu foncé)** : Prédictions correctes (Réel = Prédit).
            - **Prédit Réussite / Réel Échec** : Faux Positifs (le modèle s'est trompé en annonçant une réussite).
            - **Prédit Échec / Réel Réussite** : Faux Négatifs (le modèle a manqué une réussite).
            """)

        st.markdown("---")
        st.subheader("💾 Sauvegarder les modèles")
        
        save_path = st.text_input("Chemin et nom du fichier de sauvegarde", value=MODEL_FILE, key="save_path_input")
        
        file_exists = os.path.exists(save_path)
        can_save = True
        
        if file_exists:
            st.warning(f"⚠️ Le fichier `{save_path}` existe déjà.")
            confirm_overwrite = st.checkbox("Confirmer l'écrasement", value=False)
            if not confirm_overwrite:
                can_save = False
                st.info("Cochez la case ci-dessus pour autoriser l'écriture.")

        if st.button("💾 Sauvegarder", disabled=not can_save):
            mm = _get("mm")
            if mm:
                dir_name = os.path.dirname(save_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                mm.save_models(path=save_path)
                st.success(f"✅ Modèles sauvegardés dans {save_path}. Le modèle le plus performant sera utilisé par défaut.")
            else:
                st.error("Aucun modèle disponible.")


# ---------------------------------------------------------------------------
# PAGE 4 : Prédictions
# ---------------------------------------------------------------------------
elif page == PAGES[3]:
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
                        # Utiliser les meilleurs modèles globaux s'ils existent, sinon les modèles par défaut
                        _set("model_reg", new_mm.best_overall_reg if new_mm.best_overall_reg else new_mm.best_model_reg)
                        _set("model_clf", new_mm.best_overall_clf if new_mm.best_overall_clf else new_mm.best_model_clf)
                        _set("model_nn_reg", new_mm.best_model_nn_reg)
                        _set("model_nn_clf", new_mm.best_model_nn_clf)
                        # On suppose que les features sont les mêmes que celles du pipeline
                        # Si besoin de stocker feature_columns dans le joblib, il faudrait modifier ModelManager.save_models
                        # Pour l'instant on réutilise ce qu'il y a en session_state si présent
                        st.success(f"✅ Modèle {selected_model_file} chargé.")
                        st.rerun()
                    else:
                        st.error("Échec du chargement.")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    if _get("model_reg") is None:
        st.stop()

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
                    
                    st.dataframe(df_preds, width='stretch')
                    
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
elif page == PAGES[4]:
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
            st.image(shap_buf_succ, width='stretch')
            st.caption("Variables favorisant une note élevée.")

        with col_f:
            st.subheader("🔴 Facteurs d'Échec")
            if shap_buf_fail:
                st.image(shap_buf_fail, width='stretch')
                st.caption("Variables contribuant à une note faible (<10).")
            else:
                st.info("Aucun élève en situation d'échec dans cet échantillon pour identifier des facteurs spécifiques.")
    elif shap_error:
        st.info(f"💡 {shap_error}")


# ---------------------------------------------------------------------------
# PAGE 6 : Rapport
# ---------------------------------------------------------------------------
elif page == PAGES[5]:
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

    if st.button("📄 Générer le rapport"):
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
        st.success("✅ Rapport généré.")

    rapport_md = _get("rapport_md")
    if rapport_md is not None:
        target_reg = _get("target_reg", TARGET_REG)
        threshold = _get("seuil_reussite", SEUIL_REUSSITE)
        
        st.markdown("---")
        st.markdown(rapport_md)

        st.download_button(
            label="⬇️ Télécharger le rapport (.md)",
            data=rapport_md,
            file_name="rapport_analyse_scolaire.md",
            mime="text/markdown"
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
                labels={target_reg: target_reg, 'count': 'Nombre d\'élèves'},
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Seuil réussite ({threshold})")
            st.plotly_chart(fig, width='stretch')
