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

from src.config import COLS_TO_DROP, TARGET_CLF, TARGET_REG, MODEL_FILE
from src.data_utils import charger_donnees, generer_donnees_synthetiques, nettoyer_donnees, valider_schema
from src.explainability import generate_shap_analysis
from src.features import add_advanced_features, prenttoyer_horaires
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
                _set("df_clean", None)
                _set("df_feat", None)
                _set("models", None)
                st.success(f"✅ {len(df)} élèves générés ({', '.join(classes_sel)}).")

    with col_upload:
        st.subheader("Charger un fichier CSV")
        uploaded = st.file_uploader("Fichier CSV (séparateur ';')", type=["csv"])
        
        # On ne reset que si un NOUVEAU fichier est chargé
        last_uploaded_name = _get("last_uploaded_name")
        
        if uploaded is not None and uploaded.name != last_uploaded_name:
            try:
                df_up = pd.read_csv(uploaded, sep=';', encoding='utf-8-sig')
                _set("df_raw", df_up)
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

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        cols_a_exclure = ['nom', 'prenom', 'prénom', 'prenoms', 'prénoms', 'Nom', 'Prenom', 'Adresse']
        num_cols = [c for c in num_cols if str(c).lower() not in [x.lower() for x in cols_a_exclure]]
        cat_cols = [c for c in cat_cols if str(c).lower() not in [x.lower() for x in cols_a_exclure]]

        if num_cols:
            st.markdown("#### Variables numériques")
            # Utiliser une palette de couleurs prédéfinie
            palette = px.colors.qualitative.Prism

            for i in range(0, len(num_cols), 2):
                cols = st.columns(2)
                for j, col_name in enumerate(num_cols[i:i + 2]):
                    # Calculer l'index global pour choisir la couleur
                    idx = i + j
                    couleur = palette[idx % len(palette)]

                    with cols[j]:
                        fig = px.histogram(
                            df, x=col_name,
                            color_discrete_sequence=[couleur],
                            title=f"Distribution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"},
                        )
                        # Ajoute une bordure blanche pour bien séparer les barres de l'histogramme
                        fig.update_traces(marker_line_width=1, marker_line_color="white")
                        st.plotly_chart(fig, width='stretch')

        if cat_cols:
            st.markdown("#### Variables catégorielles")
            for i in range(0, len(cat_cols), 2):
                cols = st.columns(2)
                for j, col_name in enumerate(cat_cols[i:i + 2]):
                    with cols[j]:
                        vc = df[col_name].value_counts().reset_index(name="count")
                        # Trier par heure si c'est une distribution horaire
                        if col_name in ['Heure_lever', 'Heure_coucher', 'heure_lever', 'heure_coucher']:
                            vc = vc.sort_values(by=col_name)
                        else:
                            vc = vc.sort_values(by="count", ascending=False).head(20)
                            
                        fig = px.bar(
                            vc, x=col_name, y="count",
                            color="count",
                            color_continuous_scale='Plasma',
                            title=f"Distribution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"},
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
            df_h = prenttoyer_horaires(df_base)
            df_feat = add_advanced_features(df_h)
            _set("df_feat", df_feat)
            st.success("✅ Features avancées créées.")

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
                
                st.markdown("""
                **Signification des variables ajoutées :**
                - **`score_equilibre`** : Ratio entre le repos/détente (sommeil, sport) et la charge (devoirs, écrans). Un score élevé indique un meilleur équilibre de vie.
                - **`stress_absences`** : Produit entre le stress et les absences, soulignant un comportement potentiellement à risque.
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

    # Saisie du nom du modèle
    model_name = st.text_input("Nom du modèle (optionnel)", 
                              value=_get("model_name", ""), 
                              placeholder="Ex: XGBoost_v1",
                              help="Ce nom apparaîtra sur les graphiques et dans le rapport.")
    _set("model_name", model_name)

    if st.button("🚀 Entraîner les modèles"):
        with st.spinner("Entraînement en cours… (peut prendre quelques minutes)"):
            try:
                cols_drop = [c for c in COLS_TO_DROP if c in df_feat.columns]
                X = df_feat.drop(columns=cols_drop + [TARGET_REG, TARGET_CLF])
                y_reg = df_feat[TARGET_REG]
                y_clf = df_feat[TARGET_CLF]

                X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
                    X, y_reg, y_clf, test_size=0.2, random_state=42
                )

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
                st.markdown("""
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

        col1, col2, col3 = st.columns(3)
        with col1:
            heures_etude = st.slider("Heures d'étude / soir", 0.0, 10.0, 3.0, 0.5)
            interet_maths = st.slider("Intérêt pour les Maths (0-10)", 0, 10, 7)
            heures_sommeil = st.slider("Heures de sommeil", 4.0, 11.0, 8.0, 0.5)
            stress_1 = st.slider("Niveau de stress 1 (0-4)", 0, 4, 1)

        with col2:
            absences_sim = st.number_input("Absences (simulées)", 0, 30, 2)
            heures_jeux = st.slider("Heures jeux vidéo / jour", 0.0, 8.0, 1.0, 0.5)
            confiance_soi = st.slider("Confiance en soi (1-10)", 1, 10, 7)
            estime_soi = st.slider("Estime de soi (1-10)", 1, 10, 7)

        with col3:
            activite_sport = st.selectbox("Activité sportive", ["oui", "non"])
            classe = st.selectbox("Classe", ["6eme", "5eme", "4eme", "3eme", "2nde", "1ere", "terminale"])
            type_etab = st.selectbox("Type d'établissement", ["Public", "Privé"])

        if st.button("🔮 Prédire"):
            # Construire un DataFrame à partir d'un échantillon pour avoir toutes les colonnes
            if df_feat is not None:
                input_row = df_feat.iloc[0:1].copy()
            else:
                st.error("Les données doivent être prétraitées avant la prédiction.")
                st.stop()

            # Mise à jour des valeurs avec la saisie utilisateur
            input_row['Heures_etude_soir'] = heures_etude
            input_row['Interet_maths'] = interet_maths
            input_row['Heures_sommeil'] = heures_sommeil
            input_row['Stress_1'] = stress_1
            input_row['Activite_sportive'] = activite_sport
            input_row['Classe'] = classe
            input_row['Heures_jeux_video'] = heures_jeux
            input_row['Confiance_soi'] = confiance_soi
            input_row['Estime_soi'] = estime_soi
            
            # Recalculer les features dérivées via la fonction centrale
            input_row = add_advanced_features(input_row)

            # Conserver uniquement les colonnes attendues par le modèle
            cols_drop = [c for c in COLS_TO_DROP if c in input_row.columns]
            targets = [c for c in [TARGET_REG, TARGET_CLF] if c in input_row.columns]
            X_input = input_row.drop(columns=cols_drop + targets)
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
                    X_all = df_feat.drop(columns=cols_drop + [TARGET_REG, TARGET_CLF])
                    X_all = X_all[feature_columns]

                    df_preds = df_feat[['nom', 'prenom', TARGET_REG]].copy()
                    df_preds['Note Prédite (Moy)'] = model_reg.predict(X_all)
                    
                    if mm and mm.subject_models:
                        for sub_name, sub_model in mm.subject_models.items():
                            col_label = f"Prédit_{sub_name.replace('note_', '')}"
                            df_preds[col_label] = sub_model.predict(X_all)
                    
                    df_preds['Écart'] = df_preds['Note Prédite (Moy)'] - df_preds[TARGET_REG]
                    
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
                buf = io.BytesIO()
                sample_size = min(50, len(X_test))
                generate_shap_analysis(model_reg, X_test.iloc[:sample_size], buf=buf)
                buf.seek(0)
                _set("shap_buf", buf)
                st.success("✅ Analyse SHAP terminée.")
            except Exception as e:
                st.error(f"Erreur SHAP : {e}")

    shap_buf = _get("shap_buf")
    if shap_buf is not None:
        # Vérification que le buffer contient des données valides
        try:
            shap_buf.seek(0, io.SEEK_END)
            size = shap_buf.tell()
            shap_buf.seek(0)
            if size > 0:
                st.subheader("Importance des facteurs de réussite")
                st.image(shap_buf, width='stretch')
                st.caption(
                    "Ce graphique montre l'impact moyen (en valeur absolue) de chaque variable "
                    "sur la prédiction de la note. Plus la barre est longue, plus la variable est influente."
                )
            else:
                st.info("💡 L'analyse SHAP n'a pas pu générer de graphique.")
        except Exception:
            st.error("Erreur lors de l'affichage de l'image SHAP.")
        st.caption(
            "Ce graphique montre l'impact moyen (en valeur absolue) de chaque variable "
            "sur la prédiction de la note. Plus la barre est longue, plus la variable est influente."
        )


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

    if df_feat is None or metrics_reg is None:
        st.warning("⚠️ Entraînez d'abord les modèles (Page 3).")
        st.stop()

    if st.button("📄 Générer le rapport"):
        model_name = _get("model_name")
        rapport = generer_rapport_markdown(df_feat, metrics_reg, metrics_clf, path=None,
                                           metrics_nn_reg=metrics_nn_reg,
                                           metrics_nn_clf=metrics_nn_clf,
                                           model_name=model_name)
        _set("rapport_md", rapport)
        st.success("✅ Rapport généré.")

    rapport_md = _get("rapport_md")
    if rapport_md is not None:
        st.markdown("---")
        st.markdown(rapport_md)

        st.download_button(
            label="⬇️ Télécharger le rapport (.md)",
            data=rapport_md,
            file_name="rapport_analyse_scolaire.md",
            mime="text/markdown"
        )

        st.markdown("---")
        st.subheader("Distribution des notes moyennes")
        fig = px.histogram(
            df_feat,
            x='note_moyenne',
            color_discrete_sequence=['#ff7f0e'],
            nbins=25,
            title="Distribution des notes moyennes",
            labels={'note_moyenne': 'Note moyenne / 20', 'count': 'Nombre d\'élèves'},
        )
        fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Seuil réussite (10)")
        st.plotly_chart(fig, width='stretch')
