
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

st.title("🤖 Modélisation")

df_feat = _get("df_feat")
if df_feat is None:
    st.warning("⚠️ Effectuez d'abord le Preprocessing (Page 2).")
    st.stop()

# Saisie du nom du modèle via une liste déroulante multi-sélection
model_options = ["XGBoost", "Random Forest", "Réseau de Neurones (MLP)", "SVM",
                 "LDA", "Gaussian Naïve Bayes", "Bagging", "QDA"]
selected_models = st.multiselect(
    "Modèles à entraîner et évaluer",
    options=model_options,
    default=_get("selected_models", ["XGBoost", "Random Forest", "Réseau de Neurones (MLP)", "SVM"]),
    help="XGBoost/RF/MLP/SVM = modèles principaux. LDA/GNB/Bagging/QDA = modèles supplémentaires (Muresan et al. 2026)."
)
_set("selected_models", selected_models)
model_name = ", ".join(selected_models) if selected_models else ""
_set("model_name", model_name)
include_extra = any(m in selected_models for m in ["LDA", "Gaussian Naïve Bayes", "Bagging", "QDA"])

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

            # Modèles supplémentaires (Muresan et al. 2026)
            if include_extra:
                model_lda_clf = mm.train_lda_classification(X_train, yc_train)
                model_gnb_clf = mm.train_gnb_classification(X_train, yc_train)
                model_bag_clf = mm.train_bag_classification(X_train, yc_train)
                model_qda_clf = mm.train_qda_classification(X_train, yc_train)
                model_bag_reg = mm.train_bag_regression(X_train, yr_train)

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
            if include_extra:
                _set("metrics_extra", [
                    ("LDA (clf)", {
                        'accuracy': accuracy_score(yc_test, model_lda_clf.predict(X_test)) * 100,
                        'f1': f1_score(yc_test, model_lda_clf.predict(X_test), zero_division=0),
                    }),
                    ("Gaussian NB (clf)", {
                        'accuracy': accuracy_score(yc_test, model_gnb_clf.predict(X_test)) * 100,
                        'f1': f1_score(yc_test, model_gnb_clf.predict(X_test), zero_division=0),
                    }),
                    ("Bagging (clf)", {
                        'accuracy': accuracy_score(yc_test, model_bag_clf.predict(X_test)) * 100,
                        'f1': f1_score(yc_test, model_bag_clf.predict(X_test), zero_division=0),
                    }),
                    ("QDA (clf)", {
                        'accuracy': accuracy_score(yc_test, model_qda_clf.predict(X_test)) * 100,
                        'f1': f1_score(yc_test, model_qda_clf.predict(X_test), zero_division=0),
                    }),
                    ("Bagging (reg)", {
                        'r2': r2_score(yr_test, model_bag_reg.predict(X_test)),
                        'mae': mean_absolute_error(yr_test, model_bag_reg.predict(X_test)),
                    }),
                ])
            _set("feature_columns", list(X_train.columns))
            mm.feature_columns = list(X_train.columns)  # BUG#5 fix : pour persistance dans joblib

            # Sélection du meilleur modèle global
            reg_candidates = [
                (metrics_reg['r2'], model_reg, "XGBoost"),
                (metrics_nn_reg['r2'], model_nn_reg, "Réseau de Neurones"),
                (metrics_svm_reg['r2'], model_svm_reg, "SVM"),
            ]
            clf_candidates = [
                (metrics_clf['accuracy'], model_clf, "Random Forest"),
                (metrics_nn_clf['accuracy'], model_nn_clf, "Réseau de Neurones"),
                (metrics_svm_clf['accuracy'], model_svm_clf, "SVM"),
            ]
            if include_extra:
                reg_candidates.append((
                    r2_score(yr_test, model_bag_reg.predict(X_test)), model_bag_reg, "Bagging"
                ))
                for extra_model, extra_label in [
                    (model_lda_clf, "LDA"),
                    (model_gnb_clf, "Gaussian NB"),
                    (model_bag_clf, "Bagging"),
                    (model_qda_clf, "QDA"),
                ]:
                    clf_candidates.append((
                        accuracy_score(yc_test, extra_model.predict(X_test)) * 100,
                        extra_model, extra_label
                    ))

            best_reg_config = max(reg_candidates, key=lambda x: x[0])
            best_clf_config = max(clf_candidates, key=lambda x: x[0])

            mm.best_overall_reg = best_reg_config[1]
            _set("best_reg_type", best_reg_config[2])
            mm.best_overall_clf = best_clf_config[1]
            _set("best_clf_type", best_clf_config[2])
            
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

    # ── Modèles supplémentaires (Muresan et al. 2026) ─────────────────────────
    metrics_extra = _get("metrics_extra")
    if metrics_extra:
        st.markdown("---")
        st.subheader("🔬 Modèles supplémentaires — Muresan et al. 2026")
        st.caption("LDA · Gaussian Naïve Bayes · Bagging · QDA — comparaison avec les modèles principaux")
        rows = []
        for label, m in metrics_extra:
            if 'r2' in m:
                rows.append({"Modèle": label, "R²": f"{m['r2']:.3f}", "MAE": f"{m['mae']:.3f}", "Accuracy": "—", "F1": "—"})
            else:
                rows.append({"Modèle": label, "R²": "—", "MAE": "—", "Accuracy": f"{m['accuracy']:.1f}%", "F1": f"{m['f1']:.3f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig_cm, use_container_width=True)
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
