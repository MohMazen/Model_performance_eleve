"""
Dashboard Streamlit complet - Analyse Prédictive des Performances Scolaires.
6 pages : Données, Preprocessing, Modélisation, Prédictions, Explicabilité, Rapport.
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

# Permettre l'import depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score
)
from sklearn.model_selection import train_test_split

from src.config import COLS_TO_DROP, TARGET_CLF, TARGET_REG
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
        if st.button("🔄 Générer", key="btn_gen"):
            with st.spinner("Génération en cours…"):
                df = generer_donnees_synthetiques(int(n_eleves))
            _set("df_raw", df)
            _set("df_clean", None)
            _set("df_feat", None)
            _set("models", None)
            st.success(f"✅ {len(df)} élèves générés.")

    with col_upload:
        st.subheader("Charger un fichier CSV")
        uploaded = st.file_uploader("Fichier CSV (séparateur ';')", type=["csv"])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded, sep=';', encoding='utf-8-sig')
                _set("df_raw", df_up)
                _set("df_clean", None)
                _set("df_feat", None)
                _set("models", None)
                st.success(f"✅ Fichier chargé : {df_up.shape[0]} lignes, {df_up.shape[1]} colonnes.")
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")

    df = _get("df_raw")
    if df is not None:
        st.markdown("---")
        st.subheader("Aperçu des données")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistiques descriptives")
            st.dataframe(df.describe(), use_container_width=True)
        with col2:
            st.subheader("Valeurs manquantes")
            na_df = df.isnull().sum().rename("NaN").reset_index()
            na_df.columns = ["Colonne", "Valeurs manquantes"]
            na_df = na_df[na_df["Valeurs manquantes"] > 0]
            if na_df.empty:
                st.info("Aucune valeur manquante.")
            else:
                st.dataframe(na_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Visualisations des données")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        cols_a_exclure = ['nom', 'prenom', 'prénom', 'prenoms', 'prénoms']
        num_cols = [c for c in num_cols if str(c).lower() not in cols_a_exclure]
        cat_cols = [c for c in cat_cols if str(c).lower() not in cols_a_exclure]

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
                        st.plotly_chart(fig, use_container_width=True)

        if cat_cols:
            st.markdown("#### Variables catégorielles")
            for i in range(0, len(cat_cols), 2):
                cols = st.columns(2)
                for j, col_name in enumerate(cat_cols[i:i + 2]):
                    with cols[j]:
                        vc = df[col_name].value_counts().head(20).reset_index(name="count")
                        fig = px.bar(
                            vc, x=col_name, y="count",
                            color="count",
                            color_continuous_scale='Plasma',
                            title=f"Distribution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
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
                st.dataframe(df_clean.head(10), use_container_width=True)
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

                st.dataframe(df_feat[new_cols].head(10), use_container_width=True)
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
                model_clf = mm.train_classification(X_train, yc_train)
                model_nn_reg = mm.train_nn_regression(X_train, yr_train)
                model_nn_clf = mm.train_nn_classification(X_train, yc_train)

                yr_pred = model_reg.predict(X_test)
                yc_pred = model_clf.predict(X_test)
                yr_nn_pred = model_nn_reg.predict(X_test)
                yc_nn_pred = model_nn_clf.predict(X_test)

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
                cm = confusion_matrix(yc_test, yc_pred, labels=[0, 1])

                _set("mm", mm)
                _set("model_reg", model_reg)
                _set("model_clf", model_clf)
                _set("model_nn_reg", model_nn_reg)
                _set("model_nn_clf", model_nn_clf)
                _set("X_test", X_test)
                _set("metrics_reg", metrics_reg)
                _set("metrics_clf", metrics_clf)
                _set("metrics_nn_reg", metrics_nn_reg)
                _set("metrics_nn_clf", metrics_nn_clf)
                _set("confusion_matrix", cm)
                _set("feature_columns", list(X_train.columns))
                st.success("✅ Modèles entraînés avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de l'entraînement : {e}")

    metrics_reg = _get("metrics_reg")
    metrics_clf = _get("metrics_clf")
    metrics_nn_reg = _get("metrics_nn_reg")
    metrics_nn_clf = _get("metrics_nn_clf")
    cm = _get("confusion_matrix")

    if metrics_reg is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📈 Régression (XGBoost)")
            st.metric("R²", f"{metrics_reg['r2']:.3f}")
            st.metric("MAE", f"{metrics_reg['mae']:.3f}")
            st.metric("RMSE", f"{metrics_reg['rmse']:.3f}")

        with col2:
            st.subheader("🎯 Classification (Random Forest)")
            st.metric("Accuracy", f"{metrics_clf['accuracy']:.1f}%")
            st.metric("F1-Score", f"{metrics_clf['f1']:.3f}")
            st.metric("Precision", f"{metrics_clf['precision']:.3f}")
            st.metric("Recall", f"{metrics_clf['recall']:.3f}")

        with col3:
            st.subheader("🧠 Réseau de Neurones")
            if metrics_nn_reg is not None:
                st.markdown("**Régression (MLP)**")
                st.metric("R²", f"{metrics_nn_reg['r2']:.3f}")
                st.metric("MAE", f"{metrics_nn_reg['mae']:.3f}")
                st.metric("RMSE", f"{metrics_nn_reg['rmse']:.3f}")
            if metrics_nn_clf is not None:
                st.markdown("**Classification (MLP)**")
                st.metric("Accuracy ", f"{metrics_nn_clf['accuracy']:.1f}%")
                st.metric("F1-Score ", f"{metrics_nn_clf['f1']:.3f}")

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

        if st.button("💾 Sauvegarder les modèles"):
            mm = _get("mm")
            if mm:
                os.makedirs("outputs", exist_ok=True)
                mm.save_models()
                st.success("✅ Modèles sauvegardés dans outputs/model_final.joblib.")
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

    if model_reg is None or model_clf is None:
        st.warning("⚠️ Entraînez d'abord les modèles (Page 3).")
        st.stop()

    st.subheader("Saisir les paramètres d'un élève")

    col1, col2, col3 = st.columns(3)
    with col1:
        heures_devoirs = st.slider("Heures de devoirs / jour", 0.5, 15.0, 4.0, 0.5)
        motivation = st.slider("Motivation (1-10)", 1.0, 10.0, 7.0, 0.5)
        heures_sommeil = st.slider("Heures de sommeil", 4.0, 11.0, 8.0, 0.5)
        stress = st.slider("Stress (1-10)", 1.0, 10.0, 5.0, 0.5)

    with col2:
        absences = st.number_input("Absences", 0, 30, 2)
        temps_ecrans = st.slider("Temps d'écrans (h/j)", 0.0, 12.0, 3.0, 0.5)
        confiance_soi = st.slider("Confiance en soi (1-10)", 1.0, 10.0, 6.0, 0.5)
        perseverance = st.slider("Persévérance (1-10)", 1.0, 10.0, 7.0, 0.5)

    with col3:
        genre = st.selectbox("Genre", ["M", "F"])
        sport = st.selectbox("Pratique du sport", ["Oui", "Non"])
        classe = st.selectbox("Classe", ["1ère", "Terminale"])
        type_etab = st.selectbox("Type d'établissement", ["Public", "Privé"])

    if st.button("🔮 Prédire"):
        # Construire un DataFrame à partir d'un échantillon pour avoir toutes les colonnes
        if df_feat is not None:
            input_row = df_feat.iloc[0:1].copy()
        else:
            st.error("Les données doivent être prétraitées avant la prédiction.")
            st.stop()

        input_row['heures_devoirs'] = heures_devoirs
        input_row['motivation'] = motivation
        input_row['heures_sommeil'] = heures_sommeil
        input_row['stress'] = stress
        input_row['absences'] = absences
        input_row['temps_ecrans'] = temps_ecrans
        input_row['confiance_soi'] = confiance_soi
        input_row['perseverance'] = perseverance
        input_row['genre'] = genre
        input_row['sport'] = sport
        input_row['classe'] = classe
        input_row['type_etablissement'] = type_etab

        # Recalculer les features dérivées
        sport_num = 1 if sport == 'Oui' else 0
        input_row['score_equilibre'] = (heures_sommeil + sport_num * 2) / (heures_devoirs + temps_ecrans + 1)
        input_row['stress_absences'] = stress * absences
        input_row['motivation_travail'] = motivation * heures_devoirs

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
            if note_pred >= 14:
                color = "🟢"
            elif note_pred >= 10:
                color = "🟠"
            else:
                color = "🔴"
            st.metric(f"{color} Note prédite", f"{note_pred:.2f} / 20")

        with col_c:
            st.subheader("Probabilité de Réussite")
            st.metric("Probabilité", f"{proba_reussite:.1f}%")
            st.progress(int(proba_reussite))


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
        st.subheader("Importance des facteurs de réussite")
        st.image(shap_buf, use_container_width=True)
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
        rapport = generer_rapport_markdown(df_feat, metrics_reg, metrics_clf, path=None,
                                           metrics_nn_reg=metrics_nn_reg,
                                           metrics_nn_clf=metrics_nn_clf)
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
        st.plotly_chart(fig, use_container_width=True)
