"""
Tableau de Bord Interactif - Analyse de la Performance Scolaire.
"""
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from config import MODEL_FILE, DATA_FILE
from features import add_advanced_features, prenttoyer_horaires

st.set_page_config(page_title="EduStats Dashboard", layout="wide")

st.title("🎓 EduStats : Analyse Prédictive de la Réussite")
st.write("Ce tableau de bord permet d'explorer les facteurs de réussite scolaire et de prédire les notes des élèves.")

# 1. Chargement du modèle
if os.path.exists(MODEL_FILE):
    models = joblib.load(MODEL_FILE)
    reg_model = models['reg']
    st.success("✅ Modèle prédictif chargé avec succès.")
else:
    st.error("⚠️ Modèle non trouvé. Veuillez d'abord exécuter l'analyse complète (main_refactored.py).")
    st.stop()

# 2. Chargement des données
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8-sig')
    st.sidebar.header("Données")
    st.sidebar.info(f"Échantillon : {len(df)} élèves")
else:
    st.warning("Aucune donnée trouvée.")
    st.stop()

# 3. Visualisations
tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble", "🤖 Prédictions", "🔍 Explicabilité"])

with tab1:
    st.header("Analyse Globale")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='note_moyenne', nbins=20, title="Distribution des Notes Moyennes")
        st.plotly_chart(fig)
        
    with col2:
        fig = px.scatter(df, x='heures_devoirs', y='note_moyenne', color='motivation', 
                        title="Devoirs vs Performance")
        st.plotly_chart(fig)

with tab2:
    st.header("Simulateur de Prédiction")
    st.write("Modifiez les paramètres pour voir l'impact sur la note prédite.")
    
    # Formulaire de saisie pour un nouvel élève
    col1, col2, col3 = st.columns(3)
    with col1:
        dev = st.slider("Heures de devoirs", 0, 15, 4)
        mot = st.slider("Motivation (1-10)", 1, 10, 7)
    with col2:
        slp = st.slider("Heures de sommeil", 4, 11, 8)
        str_val = st.slider("Stress (1-10)", 1, 10, 5)
    with col3:
        abs_v = st.number_input("Absences", 0, 30, 2)
        genre = st.selectbox("Genre", ["M", "F"])

    # Préparation du vecteur
    # Pour simplifier, on prend un échantillon et on remplace les valeurs
    input_data = df.iloc[0:1].copy()
    input_data['heures_devoirs'] = dev
    input_data['motivation'] = mot
    input_data['heures_sommeil'] = slp
    input_data['stress'] = str_val
    input_data['absences'] = abs_v
    input_data['genre'] = genre

    # Preprocessing
    input_data = prenttoyer_horaires(input_data)
    input_data = add_advanced_features(input_data)
    
    # Prediction
    pred = reg_model.predict(input_data)[0]
    st.metric("Note Moyenne Prédite", f"{pred:.2f} / 20")

with tab3:
    st.header("Explicabilité (SHAP)")
    st.info("Ici, nous afficherions les graphiques SHAP pour comprendre les décisions du modèle.")
    # (Note: Les plots SHAP natifs matplotlib sont difficiles à intégrer dans Streamlit sans tweaks spécifiques)
    st.image("https://shap.readthedocs.io/en/latest/_images/shap_header.png", width=400)
