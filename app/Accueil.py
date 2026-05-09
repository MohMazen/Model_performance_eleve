import streamlit as st

st.set_page_config(
    page_title="EduStats – Analyse Scolaire",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Bienvenue sur EduStats")
st.markdown("---")
st.markdown("""
### Analyse Prédictive des Performances Scolaires v2.1

Utilisez le menu de gauche pour naviguer à travers les différentes étapes de l'analyse :
1. **📂 Données** : Chargez vos données ou générez un dataset synthétique.
2. **🔧 Preprocessing** : Nettoyez et préparez les données pour l'entraînement.
3. **🤖 Modélisation** : Entraînez divers algorithmes d'IA (XGBoost, Random Forest, Réseaux de Neurones).
4. **🔮 Prédictions** : Prédisez les résultats globaux ou saisissez un profil spécifique.
5. **📊 Explicabilité (SHAP)** : Comprenez pourquoi des prédictions spécifiques (succès ou échec) sont prises.
6. **📝 Rapport** : Exportez les résultats finaux.
""")

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v2.1")
