import streamlit as st

st.set_page_config(
    page_title="EduStats – Analyse Scolaire",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Bienvenue sur EduStats")
st.markdown("---")
st.markdown("""
### Analyse Prédictive des Performances Scolaires v3.0

Utilisez le menu de gauche pour naviguer à travers les différentes étapes de l'analyse :
1. **📂 Données** : Chargez vos données ou générez un dataset synthétique.
2. **🔧 Preprocessing** : Nettoyez et préparez les données pour l'entraînement.
3. **🤖 Modélisation** : Entraînez divers algorithmes d'IA (XGBoost, Random Forest, Réseaux de Neurones, SVM).
4. **🔮 Prédictions** : Prédisez les résultats globaux ou saisissez un profil spécifique.
5. **📊 Explicabilité (SHAP)** : Comprenez pourquoi des prédictions spécifiques sont prises.
6. **📝 Rapport** : Exportez les résultats finaux (Markdown & PDF).

---

### 🆕 Nouveautés v3.0
7. **🚨 Alertes Précoces** : Détection proactive des élèves à risque avec scoring composite (ML + règles métier).
8. **📈 Suivi Temporel** : Analyse longitudinale multi-périodes et détection de décrochage.
9. **🎯 Profils** : Segmentation automatique en archétypes comportementaux (clustering K-Means).
10. **💡 Recommandations** : Actions concrètes personnalisées par élève + simulateur What-If.
""")

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

