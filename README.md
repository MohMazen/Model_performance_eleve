# 🏫 Analyse Prédictive des Performances Scolaires — EduStats v3.0

## 📋 Description

Ce projet implémente une méthodologie complète d'analyse des données éducatives pour prédire et comprendre les facteurs de réussite scolaire. Il transforme les données brutes d'élèves en insights actionnables pour les équipes pédagogiques.

## 🎯 Objectifs

- **Prédire** la performance scolaire des élèves
- **Identifier** les facteurs clés de réussite et d'échec
- **Détecter** proactivement les élèves à risque (Early Warning System)
- **Recommander** des actions pédagogiques personnalisées basées sur l'IA
- **Suivre** l'évolution des élèves dans le temps (analyse longitudinale)
- **Segmenter** les profils d'élèves en archétypes comportementaux
- **Générer** des rapports automatisés (Markdown & PDF)
- **Exposer** les prédictions via une API REST

## 📁 Structure du projet

```
Model_performance_eleve/
├── .gitignore
├── README.md
├── requirements.txt
├── Questionnaire.html
├── main.py                         # Point d'entrée en ligne de commande
├── data/
│   └── test_synthetique_v2.csv     # Jeu de données élèves
├── src/
│   ├── __init__.py
│   ├── config.py                   # Constantes et chemins
│   ├── data_utils.py               # Chargement, nettoyage, génération de données
│   ├── features.py                 # Feature engineering adaptatif
│   ├── models.py                   # Entraînement ML (XGBoost, RF, MLP, SVM)
│   ├── explainability.py           # Analyse SHAP (globale + individuelle)
│   ├── reporting.py                # Rapports Markdown & PDF
│   ├── early_warning.py            # 🆕 Système d'alerte précoce
│   ├── clustering.py               # 🆕 Segmentation des profils d'élèves
│   ├── temporal.py                 # 🆕 Analyse temporelle / longitudinale
│   ├── recommendations.py          # 🆕 Recommandations personnalisées IA
│   └── api.py                      # 🆕 API REST (FastAPI)
├── app/
│   ├── Accueil.py                  # Page d'accueil Streamlit
│   ├── utils_st.py                 # Utilitaires session_state
│   └── pages/
│       ├── 01_Donnees.py           # Chargement / génération de données
│       ├── 02_Preprocessing.py     # Nettoyage & feature engineering
│       ├── 03_Modelisation.py      # Entraînement & benchmark des modèles
│       ├── 04_Predictions.py       # Prédictions individuelles & par lot
│       ├── 05_Explicabilite.py     # Analyse SHAP (réussite & échec)
│       ├── 06_Rapport.py           # Export Markdown & PDF
│       ├── 07_Alertes.py           # 🆕 Early Warning System
│       ├── 08_Suivi_Temporel.py    # 🆕 Analyse longitudinale
│       ├── 09_Profils.py           # 🆕 Clustering / archétypes
│       └── 10_Recommandations.py   # 🆕 Recommandations IA + What-If
├── outputs/                        # Fichiers générés (modèles, logs, rapports)
└── tests/
    ├── __init__.py
    ├── test_features.py            # Tests features & data_utils
    ├── test_models.py              # Tests modèles ML
    └── test_v3_modules.py          # 🆕 Tests v3.0 (21 tests)
```

## 🔧 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### Lancer le dashboard interactif (recommandé)
```bash
streamlit run app/Accueil.py
```

Le dashboard propose **10 pages** :
1. **📂 Données** – Générer des données synthétiques ou charger un CSV
2. **🔧 Preprocessing** – Nettoyage et feature engineering
3. **🤖 Modélisation** – Entraîner et évaluer les modèles ML
4. **🔮 Prédictions** – Simuler la note d'un élève ou prédire par lot
5. **📊 Explicabilité (SHAP)** – Comprendre les décisions du modèle
6. **📝 Rapport** – Générer et télécharger (Markdown & PDF)
7. **🚨 Alertes** – Détection proactive des élèves à risque
8. **📈 Suivi Temporel** – Analyse longitudinale multi-périodes
9. **🎯 Profils** – Segmentation en archétypes comportementaux
10. **💡 Recommandations** – Actions personnalisées + simulateur What-If

### Lancer l'API REST
```bash
uvicorn src.api:app --reload --port 8000
```

Endpoints disponibles :
- `GET /docs` — Documentation Swagger interactive
- `POST /predict` — Prédiction individuelle
- `POST /predict/batch` — Prédiction par lot (CSV upload)
- `GET /metrics` — Informations sur le modèle courant
- `GET /health` — Health check

### Lancer l'analyse en ligne de commande
```bash
python main.py
```

### Lancer les tests unitaires
```bash
python -m pytest tests/ -v
```

## 🆕 Nouveautés v3.0

### 🚨 Système d'Alerte Précoce (Early Warning System)
- Score de risque composite (60% ML + 40% règles métier éducatives)
- Classification en 4 zones : 🟢 Serein / 🟡 Vigilance / 🟠 Alerte / 🔴 Critique
- Décomposition des facteurs de risque avec recommandations par règle
- Export CSV des alertes

### 📈 Suivi Temporel & Analyse Longitudinale
- Import multi-périodes (plusieurs CSV) ou génération synthétique
- Calcul de tendances par élève (régression linéaire)
- Détection automatique de décrochage (pente significativement négative)
- Visualisation des trajectoires individuelles

### 🎯 Clustering des Profils d'Élèves
- Segmentation K-Means / DBSCAN avec détection automatique du nombre optimal de clusters
- Nommage automatique des archétypes (ex: "📖 Le Studieux Stressé", "⚖️ L'Équilibré Performant")
- Projection PCA/t-SNE interactive
- Radar charts comparatifs

### 💡 Recommandations Personnalisées par IA
- Transformation des SHAP values en actions concrètes et actionnables
- Top-N recommandations priorisées par impact
- Simulateur What-If : modifier un paramètre → voir l'impact en temps réel
- Export de plans d'action individuels (Markdown)

### 📄 Export PDF & API REST
- Rapport PDF professionnel avec graphiques intégrés (via reportlab)
- API REST FastAPI avec documentation Swagger auto-générée

## 🤖 Modèles ML

| Tâche | Algorithme | Paramètres clés |
|-------|-----------|-----------------|
| Régression (note) | XGBoost | RandomizedSearchCV, 3-fold CV |
| Classification (réussite) | Random Forest | `class_weight='balanced'` |
| Régression (note) | Réseau de Neurones (MLP) | RandomizedSearchCV, 3-fold CV |
| Classification (réussite) | Réseau de Neurones (MLP) | RandomizedSearchCV, 3-fold CV |
| Régression (note) | SVM (SVR) | RandomizedSearchCV, 3-fold CV |
| Classification (réussite) | SVM (SVC) | `probability=True` |

## 📈 Métriques d'Évaluation

### Régression
- **R²**: Part de variance expliquée
- **MAE**: Erreur absolue moyenne
- **RMSE**: Erreur quadratique moyenne

### Classification
- **Accuracy**: Taux de bonne classification
- **F1-Score**: Harmonie précision/rappel
- **Precision / Recall**: Détaillés pour la classe minoritaire (échec)

## 🔧 Format des fichiers CSV

- **Séparateur** : point-virgule (`;`) — détection automatique supportée
- **Encodage** : `UTF-8 avec BOM` (`utf-8-sig`)

## 📄 Licence

Ce projet est développé pour des fins éducatives et de recherche.

---
*Système d'Analyse Scolaire EduStats v3.0*
