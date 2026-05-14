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
- **Auditer** l'équité algorithmique (analyse de fairness par groupe)

## 📁 Structure du projet

```
Model_performance_eleve/
├── .gitignore
├── README.md
├── requirements.txt
├── Questionnaire.html              # Formulaire HTML de collecte des données
├── main.py                         # Point d'entrée en ligne de commande
├── data/
│   ├── test_synthetique_v2.csv     # Jeu de données synthétiques
│   └── questionnaire_responses.csv # Réponses collectées via le formulaire (généré à la 1ère saisie)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Constantes et chemins
│   ├── data_utils.py               # Chargement, nettoyage, génération de données
│   ├── features.py                 # Feature engineering adaptatif
│   ├── models.py                   # Entraînement ML (XGBoost, RF, MLP, SVM)
│   ├── explainability.py           # Analyse SHAP (globale + individuelle)
│   ├── reporting.py                # Rapports Markdown & PDF
│   ├── early_warning.py            # Système d'alerte précoce (EWS)
│   ├── clustering.py               # Segmentation des profils d'élèves
│   ├── temporal.py                 # Analyse temporelle / longitudinale
│   ├── recommendations.py          # Recommandations personnalisées IA
│   ├── fairness.py                 # Audit d'équité algorithmique
│   └── api.py                      # API REST (FastAPI)
├── app/
│   ├── Accueil.py                  # Page d'accueil Streamlit
│   ├── utils_st.py                 # Utilitaires session_state
│   └── pages/
│       ├── 01_Donnees.py           # Acquisition de données (3 modes)
│       ├── 02_Preprocessing.py     # Nettoyage & feature engineering
│       ├── 03_Modelisation.py      # Entraînement & benchmark des modèles
│       ├── 04_Predictions.py       # Prédictions individuelles & par lot
│       ├── 05_Explicabilite.py     # Analyse SHAP (réussite & échec)
│       ├── 06_Rapport.py           # Export Markdown & PDF
│       ├── 07_Alertes.py           # Early Warning System
│       ├── 08_Suivi_Temporel.py    # Analyse longitudinale multi-périodes
│       ├── 09_Profils.py           # Clustering / archétypes comportementaux
│       ├── 10_Recommandations.py   # Recommandations IA + simulateur What-If
│       └── 11_Fairness.py          # Audit d'équité par groupe sensible
├── outputs/                        # Fichiers générés (modèles, logs, rapports)
└── tests/
    ├── __init__.py
    ├── test_features.py            # Tests features & data_utils
    ├── test_models.py              # Tests modèles ML
    ├── test_data_leakage.py        # Tests absence de fuite de données
    ├── test_explainability.py      # Tests SHAP
    ├── test_fairness.py            # Tests module fairness
    ├── test_api.py                 # Tests API REST
    ├── test_temporal_vectorized.py # Tests analyse temporelle
    └── test_v3_modules.py          # Tests v3.0 (EWS, clustering, recommandations)
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

Le dashboard propose **11 pages** :
1. **📂 Données** – 3 modes d'acquisition : données synthétiques, import CSV, formulaire questionnaire
2. **🔧 Preprocessing** – Nettoyage et feature engineering
3. **🤖 Modélisation** – Entraîner et évaluer les modèles ML
4. **🔮 Prédictions** – Simuler la note d'un élève ou prédire par lot
5. **📊 Explicabilité (SHAP)** – Comprendre les corrélations statistiques du modèle
6. **📝 Rapport** – Générer et télécharger (Markdown & PDF)
7. **🚨 Alertes** – Détection proactive des élèves à risque
8. **📈 Suivi Temporel** – Analyse longitudinale multi-périodes
9. **🎯 Profils** – Segmentation en groupes comportementaux
10. **💡 Recommandations** – Actions personnalisées + simulateur What-If
11. **⚖️ Fairness** – Audit d'équité algorithmique par groupe sensible

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
# 93 tests, 1 xfailed attendu
```

## 📝 Acquisition des données — 3 modes

### 1. Données synthétiques
Génération aléatoire de N élèves avec un schéma calibré pour correspondre exactement aux champs du `Questionnaire.html`. Utile pour les démonstrations et l'entraînement initial.

### 2. Import CSV
Upload d'un fichier CSV réel (séparateur `,` ou `;` détecté automatiquement). Le fichier doit contenir les mêmes colonnes que le schéma du questionnaire.

### 3. Formulaire questionnaire (nouveau)
Saisie manuelle directement dans Streamlit (onglet **📝 Questionnaire individuel**) :
- Réplique les 9 sections de `Questionnaire.html` avec des widgets natifs
- Calcule automatiquement `perseverance = mean(grit1, grit2, grit3)` et `stress_personnel = mean(stress1, stress2)`
- Sauvegarde chaque soumission comme une nouvelle ligne dans `data/questionnaire_responses.csv`
- Charge le CSV cumulé en session pour exploitation immédiate

## 🆕 Nouveautés & corrections v3.0

### 🚨 Système d'Alerte Précoce (Early Warning System)
- Score de risque composite (60 % ML + 40 % règles métier éducatives)
- Classification en 4 zones : 🟢 Serein / 🟡 Vigilance / 🟠 Alerte / 🔴 Critique
- 8 règles métier documentées avec source bibliographique (`BUSINESS_RULES`)
- Seuils des zones calculés par quartiles et documentés (`RISK_ZONES`)
- Export CSV des alertes

### 📈 Suivi Temporel & Analyse Longitudinale
- Import multi-périodes (plusieurs CSV) ou génération synthétique
- Calcul de tendances par élève (régression linéaire)
- Détection automatique de décrochage (pente significativement négative)
- Visualisation des trajectoires individuelles

### 🎯 Segmentation des Profils d'Élèves
- Clustering K-Means / DBSCAN avec détection automatique du nombre optimal de clusters
- Nommage automatique par groupes descriptifs neutres (ex : "📖 Groupe Étude Soutenue", "⚖️ Groupe Équilibre Élevé") — sans étiquettes stigmatisantes
- Projection PCA/t-SNE interactive
- Radar charts comparatifs
- Avertissement explicite : les groupes sont des regroupements statistiques, non des catégories figées

### 💡 Recommandations Personnalisées par IA
- Transformation des SHAP values en actions concrètes et actionnables
- Avertissement méthodologique affiché : les valeurs SHAP indiquent des **corrélations statistiques**, non des relations causales
- Top-N recommandations priorisées par impact (libellé "Corr. SHAP" et non "Impact SHAP")
- Simulateur What-If : modifier un paramètre → voir l'impact en temps réel
- Export de plans d'action individuels (Markdown)

### ⚖️ Audit de Fairness
- Analyse des disparités de performance (accuracy, taux de faux négatifs) par groupe sensible (`genre`, `classe`)
- Rapport formaté en Markdown avec métriques par sous-groupe

### 🔬 Correction fuite de données (Data Leakage)
- L'imputation des valeurs manquantes est déplacée **dans le pipeline sklearn** (`SimpleImputer` via `prepare_pipeline`)
- `nettoyer_donnees()` ne réalise plus d'imputation statistique — elle se limite au nettoyage structurel
- Les statistiques d'imputation sont apprises exclusivement sur le train set

### 📐 Alignement schéma Questionnaire ↔ données synthétiques
- `generer_donnees_synthetiques()` produit exactement les mêmes colonnes et plages que `Questionnaire.html`
- Champs renommés : `interet_*` → `motivation_*` (7 matières du tronc commun)
- Nouveaux champs : `grit1/2/3` + agrégat `perseverance`, `stress1/2` + agrégat `stress_personnel`
- `Qualite_tuteur` / `Qualite_mentor` sur échelle 1–10
- 5 niveaux de fréquence catégoriels (`jamais` → `toujours`) pour les variables comportementales

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

Le meilleur modèle en régression et en classification est sélectionné automatiquement par score de validation croisée.

## 📈 Métriques d'Évaluation

### Régression
- **R²** : Part de variance expliquée
- **MAE** : Erreur absolue moyenne
- **RMSE** : Erreur quadratique moyenne

### Classification
- **Accuracy** : Taux de bonne classification
- **F1-Score** : Harmonie précision/rappel
- **Precision / Recall** : Détaillés pour la classe minoritaire (échec)

## 🔧 Format des fichiers CSV

- **Séparateur** : point-virgule (`;`) — détection automatique supportée
- **Encodage** : `UTF-8 avec BOM` (`utf-8-sig`)
- **Colonnes clés** : voir `src/data_utils.py` et `Questionnaire.html` pour le schéma complet

## ⚠️ Avertissements Méthodologiques

- Les prédictions du modèle reposent sur des **corrélations statistiques**, non sur des relations causales établies. Une corrélation entre une variable (ex : temps d'écran) et les résultats scolaires ne signifie pas que l'une cause l'autre.
- Les groupes de profils (clustering) sont des **regroupements statistiques exploratoires**, non des catégories diagnostiques. Un élève peut appartenir à plusieurs groupes selon les paramètres.
- Le seuil de réussite (10/20) est conforme au barème officiel de l'Éducation Nationale française. Il peut être ajusté dans `src/config.py` selon le référentiel de l'établissement.

## 📄 Licence

Ce projet est développé pour des fins éducatives et de recherche.

---
*Système d'Analyse Scolaire EduStats v3.0*
