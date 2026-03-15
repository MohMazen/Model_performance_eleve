# 🏫 Analyse Prédictive des Performances Scolaires

## 📋 Description

Ce projet implémente une méthodologie complète d'analyse des données éducatives pour prédire et comprendre les facteurs de réussite scolaire. Il transforme les données brutes d'élèves en insights actionnables pour les équipes pédagogiques.

## 🎯 Objectifs

- **Prédire** la performance scolaire des élèves
- **Identifier** les facteurs clés de réussite et d'échec
- **Fournir** des recommandations pédagogiques basées sur les données
- **Générer** des rapports automatisés pour la prise de décision

## 📁 Structure du projet

```
Model_performance_eleve/
├── .gitignore
├── README.md
├── requirements.txt
├── Questionnaire.html
├── main.py                     # Point d'entrée en ligne de commande
├── data/
│   └── test_synthetique.csv    # Jeu de données élèves
├── src/
│   ├── __init__.py
│   ├── config.py               # Constantes et chemins
│   ├── data_utils.py           # Chargement, nettoyage, génération de données
│   ├── features.py             # Feature engineering
│   ├── models.py               # Entraînement ML (XGBoost, Random Forest, Réseau de Neurones)
│   ├── explainability.py       # Analyse SHAP
│   └── reporting.py            # Génération de rapports et visualisations
├── app/
│   └── dashboard.py            # Dashboard Streamlit 6 pages
├── outputs/                    # Fichiers générés (modèles, logs, rapports)
│   └── .gitkeep
└── tests/
    ├── __init__.py
    └── test_features.py        # Tests unitaires
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
streamlit run app/dashboard.py
```

Le dashboard propose 6 pages :
1. **📂 Données** – Générer des données synthétiques ou charger un CSV
2. **🔧 Preprocessing** – Nettoyage et feature engineering
3. **🤖 Modélisation** – Entraîner et évaluer les modèles ML
4. **🔮 Prédictions** – Simuler la note d'un élève
6. **📊 Explicabilité (SHAP)** – Comprendre les décisions du modèle
7. **📝 Rapport** – Générer et télécharger le rapport Markdown

### Nouveautés v2.1
- **Multi-Output Regression** : Prédiction simultanée de la moyenne et des notes par matière (Français, Maths, Histoire-Géo, Sciences).
- **📋 Prédictions par Élève** : Nouvel onglet permettant de visualiser et télécharger (CSV) les prédictions pour l'ensemble des élèves.
- **Gestion Avancée des Modèles** : Possibilité de nommer, sauvegarder et charger différents fichiers `.joblib`.
- **Aide au Diagnostic** : Explications intégrées pour les métriques d'évaluation et la matrice de confusion.

### Lancer l'analyse en ligne de commande
```bash
python main.py
```

Cette commande :
1. Génère ou charge les données (`data/test_synthetique.csv`)
2. Applique le preprocessing et le feature engineering
3. Entraîne un modèle XGBoost (régression) et un Random Forest (classification) avec séparation train/test
4. Lance l'analyse SHAP
5. Sauvegarde les modèles dans `outputs/model_final.joblib`
6. Génère un rapport dans `outputs/rapport_analyse_scolaire.md`

### Lancer les tests unitaires
```bash
python -m pytest tests/
```

## 📊 Structure des Données

### Variables d'entrée attendues

#### 🧑‍🎓 Facteurs Individuels
- `age`: Âge de l'élève
- `genre`: Genre (M/F)
- `absences`, `retards`: Indicateurs d'assiduité
- `heures_devoirs`: Temps de travail personnel
- `heures_sommeil`: Heures de sommeil par nuit
- `temps_ecrans`: Temps d'écrans quotidien (heures)
- `motivation`, `confiance_soi`, `stress`, `perseverance`: Facteurs psychologiques (1-10)
- `heure_coucher`, `heure_lever`: Horaires de repos (format `22h30`)
- `duree_trajet`: Temps de transport en minutes

#### 👨‍👩‍👧‍👦 Facteurs Familiaux
- `niveau_etudes_parents`: Niveau d'éducation des parents
- `revenus_famille`: Niveau socio-économique
- `suivi_parental`: Intensité du suivi parental
- `nombre_fratrie`: Nombre de frères et sœurs

#### 🏫 Facteurs Scolaires
- `classe`: Niveau (4ème ou 3ème)
- `taille_classe`: Nombre d'élèves par classe
- `type_etablissement`: Public/Privé
- `climat_scolaire`: Qualité de l'environnement scolaire
- `soutien_scolaire`: Aide supplémentaire (Oui/Non)

#### 📝 Notes
- `note_francais`, `note_maths`, `note_histoire_geo`, `note_sciences`: Notes par matière (calculées automatiquement dans la génération synthétique)
- `note_moyenne`: Moyenne générale (variable cible)

### Variables créées par le Feature Engineering
- `score_equilibre`: ratio (sommeil + sport) / (devoirs + écrans + 1)
- `stress_absences`: interaction stress × absences
- `motivation_travail`: interaction motivation × heures de devoirs
- `reussite`: 1 si note_moyenne ≥ 10, sinon 0 (variable cible binaire)
- `heure_coucher_num`, `heure_lever_num`: horaires convertis en float

## 🤖 Modèles ML

| Tâche | Algorithme | Paramètres clés |
|-------|-----------|-----------------|
| Régression (note) | XGBoost | RandomizedSearchCV, 3-fold CV |
| Classification (réussite) | Random Forest | `class_weight='balanced'` pour gérer le déséquilibre |
| Régression (note) | Réseau de Neurones (MLPRegressor) | RandomizedSearchCV, 3-fold CV |
| Classification (réussite) | Réseau de Neurones (MLPClassifier) | RandomizedSearchCV, 3-fold CV |

## 📈 Métriques d'Évaluation

### Régression
- **R²**: Part de variance expliquée
- **MAE**: Erreur absolue moyenne
- **RMSE**: Erreur quadratique moyenne

### Classification
- **Accuracy**: Taux de bonne classification
- **F1-Score**: Harmonie précision/rappel
- **Precision / Recall**: Détaillés pour la classe minoritaire (échec)

## 🐛 Corrections de Bugs apportées

| Bug | Correction |
|-----|-----------|
| Data leakage : évaluation sur données d'entraînement | `train_test_split` (80/20) dans `main.py` |
| Classes déséquilibrées (~4% d'échec) | `class_weight='balanced'` dans `RandomForestClassifier` |
| `except:` nu (attrape tout) | `except (FileNotFoundError, KeyError, Exception) as e:` avec logging |
| Absence de validation de schéma | Fonction `valider_schema()` dans `src/data_utils.py` |
| Dashboard incompatible avec le pipeline | Colonnes correctement droppées dans `app/dashboard.py` |

## 🔧 Format des fichiers CSV

Pour assurer une compatibilité optimale entre le questionnaire HTML et l'analyse Python :
- **Séparateur** : point-virgule (`;`)
- **Encodage** : `UTF-8 avec BOM` (`utf-8-sig`)

## 📄 Licence

Ce projet est développé pour des fins éducatives et de recherche.

---
*Système d'Analyse Scolaire v2.1*
