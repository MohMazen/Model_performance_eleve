"""
Configuration des constantes pour le projet d'analyse scolaire.
"""

import os

# Noms des fichiers
DATA_FILE = os.path.join('data', 'test_synthetique.csv')
MODEL_FILE = os.path.join('outputs', 'model_final.joblib')
REPORT_FILE = os.path.join('outputs', 'rapport_analyse_scolaire.md')
LOG_FILE = os.path.join('outputs', 'analysis.log')

# Paramètres de génération de données
N_ELEVES_TRAIN = 300
N_ELEVES_TEST = 100

# Paramètres du modèle
TARGET_REG = 'note_moyenne'
TARGET_CLF = 'reussite'  # Note >= 10
SEUIL_REUSSITE = 10

# Colonnes d'identité (à exclure de la modélisation)
ID_COLUMNS = [c.lower() for c in ['Nom', 'Prenom', 'Adresse', 'Etablissement', 'Nom_plateforme', 'Autre_etat_psy', 'Specialite1ere_1_nom', 'Specialite1ere_2_nom', 'Specialite1ere_3_nom', 'SpecialiteTerm_1_nom', 'SpecialiteTerm_2_nom', 'Soutien_matieres', 'Etat_psychologique']]

# Colonnes de notes (pour le calcul de la moyenne)
GRADE_COLUMNS = ['note_francais', 'note_maths', 'note_histoire_geo', 'note_sciences']

# Colonnes à exclure avant modélisation (Identité + Cibles directes)
COLS_TO_DROP = ID_COLUMNS + GRADE_COLUMNS + ['heure_coucher', 'heure_lever']
