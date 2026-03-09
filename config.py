"""
Configuration des constantes pour le projet d'analyse scolaire.
"""

# Noms des fichiers
DATA_FILE = 'test_synthetique.csv'
MODEL_FILE = 'model_final.joblib'
REPORT_FILE = 'rapport_analyse_scolaire.md'
LOG_FILE = 'analysis.log'

# Paramètres de génération de données
N_ELEVES_TRAIN = 300
N_ELEVES_TEST = 100

# Paramètres du modèle
TARGET_REG = 'note_moyenne'
TARGET_CLF = 'reussite'  # Note >= 10
SEUIL_REUSSITE = 10

# Colonnes d'identité (à exclure de la modélisation)
ID_COLUMNS = ['nom', 'prenom']

# Colonnes de notes (pour le calcul de la moyenne)
GRADE_COLUMNS = ['note_francais', 'note_maths', 'note_lecture']
