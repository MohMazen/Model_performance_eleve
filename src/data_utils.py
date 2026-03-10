"""
Utilitaires pour le chargement, le nettoyage et la génération de données.
"""
import pandas as pd
import numpy as np
import logging
from src.config import DATA_FILE, ID_COLUMNS, GRADE_COLUMNS

logger = logging.getLogger(__name__)


def valider_schema(df, colonnes_requises):
    """
    Vérifie que les colonnes requises existent dans le DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à valider.
    colonnes_requises : list[str]
        Liste des noms de colonnes attendus.

    Raises
    ------
    ValueError
        Si une ou plusieurs colonnes requises sont absentes.
    """
    manquantes = [col for col in colonnes_requises if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {manquantes}")
    logger.info("Validation du schéma réussie.")


def generer_donnees_synthetiques(n_eleves=300):
    """
    Génère un jeu de données synthétiques réaliste.
    """
    logger.info(f"Génération de {n_eleves} données synthétiques...")
    np.random.seed(42)  # Reproductibilité

    data = {
        'age': np.random.normal(15.5, 1.2, n_eleves).clip(12, 19),
        'genre': np.random.choice(['M', 'F'], n_eleves),
        'absences': np.random.poisson(3, n_eleves).clip(0, 30),
        'retards': np.random.poisson(1, n_eleves).clip(0, 15),
        'heures_devoirs': np.random.gamma(4, 1.5, n_eleves).clip(0.5, 15),
        'heures_sommeil': np.random.normal(7.5, 1, n_eleves).clip(4, 11),
        'temps_ecrans': np.random.gamma(3, 1.2, n_eleves).clip(0, 12),
        'motivation': np.random.beta(5, 2, n_eleves) * 9 + 1,
        'confiance_soi': np.random.beta(4, 2, n_eleves) * 9 + 1,
        'stress': np.random.beta(3, 3, n_eleves) * 9 + 1,
        'perseverance': np.random.beta(5, 3, n_eleves) * 9 + 1,
        'niveau_etudes_parents': np.random.choice(['Primaire', 'Secondaire', 'Supérieur'], n_eleves, p=[0.2, 0.5, 0.3]),
        'revenus_famille': np.random.choice(['Faible', 'Moyen', 'Élevé'], n_eleves, p=[0.3, 0.5, 0.2]),
        'suivi_parental': np.random.choice(['Faible', 'Modéré', 'Fort'], n_eleves, p=[0.2, 0.5, 0.3]),
        'nombre_fratrie': np.random.poisson(1.5, n_eleves).clip(0, 6),
        'taille_classe': np.random.normal(25, 4, n_eleves).clip(15, 35),
        'type_etablissement': np.random.choice(['Public', 'Privé'], n_eleves, p=[0.8, 0.2]),
        'climat_scolaire': np.random.beta(3, 2, n_eleves) * 9 + 1,
        'soutien_scolaire': np.random.choice(['Oui', 'Non'], n_eleves, p=[0.3, 0.7]),
        'nom': [f"NOM_{i}" for i in range(n_eleves)],
        'prenom': [f"PRENOM_{i}" for i in range(n_eleves)],
        'etat_civil': np.random.choice(['Célibataire', 'Marié', 'Divorcé', 'Veuf'], n_eleves, p=[0.9, 0.05, 0.03, 0.02]),
        'duree_trajet': np.random.gamma(2, 15, n_eleves).clip(5, 120),
        'heure_coucher': [f"{np.random.randint(21, 24)}h{np.random.choice([0, 15, 30, 45]):02d}" for _ in range(n_eleves)],
        'heure_lever': [f"{np.random.randint(6, 9)}h{np.random.choice([0, 15, 30, 45]):02d}" for _ in range(n_eleves)],
        'classe': np.random.choice(['1ère', 'Terminale'], n_eleves),
        'education_physique': np.random.normal(12, 3, n_eleves).clip(0, 20),
        'matiere_enseignement_scientifique': np.random.normal(11, 4, n_eleves).clip(0, 20),
        'specialite1': np.random.normal(13, 3, n_eleves).clip(0, 20),
        'specialite2': np.random.normal(13, 3, n_eleves).clip(0, 20),
        'specialite3': np.random.normal(13, 3, n_eleves).clip(0, 20),
        'sport': np.random.choice(['Oui', 'Non'], n_eleves, p=[0.6, 0.4]),
        'musique': np.random.choice(['Oui', 'Non'], n_eleves, p=[0.4, 0.6]),
        'lecture_loisir': np.random.gamma(1.5, 2, n_eleves).clip(0, 10)
    }

    df = pd.DataFrame(data)

    # Création des cibles réalistes
    for i in range(n_eleves):
        bonus = 0
        bonus += df.loc[i, 'heures_devoirs'] * 0.4
        bonus += df.loc[i, 'motivation'] * 0.2
        bonus -= df.loc[i, 'temps_ecrans'] * 0.3
        bonus -= df.loc[i, 'absences'] * 0.05
        if df.loc[i, 'duree_trajet'] > 60:
            bonus -= 0.7

        df.loc[i, 'note_francais'] = np.clip(np.random.normal(13, 2) + bonus, 0, 20)
        df.loc[i, 'note_maths'] = np.clip(np.random.normal(12, 3) + bonus, 0, 20)
        df.loc[i, 'note_lecture'] = np.clip(np.random.normal(14, 2) + bonus, 0, 20)

    df['note_moyenne'] = df[GRADE_COLUMNS].mean(axis=1)
    return df


def charger_donnees(chemin):
    """
    Charge les données depuis CSV.
    """
    try:
        df = pd.read_csv(chemin, sep=';', encoding='utf-8-sig')
        logger.info(f"Données chargées : {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erreur de chargement de {chemin}: {e}")
        return None


def nettoyer_donnees(df):
    """
    Nettoyage basique des données.
    """
    if df is None:
        return None
    df_clean = df.copy()

    # Remplissage des valeurs manquantes
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    return df_clean
