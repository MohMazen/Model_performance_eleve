"""
Utilitaires pour le chargement, le nettoyage et la génération de données.
Version alignée avec le Questionnaire.html (79 champs).
"""
import pandas as pd
import numpy as np
import logging
import io
from src.config import DATA_FILE, ID_COLUMNS, GRADE_COLUMNS

logger = logging.getLogger(__name__)


def valider_schema(df, colonnes_requises):
    """Vérifie que les colonnes requises existent dans le DataFrame."""
    manquantes = [col for col in colonnes_requises if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {manquantes}")
    logger.info("Validation du schéma réussie.")


def generer_donnees_synthetiques(n_eleves=300, classes_selectionnees=None):
    """
    Génère un jeu de données synthétiques 100% compatible avec Questionnaire.html.
    """
    logger.info(f"Génération de {n_eleves} données synthétiques (Schéma Questionnaire)...")
    np.random.seed(42)

    if classes_selectionnees is None:
        classes_selectionnees = ['4ème', '3ème']
    
    # Mapping exact des noms de classes du questionnaire (Select HTML)
    map_classes = {
        'Sixième': '6eme', 'Cinquième': '5eme', 'Quatrième': '4eme', 
        'Troisième': '3eme', 'Seconde': '2nde', 'Première': '1ere', 'Terminale': 'terminale'
    }
    # On s'assure d'utiliser les valeurs internes du questionnaire
    classes_values = [map_classes.get(c, c) for c in classes_selectionnees]

    data = {} # type: dict

    # --- SECTION 1 : État Civil ---
    data['Nom'] = [f"NOM_{i}" for i in range(n_eleves)]
    data['Prenom'] = [f"PRENOM_{i}" for i in range(n_eleves)]
    data['Adresse'] = [f"{np.random.randint(1, 100)} Rue de l'Ecole" for _ in range(n_eleves)]
    data['Age'] = np.random.normal(15.5, 1.2, n_eleves).clip(10, 20).astype(int)
    data['Classe'] = np.random.choice(classes_values, n_eleves)
    data['Etablissement'] = np.random.choice(['College Pasteur', 'Lycee Curie', 'Inst. Voltaire'], n_eleves)
    data['Duree_trajet_AR_min'] = np.random.gamma(2, 15, n_eleves).clip(5, 120).astype(int)

    # --- SECTION 2 : Capacités Organisationnelles ---
    data['Organisation'] = np.random.randint(1, 11, n_eleves)
    data['Gestion_temps'] = np.random.choice(['faible', 'moyen', 'fort'], n_eleves, p=[0.2, 0.5, 0.3])

    # --- SECTION 3 : Motivation et Engagement ---
    matieres = ['francais', 'maths', 'HGEMC', 'anglais', 'arabe', 'sciences', 'EPS', 'ens_scientifique']
    for m in matieres:
        data[f'Interet_{m}'] = np.random.randint(0, 11, n_eleves)
    
    # Pour simplifier, on laisse les spécialités vides ou avec des valeurs par défaut
    for i in range(1, 4):
        data[f'Specialite1ere_{i}_nom'] = ""
        data[f'Specialite1ere_{i}_interet'] = 5
    for i in range(1, 3):
        data[f'SpecialiteTerm_{i}_nom'] = ""
        data[f'SpecialiteTerm_{i}_interet'] = 5
        
    data['Motivation_famille'] = np.random.randint(0, 11, n_eleves)
    data['Motivation_recompenses'] = np.random.randint(0, 11, n_eleves)
    for i in range(1, 4):
        data[f'Perseverance_{i}'] = np.random.randint(1, 6, n_eleves)

    # --- SECTION 4 : Temps d'étude et écrans ---
    data['Heures_etude_soir'] = np.random.gamma(4, 0.5, n_eleves).clip(0, 6)
    data['Heures_jeux_video'] = np.random.gamma(2, 0.5, n_eleves).clip(0, 5)
    data['Heures_reseaux_sociaux'] = np.random.gamma(2, 0.5, n_eleves).clip(0, 5)
    data['Heures_streaming'] = np.random.gamma(2, 0.5, n_eleves).clip(0, 5)
    data['Heures_sites_educatifs'] = np.random.gamma(1.5, 0.4, n_eleves).clip(0, 3)

    # --- SECTION 5 : Sommeil et bien-être ---
    data['Heures_sommeil'] = np.random.normal(7.5, 1, n_eleves).clip(4, 11)
    data['Qualite_sommeil'] = np.random.randint(1, 11, n_eleves)
    
    h_coucher = np.random.randint(21, 24, n_eleves)
    m_coucher = np.random.choice([0, 15, 30, 45], n_eleves)
    data['Heure_coucher'] = [f"{h:02d}:{m:02d}" for h, m in zip(h_coucher, m_coucher)]
    
    h_lever = np.random.randint(6, 9, n_eleves)
    m_lever = np.random.choice([0, 15, 30, 45], n_eleves)
    data['Heure_lever'] = [f"{h:02d}:{m:02d}" for h, m in zip(h_lever, m_lever)]
    
    data['Stress_1'] = np.random.randint(0, 5, n_eleves)
    data['Stress_2'] = np.random.randint(0, 5, n_eleves)

    # --- SECTION 6 : Nutrition ---
    data['Nb_repas'] = np.random.choice(['1', '2', '3', 'plus'], n_eleves)
    data['Repas_equilibres'] = np.random.choice(['jamais', 'parfois', 'souvent', 'toujours'], n_eleves)
    data['Activite_sportive'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Heures_activite_physique'] = np.random.gamma(2, 1, n_eleves).clip(0, 10)
    data['Niveau_sportif'] = np.random.choice(['debutant', 'intermediaire', 'confirme'], n_eleves)

    # --- SECTION 7 : Relations ---
    data['Pref_travail'] = np.random.choice(['seul', 'groupe', 'indifferent'], n_eleves)
    data['Soutien_mutuel'] = np.random.choice(['jamais', 'rarement', 'parfois', 'souvent'], n_eleves)
    data['Tuteur'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Qualite_tuteur'] = np.random.randint(0, 101, n_eleves)
    data['Mentor'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Qualite_mentor'] = np.random.randint(0, 101, n_eleves)

    # --- SECTION 8 : Confiance ---
    data['Confiance_soi'] = np.random.randint(1, 11, n_eleves)
    data['Estime_soi'] = np.random.randint(1, 11, n_eleves)
    data['Evitement'] = np.random.choice(['jamais', 'rarement', 'parfois', 'souvent'], n_eleves)
    data['Abandon'] = np.random.choice(['jamais', 'rarement', 'parfois', 'souvent'], n_eleves)
    data['Etat_psychologique'] = "Serein"
    data['Autre_etat_psy'] = ""
    data['Suivi_psy'] = np.random.choice(['oui', 'non'], n_eleves, p=[0.1, 0.9])
    data['Stress_examens'] = np.random.randint(1, 11, n_eleves)
    data['Pression_familiale'] = np.random.randint(1, 11, n_eleves)

    # --- SECTION 9 : Environnement ---
    data['Bureau_personnel'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Calme_maison'] = np.random.randint(1, 11, n_eleves)
    data['Lumiere_adaptee'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Perturbations'] = np.random.choice(['jamais', 'rarement', 'parfois', 'souvent'], n_eleves)
    data['Temps_libre'] = np.random.randint(0, 6, n_eleves)
    data['Taches_menageres'] = np.random.choice(['aucune', 'peu', 'moyen', 'beaucoup'], n_eleves)
    data['Heures_garde_freres_soeurs'] = np.random.randint(0, 5, n_eleves)
    data['Soutien_matieres'] = ""
    data['Heures_soutien_francais'] = 0
    data['Heures_soutien_maths'] = 0
    data['Heures_soutien_sciences'] = 0
    data['Heures_soutien_anglais'] = 0
    data['Heures_soutien_histoire_geo'] = 0
    data['Abonnement_plateforme'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Nom_plateforme'] = ""
    data['Frequence_mediatheque'] = np.random.choice(['jamais', 'rarement', 'parfois', 'souvent'], n_eleves)

    df = pd.DataFrame(data)

    # --- CALCUL DES CIBLES RÉALISTES ---
    for i in range(n_eleves):
        bonus = 0
        bonus += df.loc[i, 'Heures_etude_soir'] * 0.5
        bonus += (df.loc[i, 'Interet_maths'] + df.loc[i, 'Interet_francais']) * 0.1
        bonus -= (df.loc[i, 'Heures_jeux_video'] + df.loc[i, 'Heures_reseaux_sociaux']) * 0.3
        bonus -= (df.loc[i, 'Stress_1'] + df.loc[i, 'Stress_2']) * 0.2
        if df.loc[i, 'Activite_sportive'] == 'oui': bonus += 0.5
        if df.loc[i, 'Duree_trajet_AR_min'] > 60: bonus -= 0.7

        df.loc[i, 'note_francais'] = np.clip(np.random.normal(12, 2) + bonus, 0, 20)
        df.loc[i, 'note_maths'] = np.clip(np.random.normal(11, 3) + bonus, 0, 20)
        df.loc[i, 'note_histoire_geo'] = np.clip(np.random.normal(13, 2) + bonus, 0, 20)
        df.loc[i, 'note_sciences'] = np.clip(np.random.normal(12, 2.5) + bonus, 0, 20)

    df['note_moyenne'] = df[GRADE_COLUMNS].mean(axis=1)
    return df


def charger_donnees(chemin):
    """Charge les données depuis CSV."""
    try:
        df = pd.read_csv(chemin, sep=';', encoding='utf-8-sig')
        logger.info(f"Données chargées : {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erreur de chargement de {chemin}: {e}")
        return None


def nettoyer_donnees(df):
    """Nettoyage basique des données."""
    if df is None: return None
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        if not df_clean[col].empty:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "")
    return df_clean
