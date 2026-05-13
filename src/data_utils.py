"""
Utilitaires pour le chargement, le nettoyage et la génération de données.
Schéma 100 % aligné sur Questionnaire.html.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from src.config import DATA_FILE, ID_COLUMNS, GRADE_COLUMNS

logger = logging.getLogger(__name__)


def valider_schema(df: pd.DataFrame, colonnes_requises: List[str]) -> None:
    """Vérifie que les colonnes requises existent dans le DataFrame."""
    manquantes = [col for col in colonnes_requises if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {manquantes}")
    logger.info("Validation du schéma réussie.")


# ── Référentiels alignés sur Questionnaire.html ──────────────────────────────
# Toute modification de ces listes doit être répercutée dans Questionnaire.html
# et inversement, pour que les saisies réelles soient compatibles avec le modèle.
_MATIERES_TRONC = ['francais', 'maths', 'hgemc', 'anglais', 'arabe', 'sciences', 'eps']
_SPE_POSSIBLES = ['Maths', 'Physique-Chimie', 'SVT', 'SES', 'HLP', 'HGGSP', 'NSI', 'LLCE']
_FREQ_5 = ['jamais', 'rarement', 'parfois', 'souvent', 'toujours']
_FREQ_MEDIATHEQUE = ['jamais', 'rarement', 'parfois', 'souvent', 'plusieurs_fois_semaine']
_TACHES_5 = ['jamais', 'rarement', 'parfois', 'souvent', 'toujours']
_REPAS_5 = ['jamais', 'rarement', 'parfois', 'souvent', 'toujours']
_ETATS_PSY = ['serein', 'amoureux', 'anxiete', 'deprime', 'depression', 'autre']
_SOUTIEN_MATIERES_POOL = ['francais', 'maths', 'sciences', 'anglais', 'histoire_geo']


def generer_donnees_synthetiques(
    n_eleves: int = 300,
    classes_selectionnees: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Génère un jeu de données synthétiques dont le schéma (noms de colonnes,
    plages, valeurs catégorielles) correspond exactement à Questionnaire.html.

    Tout modèle entraîné sur la sortie de cette fonction peut donc être
    appliqué directement à une saisie réelle issue du questionnaire.
    """
    logger.info(f"Génération de {n_eleves} données synthétiques (schéma Questionnaire)...")
    np.random.seed(42)

    if classes_selectionnees is None:
        classes_selectionnees = ['6eme', '5eme', '4eme', '3eme', '2nde', '1ere', 'terminale']
    map_classes_reverse = {
        'Sixième': '6eme', 'Cinquième': '5eme', 'Quatrième': '4eme',
        'Troisième': '3eme', 'Seconde': '2nde', 'Première': '1ere', 'Terminale': 'terminale',
    }
    classes_values = [map_classes_reverse.get(c, c) for c in classes_selectionnees]

    data: dict = {}

    # ── SECTION 1 : État Civil ────────────────────────────────────────────
    data['Nom'] = [f"NOM_{i}" for i in range(n_eleves)]
    data['Prenom'] = [f"PRENOM_{i}" for i in range(n_eleves)]
    data['Adresse'] = [f"{np.random.randint(1, 100)} Rue de l'Ecole" for _ in range(n_eleves)]
    # Plage [10, 20] — sous-ensemble réaliste de la plage [5, 25] tolérée par le formulaire.
    data['Age'] = np.random.normal(15.5, 1.2, n_eleves).clip(10, 20).astype(int)
    data['Classe'] = np.random.choice(classes_values, n_eleves)
    data['Etablissement'] = np.random.choice(['College Pasteur', 'Lycee Curie', 'Inst. Voltaire'], n_eleves)
    # 0–300 min côté formulaire, distribution réaliste 5–120 min.
    data['Duree_trajet_AR_min'] = np.random.gamma(2, 15, n_eleves).clip(5, 120).astype(int)

    # ── SECTION 2 : Capacités Organisationnelles ──────────────────────────
    data['Organisation'] = np.random.randint(1, 11, n_eleves)
    data['Gestion_temps'] = np.random.choice(['faible', 'moyen', 'fort'], n_eleves, p=[0.2, 0.5, 0.3])

    # ── SECTION 3 : Motivation et Engagement ──────────────────────────────
    # Renommage interet_* → motivation_* (alignement Questionnaire.html).
    for m in _MATIERES_TRONC:
        data[f'Motivation_{m}'] = np.random.randint(0, 11, n_eleves)
    # Champ « enseignement scientifique » exposé uniquement en lycée par le formulaire.
    # Valeur 0 par défaut, fillée plus bas pour les classes ≥ 1ere.
    data['Motivation_enseignement_scientifique'] = [0] * n_eleves

    # Spécialités (lycée) — défauts pour collège, remplis plus bas pour 1ere/term.
    for i in range(1, 4):
        data[f'Specialite1ere_{i}_nom'] = [""] * n_eleves
        data[f'Specialite1ere_{i}_motivation'] = [0] * n_eleves
        data[f'note_specialite1ere_{i}'] = [np.nan] * n_eleves
    for i in range(1, 3):
        data[f'SpecialiteTerm_{i}_nom'] = [""] * n_eleves
        data[f'SpecialiteTerm_{i}_motivation'] = [0] * n_eleves
        data[f'note_specialiteterm_{i}'] = [np.nan] * n_eleves

    data['Motivation_famille'] = np.random.randint(0, 11, n_eleves)
    data['Motivation_recompenses'] = np.random.randint(0, 11, n_eleves)

    # Grit : 3 questions séparées (échelle 1-5) + agrégation perseverance.
    data['Grit1'] = np.random.randint(1, 6, n_eleves)
    data['Grit2'] = np.random.randint(1, 6, n_eleves)
    data['Grit3'] = np.random.randint(1, 6, n_eleves)
    grit_stack = np.column_stack([data['Grit1'], data['Grit2'], data['Grit3']])
    data['Perseverance'] = np.round(grit_stack.mean(axis=1)).astype(int)

    # ── SECTION 4 : Temps d'étude et écrans ───────────────────────────────
    # Plages alignées sur Questionnaire.html (max 8/12/12/12/8 h).
    data['Heures_etude_soir'] = np.random.gamma(4, 0.5, n_eleves).clip(0, 8)
    data['Heures_jeux_video'] = np.random.gamma(2, 0.5, n_eleves).clip(0, 12)
    data['Heures_reseaux_sociaux'] = np.random.gamma(2, 0.5, n_eleves).clip(0, 12)
    data['Heures_streaming'] = np.random.gamma(2, 0.5, n_eleves).clip(0, 12)
    data['Heures_sites_educatifs'] = np.random.gamma(1.5, 0.4, n_eleves).clip(0, 8)

    # ── SECTION 5 : Sommeil et bien-être ──────────────────────────────────
    data['Heures_sommeil'] = np.random.normal(7.5, 1, n_eleves).clip(4, 12)
    data['Qualite_sommeil'] = np.random.randint(1, 11, n_eleves)

    # Heures au format HH:MM par 30 min (aligné sur les options du formulaire).
    h_coucher = np.random.randint(19, 24, n_eleves)  # 19:00 à 23:30
    m_coucher = np.random.choice([0, 30], n_eleves)
    data['Heure_coucher'] = [f"{h:02d}:{m:02d}" for h, m in zip(h_coucher, m_coucher)]
    h_lever = np.random.randint(5, 10, n_eleves)  # 05:00 à 09:30
    m_lever = np.random.choice([0, 30], n_eleves)
    data['Heure_lever'] = [f"{h:02d}:{m:02d}" for h, m in zip(h_lever, m_lever)]

    # Stress : 2 questions séparées (échelle 0-4) + agrégation stress_personnel.
    data['Stress1'] = np.random.randint(0, 5, n_eleves)
    data['Stress2'] = np.random.randint(0, 5, n_eleves)
    stress_stack = np.column_stack([data['Stress1'], data['Stress2']])
    data['Stress_personnel'] = np.round(stress_stack.mean(axis=1)).astype(int)

    # ── SECTION 6 : Nutrition ─────────────────────────────────────────────
    data['Nb_repas'] = np.random.choice(['1', '2', '3', 'plus'], n_eleves)
    data['Repas_equilibres'] = np.random.choice(_REPAS_5, n_eleves)
    data['Activite_sportive'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Heures_activite_physique'] = np.random.gamma(2, 1, n_eleves).clip(0, 20)
    data['Niveau_sportif'] = np.random.choice(['debutant', 'intermediaire', 'confirme'], n_eleves)

    # ── SECTION 7 : Relations ─────────────────────────────────────────────
    data['Pref_travail'] = np.random.choice(['seul', 'groupe', 'mixte'], n_eleves)
    data['Soutien_mutuel'] = np.random.choice(_FREQ_5, n_eleves)
    data['Tuteur'] = np.random.choice(['oui', 'non'], n_eleves)
    # Échelle 1-10 (alignée sur qualiteTuteurScore du formulaire). 0 si pas de tuteur.
    data['Qualite_tuteur'] = np.where(
        np.array(data['Tuteur']) == 'oui',
        np.random.randint(1, 11, n_eleves),
        0,
    )
    data['Mentor'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Qualite_mentor'] = np.where(
        np.array(data['Mentor']) == 'oui',
        np.random.randint(1, 11, n_eleves),
        0,
    )

    # ── SECTION 8 : Confiance et psychologie ──────────────────────────────
    data['Confiance_soi'] = np.random.randint(1, 11, n_eleves)
    data['Estime_soi'] = np.random.randint(1, 11, n_eleves)
    data['Evitement'] = np.random.choice(_FREQ_5, n_eleves)
    data['Abandon'] = np.random.choice(_FREQ_5, n_eleves)
    # État psychologique : distribution réaliste, dominée par "serein".
    data['Etat_psychologique'] = np.random.choice(
        _ETATS_PSY, n_eleves, p=[0.60, 0.10, 0.12, 0.10, 0.05, 0.03],
    )
    # Champ libre "autre état" : rempli seulement si l'élève a coché "autre".
    data['Autre_etat_psy'] = np.where(
        np.array(data['Etat_psychologique']) == 'autre',
        'À préciser',
        '',
    )
    data['Suivi_psy'] = np.random.choice(['oui', 'non'], n_eleves, p=[0.1, 0.9])
    data['Stress_examens'] = np.random.randint(1, 11, n_eleves)
    data['Pression_familiale'] = np.random.randint(1, 11, n_eleves)

    # ── SECTION 9 : Environnement ─────────────────────────────────────────
    data['Bureau_personnel'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Calme_maison'] = np.random.randint(1, 11, n_eleves)
    data['Lumiere_adaptee'] = np.random.choice(['oui', 'non'], n_eleves)
    data['Perturbations'] = np.random.choice(_FREQ_5, n_eleves)
    data['Temps_libre'] = np.random.randint(0, 11, n_eleves)  # 0-10 h (aligné formulaire)
    data['Taches_menageres'] = np.random.choice(_TACHES_5, n_eleves)
    # Garde frères/sœurs : 0-40 h/semaine (aligné formulaire).
    data['Heures_garde_freres_soeurs'] = np.random.gamma(1.5, 4, n_eleves).clip(0, 40).astype(int)

    # Soutien scolaire : tirage aléatoire de 0 à 3 matières concernées.
    soutien_lists = []
    heures_soutien_par_matiere = {m: [] for m in _SOUTIEN_MATIERES_POOL}
    for _ in range(n_eleves):
        n_matieres = np.random.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.15, 0.05])
        if n_matieres == 0:
            soutien_lists.append('')
            for m in _SOUTIEN_MATIERES_POOL:
                heures_soutien_par_matiere[m].append(0)
        else:
            choisies = list(np.random.choice(_SOUTIEN_MATIERES_POOL, n_matieres, replace=False))
            soutien_lists.append(','.join(choisies))
            for m in _SOUTIEN_MATIERES_POOL:
                heures_soutien_par_matiere[m].append(
                    int(np.clip(np.random.gamma(1.5, 1.5), 0, 8)) if m in choisies else 0
                )
    data['Soutien_matieres'] = soutien_lists
    for m in _SOUTIEN_MATIERES_POOL:
        data[f'Heures_soutien_{m}'] = heures_soutien_par_matiere[m]

    data['Abonnement_plateforme'] = np.random.choice(['oui', 'non'], n_eleves)
    _plateformes_pool = ['Kartable', 'Schoolmouv', 'Maxicours', 'Khan Academy', '']
    data['Nom_plateforme'] = [
        np.random.choice(_plateformes_pool[:-1]) if ab == 'oui' else ''
        for ab in data['Abonnement_plateforme']
    ]
    data['Frequence_mediatheque'] = np.random.choice(_FREQ_MEDIATHEQUE, n_eleves)

    # ── CALCUL DES NOTES & SPÉCIALITÉS (par élève) ────────────────────────
    data['note_francais'] = [0.0] * n_eleves
    data['note_maths'] = [0.0] * n_eleves
    data['note_histoire_geo'] = [0.0] * n_eleves
    data['note_sciences'] = [0.0] * n_eleves

    for i in range(n_eleves):
        classe = data['Classe'][i]

        # Spécialités attribuées aléatoirement pour les lycéens.
        if classe in ['1ere', 'terminale']:
            spes_1ere = np.random.choice(_SPE_POSSIBLES, 3, replace=False)
            for j, s in enumerate(spes_1ere):
                data[f'Specialite1ere_{j + 1}_nom'][i] = s
                data[f'Specialite1ere_{j + 1}_motivation'][i] = np.random.randint(0, 11)
            # Enseignement scientifique : tronc commun lycée hors série S.
            data['Motivation_enseignement_scientifique'][i] = np.random.randint(0, 11)

            if classe == 'terminale':
                spes_term = np.random.choice(spes_1ere, 2, replace=False)
                for j, s in enumerate(spes_term):
                    data[f'SpecialiteTerm_{j + 1}_nom'][i] = s
                    data[f'SpecialiteTerm_{j + 1}_motivation'][i] = np.random.randint(0, 11)

        # Signal déterministe pour les notes.
        bonus = 0
        bonus += data['Heures_etude_soir'][i] * 1.5
        bonus += (data['Motivation_maths'][i] + data['Motivation_francais'][i]) * 0.4
        bonus -= (data['Heures_jeux_video'][i] + data['Heures_reseaux_sociaux'][i]) * 0.8
        bonus -= data['Stress_personnel'][i] * 1.0
        bonus += data['Heures_sommeil'][i] * 0.3
        bonus += data['Organisation'][i] * 0.2
        bonus += data['Perseverance'][i] * 0.5

        if data['Activite_sportive'][i] == 'oui':
            bonus += 1.0
        if data['Duree_trajet_AR_min'][i] > 60:
            bonus -= 1.0

        bonus_centre = bonus - 7.5

        data['note_francais'][i] = np.clip(np.random.normal(11, 2.0) + bonus_centre, 0, 20)
        data['note_maths'][i] = np.clip(np.random.normal(10, 2.5) + bonus_centre, 0, 20)
        data['note_histoire_geo'][i] = np.clip(np.random.normal(12, 2.0) + bonus_centre, 0, 20)
        data['note_sciences'][i] = np.clip(np.random.normal(11, 2.5) + bonus_centre, 0, 20)

        if classe in ['1ere', 'terminale']:
            for j in range(1, 4):
                data[f'note_specialite1ere_{j}'][i] = np.clip(
                    np.random.normal(12, 3.0) + bonus_centre, 0, 20
                )
            if classe == 'terminale':
                for j in range(1, 3):
                    data[f'note_specialiteterm_{j}'][i] = np.clip(
                        np.random.normal(12, 3.0) + bonus_centre, 0, 20
                    )

    # Normalisation finale : toutes les colonnes en snake_case lower.
    data = {k.lower(): v for k, v in data.items()}
    df = pd.DataFrame(data)

    # Cible régression : moyenne des notes (NaN ignorés).
    df['note_moyenne'] = df[GRADE_COLUMNS].mean(axis=1)
    return df


def charger_donnees(chemin: str) -> Optional[pd.DataFrame]:
    """Charge les données depuis CSV avec détection automatique du séparateur."""
    try:
        df = pd.read_csv(chemin, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = [str(c).lower() for c in df.columns]
        logger.info(f"Données chargées ({df.shape[0]} lignes) depuis {chemin} (séparateur détecté)")
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.error(f"Fichier introuvable ou vide {chemin}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur de chargement de {chemin}: {e}")
        return None


def nettoyer_donnees(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Nettoyage structurel des données : suppression des colonnes entièrement vides.

    L'imputation des valeurs manquantes (médiane / mode) est intentionnellement
    absente ici. Elle est prise en charge par le pipeline sklearn (SimpleImputer
    dans prepare_pipeline), ce qui garantit que les statistiques d'imputation
    sont apprises exclusivement sur le train set et jamais sur le test set
    (absence de fuite de données par contamination du prétraitement).
    """
    if df is None:
        return None
    df_clean = df.copy()

    cols_vides = [c for c in df_clean.columns if df_clean[c].isna().all()]
    if cols_vides:
        logger.info(f"Suppression des colonnes 100% vides : {cols_vides}")
        df_clean = df_clean.drop(columns=cols_vides)

    return df_clean
