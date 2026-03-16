"""
Feature engineering et transformation des données.
"""
import pandas as pd
import numpy as np
import logging
from src.config import SEUIL_REUSSITE, TARGET_CLF

logger = logging.getLogger(__name__)


def add_advanced_features(df):
    """
    Ajoute des variables calculées et des interactions basées sur le nouveau schéma.
    """
    logger.info("Ajout des features avancées...")
    df_feat = df.copy()

    sport_num = (df_feat['activite_sportive'] == 'oui').apply(lambda x: 1 if x else 0)
    ecrans = df_feat[['heures_jeux_video', 'heures_reseaux_sociaux', 'heures_streaming']].sum(axis=1)
    
    df_feat['score_equilibre'] = (df_feat['heures_sommeil'] + sport_num * 2) / (df_feat['heures_etude_soir'] + ecrans + 1)

    # 2. Interactions logiques
    # stress_1 est souvent entre 0 et 4 dans ce genre de questionnaire (perçu)
    # On s'assure que ce sont des entiers/floats
    df_feat['stress_total'] = pd.to_numeric(df_feat['stress_1'], errors='coerce').fillna(0) + \
                              pd.to_numeric(df_feat['stress_2'], errors='coerce').fillna(0)
    
    # 3. Target de classification (Succès/Échec)
    if 'note_moyenne' in df_feat.columns:
        df_feat[TARGET_CLF] = (df_feat['note_moyenne'] >= SEUIL_REUSSITE).astype(int)

    return df_feat


def parse_heure(v):
    """Convertit '22:30' ou '22h30' en float (22.5)."""
    try:
        if v is None or pd.isna(v):
            return 0.0
        s = str(v).replace('h', ':')
        if ':' in s:
            h, m = s.split(':')
            return float(h) + float(m) / 60
        return float(s)
    except (ValueError, AttributeError, TypeError):
        return 0.0


def prenttoyer_horaires(df):
    """Convertit les colonnes horaires en numérique."""
    df_h = df.copy()
    for col in ['heure_coucher', 'heure_lever']:
        if col in df_h.columns:
            df_h[f"{col}_num"] = df_h[col].apply(parse_heure)
    return df_h
