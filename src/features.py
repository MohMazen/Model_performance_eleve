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
    Inclut des indicateurs plus précis pour améliorer le R².
    """
    logger.info("Ajout des features avancées...")
    df_feat = df.copy()

    # 1. Scores de base
    sport_num = (df_feat['activite_sportive'] == 'oui').apply(lambda x: 1 if x else 0)
    
    # Temps d'écrans cumulé
    ecrans_cols = ['heures_jeux_video', 'heures_reseaux_sociaux', 'heures_streaming']
    df_feat['temps_ecrans_total'] = df_feat[ecrans_cols].sum(axis=1)
    
    # Score d'équilibre (mis à jour)
    df_feat['score_equilibre'] = (df_feat['heures_sommeil'] + sport_num * 2) / (df_feat['heures_etude_soir'] + df_feat['temps_ecrans_total'] + 1)

    # Ratio étude / écrans (indicateur de focus)
    df_feat['ratio_etude_ecrans'] = df_feat['heures_etude_soir'] / (df_feat['temps_ecrans_total'] + 0.5)

    # Stress personnel direct
    df_feat['stress_total'] = pd.to_numeric(df_feat['stress_personnel'], errors='coerce').fillna(0)
    
    # Indice de motivation moyen (moyenne des intérêts pour les matières)
    interet_cols = [c for c in df_feat.columns if c.startswith('interet_')]
    if interet_cols:
        df_feat['indice_motivation'] = df_feat[interet_cols].mean(axis=1)
    else:
        df_feat['indice_motivation'] = 5.0

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
