"""
Feature engineering et transformation des données.
"""
import pandas as pd
import numpy as np
import logging
from .config import SEUIL_REUSSITE, TARGET_CLF

logger = logging.getLogger(__name__)

def add_advanced_features(df):
    """
    Ajoute des variables calculées et des interactions.
    """
    logger.info("Ajout des features avancées...")
    df_feat = df.copy()
    
    # 1. Score d'Équilibre Vie (Sommeil + Sport vs Devoirs + Écran)
    # On transforme Sport en numérique (1/0)
    sport_num = (df_feat['sport'] == 'Oui').astype(int)
    df_feat['score_equilibre'] = (df_feat['heures_sommeil'] + sport_num * 2) / (df_feat['heures_devoirs'] + df_feat['temps_ecrans'] + 1)
    
    # 2. Interactions logiques
    df_feat['stress_absences'] = df_feat['stress'] * df_feat['absences']
    df_feat['motivation_travail'] = df_feat['motivation'] * df_feat['heures_devoirs']
    
    # 3. Target de classification (Succès/Échec)
    if 'note_moyenne' in df_feat.columns:
        df_feat[TARGET_CLF] = (df_feat['note_moyenne'] >= SEUIL_REUSSITE).astype(int)
        
    return df_feat

def parse_heure(v):
    """Convertit '22h30' en float (22.5)"""
    try:
        if 'h' in str(v):
            h, m = str(v).split('h')
            return float(h) + float(m)/60
        return float(v)
    except:
        return 0.0

def prenttoyer_horaires(df):
    """Convertit les colonnes horaires en numérique."""
    df_h = df.copy()
    for col in ['heure_coucher', 'heure_lever']:
        if col in df_h.columns:
            df_h[f"{col}_num"] = df_h[col].apply(parse_heure)
    return df_h
