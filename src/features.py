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
    # Gestion robuste de 'activite_sportive' (avec alias possible)
    col_sport = next((c for c in df_feat.columns if c in ['activite_sportive', 'activite_sport']), None)
    if col_sport:
        sport_num = (df_feat[col_sport] == 'oui').astype(int)
    else:
        sport_num = pd.Series(0, index=df_feat.index)
    
    # Temps d'écrans cumulé (vérification présence colonnes)
    ecrans_cols = ['heures_jeux_video', 'heures_reseaux_sociaux', 'heures_streaming']
    existing_ecrans = [c for c in ecrans_cols if c in df_feat.columns]
    df_feat['temps_ecrans_total'] = df_feat[existing_ecrans].sum(axis=1) if existing_ecrans else 0
    
    # Score d’équilibre (sécurisé)
    h_sommeil = df_feat['heures_sommeil'] if 'heures_sommeil' in df_feat.columns else 8.0
    h_etude = df_feat['heures_etude_soir'] if 'heures_etude_soir' in df_feat.columns else 2.0
    df_feat['score_equilibre'] = (h_sommeil + sport_num * 2) / (h_etude + df_feat['temps_ecrans_total'] + 1)

    # Ratio étude / écrans
    df_feat['ratio_etude_ecrans'] = h_etude / (df_feat['temps_ecrans_total'] + 0.5)

    # Stress personnel direct
    if 'stress_personnel' in df_feat.columns:
        df_feat['stress_total'] = pd.to_numeric(df_feat['stress_personnel'], errors='coerce').fillna(0)
    else:
        df_feat['stress_total'] = 0
    
    # Indice de motivation moyen
    interet_cols = [c for c in df_feat.columns if c.startswith('interet_')]
    if interet_cols:
        df_feat['indice_motivation'] = df_feat[interet_cols].mean(axis=1)
    else:
        df_feat['indice_motivation'] = 5.0  # Neutre

    # 3. Target de classification (Succès/Échec)
    if 'note_moyenne' in df_feat.columns:
        df_feat[TARGET_CLF] = (df_feat['note_moyenne'] >= SEUIL_REUSSITE).astype(int)
    elif all(c in df_feat.columns for c in ['note_francais', 'note_maths']):
        # Fallback si note_moyenne absente mais notes individuelles présentes
        from src.config import GRADE_COLUMNS
        existing_grades = [c for c in GRADE_COLUMNS if c in df_feat.columns]
        if existing_grades:
            df_feat['note_moyenne'] = df_feat[existing_grades].mean(axis=1)
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
