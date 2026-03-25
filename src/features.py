import pandas as pd
import numpy as np
import logging
import re
from src.config import SEUIL_REUSSITE, TARGET_CLF

logger = logging.getLogger(__name__)

# Mots-clés pour la détection automatique des colonnes
KEYWORDS = {
    'sport': ['sport', 'activite_sportive', 'activite_sport', 'physique'],
    'jeux_video': ['jeux_video', 'gaming', 'video_games', 'manette'],
    'reseaux': ['reseaux', 'social', 'instagram', 'tiktok', 'facebook', 'snapchat'],
    'streaming': ['streaming', 'netflix', 'youtube', 'disney', 'video'],
    'sommeil': ['sommeil', 'dodo', 'sleep', 'repos'],
    'etude': ['etude', 'devoirs', 'travail', 'study', 'soir'],
    'stress': ['stress', 'anxiete', 'pression', 'personnel'],
    'motivation': ['motivation', 'interet', 'envie', 'engagement'],
    'note_moyenne': ['moyenne', 'note_moyenne', 'grade_avg', 'resultat'],
    'heure_coucher': ['coucher', 'bedtime', 'sleep_at'],
    'heure_lever': ['lever', 'wakeup', 'wake_up']
}

def get_column_mapping(df_columns):
    """
    Tente de mapper les colonnes du DataFrame aux concepts du modèle.
    Retourne un dictionnaire {concept: colonne_reelle}.
    """
    mapping = {}
    cols_lower = [str(c).lower() for c in df_columns]
    
    for concept, words in KEYWORDS.items():
        found = False
        for word in words:
            for i, col in enumerate(cols_lower):
                if word in col:
                    mapping[concept] = df_columns[i]
                    found = True
                    break
            if found: break
    return mapping

def add_advanced_features(df, mapping=None):
    """
    Ajoute des variables calculées et des interactions de manière adaptative.
    mapping: dict optionnel {concept: nom_colonne}
    """
    logger.info("Ajout des features avancées (mode adaptatif)...")
    df_feat = df.copy()
    
    if mapping is None:
        mapping = get_column_mapping(df_feat.columns)
    
    # 1. Scores de base
    # Sport
    col_sport = mapping.get('sport')
    if col_sport and col_sport in df_feat.columns:
        # Conversion intelligente : 'oui'/1/True -> 1, sinon 0
        sport_num = df_feat[col_sport].apply(lambda x: 1 if str(x).lower() in ['oui', 'yes', '1', 'true'] else 0)
    else:
        sport_num = pd.Series(0, index=df_feat.index)
    
    # Temps d'écrans cumulé
    ecran_concepts = ['jeux_video', 'reseaux', 'streaming']
    existing_ecrans = [mapping[c] for c in ecran_concepts if c in mapping and mapping[c] in df_feat.columns]
    
    if existing_ecrans:
        df_feat['temps_ecrans_total'] = df_feat[existing_ecrans].sum(axis=1)
    else:
        df_feat['temps_ecrans_total'] = 0
    
    # 2. Indicateurs avancés (avec valeurs par défaut sécurisées)
    h_sommeil = df_feat[mapping['sommeil']] if 'sommeil' in mapping and mapping['sommeil'] in df_feat.columns else 8.0
    h_etude = df_feat[mapping['etude']] if 'etude' in mapping and mapping['etude'] in df_feat.columns else 2.0
    
    # Score d’équilibre
    df_feat['score_equilibre'] = (h_sommeil + sport_num * 2) / (df_feat['temps_ecrans_total'] + h_etude + 1)

    # Ratio étude / écrans
    df_feat['ratio_etude_ecrans'] = h_etude / (df_feat['temps_ecrans_total'] + 0.5)

    # Stress total
    col_stress = mapping.get('stress')
    if col_stress and col_stress in df_feat.columns:
        df_feat['stress_total'] = pd.to_numeric(df_feat[col_stress], errors='coerce').fillna(0)
    else:
        df_feat['stress_total'] = 0
    
    # Indice de motivation (moyenne des colonnes d'intérêt si présentes)
    interet_cols = [c for c in df_feat.columns if 'interet' in c.lower() or 'motivation' in c.lower()]
    if interet_cols:
        df_feat['indice_motivation'] = df_feat[interet_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(5.0)
    else:
        df_feat['indice_motivation'] = 5.0

    # 3. Target de classification (Succès/Échec)
    col_note = mapping.get('note_moyenne')
    if col_note and col_note in df_feat.columns:
        df_feat[TARGET_CLF] = (df_feat[col_note] >= SEUIL_REUSSITE).astype(int)
    
    return df_feat


def parse_heure(v):
    """Convertit '22:30' ou '22h30' en float (22.5)."""
    try:
        if v is None or pd.isna(v):
            return 0.0
        s = str(v).lower().replace('h', ':')
        if ':' in s:
            parts = s.split(':')
            h = parts[0]
            m = parts[1] if len(parts) > 1 else "0"
            return float(h) + float(m or 0) / 60
        return float(s)
    except (ValueError, AttributeError, TypeError):
        return 0.0


def prenttoyer_horaires(df, mapping=None):
    """Convertit les colonnes horaires en numérique."""
    df_h = df.copy()
    if mapping is None:
        mapping = get_column_mapping(df_h.columns)
        
    for concept in ['heure_coucher', 'heure_lever']:
        col = mapping.get(concept)
        if col and col in df_h.columns:
            df_h[f"{col}_num"] = df_h[col].apply(parse_heure)
    return df_h
