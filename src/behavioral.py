"""
Module de prédiction comportementale des élèves.

Modèle dédié aux 14 paramètres comportementaux :
écrans, sommeil, stress, psychologie, environnement.
"""
import logging
import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional

logger = logging.getLogger(__name__)

BEHAVIORAL_MODEL_FILE = os.path.join('outputs', 'behavioral_model.joblib')

FEATURE_NAMES = [
    'heures_etude_soir', 'heures_sommeil', 'qualite_sommeil',
    'stress_personnel', 'perseverance', 'heures_jeux_video',
    'heures_reseaux_sociaux', 'heures_streaming',
    'confiance_soi', 'estime_soi', 'activite_sportive_num',
    'calme_maison', 'indice_motivation', 'classe_num',
    'temps_ecrans_total', 'score_equilibre', 'ratio_etude_ecrans',
]

_cache: Optional[dict] = None

_CLASSE_MAP = {
    '6eme': 1, '5eme': 2, '4eme': 3, '3eme': 4,
    '2nde': 5, '1ere': 6, 'terminale': 7,
}


def _encode_classe(classe: str) -> float:
    return float(_CLASSE_MAP.get(str(classe).lower(), 4))


def _build_features(profile: dict) -> pd.DataFrame:
    sport = 1.0 if str(profile.get('activite_sportive', 'non')).lower() in ['oui', '1', 'true', 'yes'] else 0.0
    classe_num = _encode_classe(str(profile.get('classe', '3eme')))
    h_etude = float(profile.get('heures_etude_soir', 2.0))
    h_sommeil = float(profile.get('heures_sommeil', 8.0))
    q_sommeil = float(profile.get('qualite_sommeil', 7.0))
    h_jeux = float(profile.get('heures_jeux_video', 1.0))
    h_rs = float(profile.get('heures_reseaux_sociaux', 1.0))
    h_stream = float(profile.get('heures_streaming', 1.0))

    ecrans = h_jeux + h_rs + h_stream
    score_eq = h_sommeil * q_sommeil / 10.0
    ratio = h_etude / (ecrans + 0.1)

    return pd.DataFrame([{
        'heures_etude_soir': h_etude,
        'heures_sommeil': h_sommeil,
        'qualite_sommeil': q_sommeil,
        'stress_personnel': float(profile.get('stress_personnel', 2)),
        'perseverance': float(profile.get('perseverance', 3)),
        'heures_jeux_video': h_jeux,
        'heures_reseaux_sociaux': h_rs,
        'heures_streaming': h_stream,
        'confiance_soi': float(profile.get('confiance_soi', 7)),
        'estime_soi': float(profile.get('estime_soi', 7)),
        'activite_sportive_num': sport,
        'calme_maison': float(profile.get('calme_maison', 7)),
        'indice_motivation': float(profile.get('indice_motivation', 6)),
        'classe_num': classe_num,
        'temps_ecrans_total': ecrans,
        'score_equilibre': score_eq,
        'ratio_etude_ecrans': ratio,
    }])


def _generate_synthetic(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        ptype = rng.choice(['bon', 'moyen', 'faible'], p=[0.35, 0.40, 0.25])
        if ptype == 'bon':
            h_etude = rng.uniform(2.5, 5)
            h_sommeil = rng.uniform(7.5, 9)
            q_sommeil = int(rng.integers(7, 11))
            stress = int(rng.integers(0, 3))
            persev = int(rng.integers(3, 6))
            h_jeux = rng.uniform(0, 1.5)
            h_rs = rng.uniform(0, 1.5)
            h_stream = rng.uniform(0, 1.5)
            confiance = int(rng.integers(7, 11))
            estime = int(rng.integers(7, 11))
            sport_num = float(rng.choice([1.0, 0.0], p=[0.75, 0.25]))
            calme = int(rng.integers(7, 11))
            motivation = rng.uniform(7, 10)
        elif ptype == 'moyen':
            h_etude = rng.uniform(1, 3)
            h_sommeil = rng.uniform(6.5, 8.5)
            q_sommeil = int(rng.integers(5, 9))
            stress = int(rng.integers(1, 4))
            persev = int(rng.integers(2, 5))
            h_jeux = rng.uniform(0.5, 3)
            h_rs = rng.uniform(0.5, 3)
            h_stream = rng.uniform(0.5, 2.5)
            confiance = int(rng.integers(5, 9))
            estime = int(rng.integers(5, 9))
            sport_num = float(rng.choice([1.0, 0.0], p=[0.55, 0.45]))
            calme = int(rng.integers(5, 9))
            motivation = rng.uniform(4, 8)
        else:
            h_etude = rng.uniform(0, 1.5)
            h_sommeil = rng.uniform(5, 7.5)
            q_sommeil = int(rng.integers(1, 6))
            stress = int(rng.integers(2, 5))
            persev = int(rng.integers(1, 4))
            h_jeux = rng.uniform(2, 6)
            h_rs = rng.uniform(2, 5)
            h_stream = rng.uniform(1.5, 5)
            confiance = int(rng.integers(1, 6))
            estime = int(rng.integers(1, 6))
            sport_num = float(rng.choice([1.0, 0.0], p=[0.30, 0.70]))
            calme = int(rng.integers(1, 6))
            motivation = rng.uniform(1, 5)

        classe_num = float(rng.integers(1, 8))
        ecrans = h_jeux + h_rs + h_stream
        score_eq = h_sommeil * q_sommeil / 10.0
        ratio = h_etude / (ecrans + 0.1)

        note = (
            h_etude * 1.2 + h_sommeil * 0.4 + q_sommeil * 0.2
            - ecrans * 0.4 - stress * 0.5 + persev * 0.6
            + confiance * 0.2 + estime * 0.1 + sport_num * 0.5
            + calme * 0.15 + motivation * 0.3
            + score_eq * 0.3 + ratio * 0.4
            + rng.normal(0, 1.2)
        )
        note = float(np.clip(note, 0, 20))

        rows.append({
            'heures_etude_soir': h_etude, 'heures_sommeil': h_sommeil,
            'qualite_sommeil': q_sommeil, 'stress_personnel': stress,
            'perseverance': persev, 'heures_jeux_video': h_jeux,
            'heures_reseaux_sociaux': h_rs, 'heures_streaming': h_stream,
            'confiance_soi': confiance, 'estime_soi': estime,
            'activite_sportive_num': sport_num, 'calme_maison': calme,
            'indice_motivation': motivation, 'classe_num': classe_num,
            'temps_ecrans_total': ecrans, 'score_equilibre': score_eq,
            'ratio_etude_ecrans': ratio,
            'note_moyenne': note, 'reussite': 1 if note >= 10 else 0,
        })
    return pd.DataFrame(rows)


def train_behavioral_model() -> dict:
    """Entraîne XGBoost + RandomForest sur données comportementales synthétiques."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, f1_score
    from sklearn.model_selection import train_test_split
    try:
        from xgboost import XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    logger.info("Entraînement du modèle comportemental (600 échantillons synthétiques)...")
    df = _generate_synthetic(600)
    X = df[FEATURE_NAMES]
    y_reg = df['note_moyenne']
    y_clf = df['reussite']

    X_train, X_test, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    reg_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
    ])
    clf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced')),
    ])

    reg_pipe.fit(X_train, yr_tr)
    clf_pipe.fit(X_train, yc_tr)

    r2 = float(r2_score(yr_te, reg_pipe.predict(X_test)))
    f1 = float(f1_score(yc_te, clf_pipe.predict(X_test)))
    importances = dict(zip(FEATURE_NAMES, clf_pipe.named_steps['model'].feature_importances_))

    payload = {
        'reg': reg_pipe, 'clf': clf_pipe,
        'importances': importances,
        'r2_score': round(r2, 4), 'f1_score': round(f1, 4),
        'n_synthetic': 600,
    }
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(payload, BEHAVIORAL_MODEL_FILE)
    logger.info(f"Modèle comportemental sauvegardé (R²={r2:.4f}, F1={f1:.4f})")
    return payload


def get_behavioral_model() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(BEHAVIORAL_MODEL_FILE):
        try:
            _cache = joblib.load(BEHAVIORAL_MODEL_FILE)
            return _cache
        except Exception:
            pass
    _cache = train_behavioral_model()
    return _cache


# ── Conseils personnalisés ────────────────────────────────────────────────
_CONSEIL_MAP = {
    'heures_etude_soir': "Augmentez le temps d'étude le soir (visez 2h+)",
    'ratio_etude_ecrans': "Réduisez les écrans pour libérer du temps d'étude",
    'score_equilibre': "Améliorez l'équilibre sommeil / activité / écrans",
    'temps_ecrans_total': "Limitez les écrans à moins de 2h par jour",
    'heures_sommeil': "Dormez 8h minimum pour consolider les apprentissages",
    'qualite_sommeil': "Améliorez la qualité du sommeil (routine régulière)",
    'perseverance': "Divisez les tâches en petits objectifs pour renforcer la persévérance",
    'stress_personnel': "Pratiquez la gestion du stress (respiration, sport)",
    'indice_motivation': "Identifiez vos centres d'intérêt pour renforcer la motivation",
    'confiance_soi': "Célébrez chaque petite victoire pour renforcer la confiance",
    'activite_sportive_num': "Pratiquez une activité sportive régulière (2–3×/semaine)",
    'calme_maison': "Aménagez un espace de travail calme et sans distractions",
    'heures_jeux_video': "Réduisez le temps de jeux vidéo en semaine",
    'heures_reseaux_sociaux': "Limitez les réseaux sociaux (notifications désactivées)",
}

_IDEAL = {
    'heures_etude_soir': 3.0, 'heures_sommeil': 8.5, 'qualite_sommeil': 8.0,
    'stress_personnel': 1.0, 'perseverance': 4.5, 'heures_jeux_video': 0.5,
    'heures_reseaux_sociaux': 0.5, 'heures_streaming': 0.5,
    'confiance_soi': 8.0, 'estime_soi': 8.0, 'activite_sportive_num': 1.0,
    'calme_maison': 8.0, 'indice_motivation': 8.0,
    'temps_ecrans_total': 1.5, 'score_equilibre': 8.0, 'ratio_etude_ecrans': 2.0,
}

_HIGHER_IS_BETTER = {
    'heures_etude_soir', 'heures_sommeil', 'qualite_sommeil', 'perseverance',
    'confiance_soi', 'estime_soi', 'activite_sportive_num', 'calme_maison',
    'indice_motivation', 'score_equilibre', 'ratio_etude_ecrans',
}


def _generate_conseils(features_row: pd.Series, importances: dict, max_n: int = 5) -> list:
    gaps = {}
    for feat, text in _CONSEIL_MAP.items():
        if feat not in features_row.index:
            continue
        val = float(features_row[feat])
        ideal = _IDEAL.get(feat, 5.0)
        imp = importances.get(feat, 0.0)
        gap = max(0.0, ideal - val) if feat in _HIGHER_IS_BETTER else max(0.0, val - ideal)
        if gap > 0.05:
            gaps[feat] = gap * imp
    top = sorted(gaps, key=lambda k: gaps[k], reverse=True)[:max_n]
    return [_CONSEIL_MAP[f] for f in top]


def predict_behavioral(profile: dict) -> dict:
    """Retourne note_predite, probabilite_reussite, risk_score, risk_zone, conseils."""
    model = get_behavioral_model()
    X = _build_features(profile)

    note = float(np.clip(model['reg'].predict(X)[0], 0, 20))

    try:
        classes = list(model['clf'].classes_)
        probas = model['clf'].predict_proba(X)[0]
        proba = float(probas[classes.index(1)] * 100) if 1 in classes else 50.0
    except Exception:
        proba = 50.0

    proba = round(proba, 1)
    risk_zone = 'HIGH' if proba < 40 else ('MEDIUM' if proba < 70 else 'LOW')
    conseils = _generate_conseils(X.iloc[0], model['importances'])

    return {
        'note_predite': round(note, 2),
        'probabilite_reussite': proba,
        'risk_score': round(100 - proba, 1),
        'risk_zone': risk_zone,
        'conseils': conseils,
    }


def get_behavioral_status() -> dict:
    model = get_behavioral_model()
    return {
        'is_trained': True,
        'r2_score': model.get('r2_score', 0.0),
        'f1_score': model.get('f1_score', 0.0),
        'n_synthetic': model.get('n_synthetic', 600),
        'model_file': BEHAVIORAL_MODEL_FILE,
    }
