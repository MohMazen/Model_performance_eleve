"""
API REST pour EduStats — Exposition des prédictions via FastAPI.
"""
import logging
import os
import io
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_FILE, TARGET_REG, TARGET_CLF, COLS_TO_DROP, SEUIL_REUSSITE
from src.models import ModelManager
from src.features import add_advanced_features, nettoyer_horaires
from src.data_utils import nettoyer_donnees
from src.early_warning import EarlyWarningSystem
from src.behavioral import predict_behavioral, get_behavioral_status, train_behavioral_model

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EduStats API",
    description="API de prédiction des performances scolaires et d'alerte précoce.",
    version="3.1",
)

# ── État global ────────────────────────────────────────────────────
_mm: Optional[ModelManager] = None


def _get_mm() -> ModelManager:
    global _mm
    if _mm is None:
        _mm = ModelManager()
        if not _mm.load_models(MODEL_FILE):
            raise HTTPException(500, "Aucun modèle chargé. Entraînez d'abord via main.py.")
    return _mm


# ── Schémas Pydantic ───────────────────────────────────────────────
class StudentInput(BaseModel):
    heures_etude_soir: float = Field(3.0, ge=0, le=10)
    heures_sommeil: float = Field(8.0, ge=4, le=12)
    stress_personnel: int = Field(1, ge=0, le=4)
    perseverance: int = Field(3, ge=1, le=5)
    heures_jeux_video: float = Field(1.0, ge=0, le=8)
    heures_reseaux_sociaux: float = Field(1.0, ge=0, le=8)
    heures_streaming: float = Field(1.0, ge=0, le=8)
    organisation: int = Field(5, ge=1, le=10)
    confiance_soi: int = Field(7, ge=1, le=10)
    estime_soi: int = Field(7, ge=1, le=10)
    activite_sportive: str = Field("oui")
    classe: str = Field("3eme")
    qualite_sommeil: int = Field(7, ge=1, le=10)
    calme_maison: int = Field(7, ge=1, le=10)


class PredictionOutput(BaseModel):
    note_predite: float
    probabilite_reussite: float
    risk_score: float
    risk_zone: str
    top_recommendations: List[Dict[str, Any]] = []


class MetricsOutput(BaseModel):
    model_loaded: bool
    model_path: str
    available_models: List[str]


class BehavioralInput(BaseModel):
    heures_etude_soir: float = Field(2.0, ge=0, le=10)
    heures_sommeil: float = Field(8.0, ge=4, le=12)
    qualite_sommeil: int = Field(7, ge=1, le=10)
    stress_personnel: int = Field(2, ge=0, le=4)
    perseverance: int = Field(3, ge=1, le=5)
    heures_jeux_video: float = Field(1.0, ge=0, le=8)
    heures_reseaux_sociaux: float = Field(1.0, ge=0, le=8)
    heures_streaming: float = Field(1.0, ge=0, le=8)
    confiance_soi: int = Field(7, ge=1, le=10)
    estime_soi: int = Field(7, ge=1, le=10)
    activite_sportive: str = Field("oui")
    calme_maison: int = Field(7, ge=1, le=10)
    indice_motivation: float = Field(6.0, ge=1, le=10)
    classe: str = Field("3eme")


class BehavioralOutput(BaseModel):
    note_predite: float
    probabilite_reussite: float
    risk_score: float
    risk_zone: str
    conseils: List[str] = []


class BehavioralStatusOutput(BaseModel):
    is_trained: bool
    r2_score: float
    f1_score: float
    n_synthetic: int
    model_file: str


# ── Routes principales ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "EduStats API v3.1", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsOutput)
async def get_metrics():
    """Retourne les informations sur le modèle courant."""
    available = [f for f in os.listdir("outputs") if f.endswith(".joblib")] if os.path.exists("outputs") else []
    mm = _get_mm()
    return MetricsOutput(model_loaded=mm.best_model_reg is not None, model_path=MODEL_FILE, available_models=available)


@app.post("/predict", response_model=PredictionOutput)
async def predict_student(student: StudentInput):
    """Prédiction individuelle pour un élève."""
    mm = _get_mm()
    model_reg = mm.best_overall_reg or mm.best_model_reg
    model_clf = mm.best_overall_clf or mm.best_model_clf
    if model_reg is None or model_clf is None:
        raise HTTPException(500, "Modèles non disponibles.")

    data = student.model_dump()
    df_input = pd.DataFrame([data])
    df_input = add_advanced_features(df_input)

    cols_drop = [c for c in COLS_TO_DROP if c in df_input.columns]
    targets = [c for c in [TARGET_REG, TARGET_CLF] if c in df_input.columns]
    X = df_input.drop(columns=cols_drop + targets, errors="ignore")

    note_pred = float(model_reg.predict(X)[0])

    try:
        classes = list(model_clf.classes_)
        probas = model_clf.predict_proba(X)[0]
        proba_reussite = float(probas[classes.index(1)] * 100) if 1 in classes else 0.0
    except Exception:
        proba_reussite = 0.0

    ews = EarlyWarningSystem()
    proba_echec = (100 - proba_reussite) / 100
    risk = ews.compute_risk_score(df_input.iloc[0], proba_echec)

    return PredictionOutput(
        note_predite=round(note_pred, 2),
        probabilite_reussite=round(proba_reussite, 1),
        risk_score=risk["risk_score"],
        risk_zone=risk["risk_zone_label"],
    )


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Prédiction par lot via upload CSV."""
    mm = _get_mm()
    model_reg = mm.best_overall_reg or mm.best_model_reg
    if model_reg is None:
        raise HTTPException(500, "Modèles non disponibles.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.lower() for c in df.columns]
    df = nettoyer_donnees(df)
    df = nettoyer_horaires(df)
    df = add_advanced_features(df)

    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    targets = [c for c in [TARGET_REG, TARGET_CLF] if c in df.columns]
    X = df.drop(columns=cols_drop + targets, errors="ignore")

    predictions = model_reg.predict(X)
    df["note_predite"] = predictions.round(2)

    result = df[["nom", "prenom", "note_predite"]].to_dict(orient="records") if "nom" in df.columns else \
        df[["note_predite"]].to_dict(orient="records")
    return {"count": len(result), "predictions": result}


# ── Routes comportementales ──────────────────────────────────────────
@app.get("/behavioral/status", response_model=BehavioralStatusOutput)
async def behavioral_status():
    """Retourne l'état du modèle comportemental (auto-entraîne si absent)."""
    try:
        status = get_behavioral_status()
        return BehavioralStatusOutput(**status)
    except Exception as e:
        raise HTTPException(500, f"Erreur modèle comportemental : {e}")


@app.post("/behavioral/train")
async def behavioral_train():
    """Force le ré-entraînement du modèle comportemental."""
    try:
        result = train_behavioral_model()
        return {
            "message": "Modèle comportemental entraîné avec succès",
            "r2_score": result['r2_score'],
            "f1_score": result['f1_score'],
        }
    except Exception as e:
        raise HTTPException(500, f"Erreur d'entraînement : {e}")


@app.post("/behavioral/predict", response_model=BehavioralOutput)
async def behavioral_predict(student: BehavioralInput):
    """
    Prédiction comportementale pour un élève.

    Utilise un modèle dédié entraîné sur les 14 paramètres comportementaux.
    Auto-entraîne si aucun modèle sauvegardé.
    """
    try:
        result = predict_behavioral(student.model_dump())
        return BehavioralOutput(**result)
    except Exception as e:
        raise HTTPException(500, f"Erreur de prédiction comportementale : {e}")
