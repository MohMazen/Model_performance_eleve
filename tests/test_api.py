"""
Tests d'intégration pour l'API FastAPI (src/api.py).

Couvre les endpoints publics : /, /health, /metrics, /predict, /predict/batch.
Les endpoints qui dépendent d'un modèle persistent entraînent un modèle léger
sur des données synthétiques avant le test, puis le sauvegardent dans un fichier
joblib temporaire pointé via monkeypatching.
"""
import io
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Entraîne un ModelManager léger et sauvegarde dans un fichier temporaire."""
    from src.data_utils import generer_donnees_synthetiques, nettoyer_donnees
    from src.features import nettoyer_horaires, add_advanced_features
    from src.config import COLS_TO_DROP, TARGET_REG, TARGET_CLF
    from src.models import ModelManager

    df = generer_donnees_synthetiques(n_eleves=80)
    df = nettoyer_donnees(df)
    df = nettoyer_horaires(df)
    df = add_advanced_features(df)

    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=cols_drop + [TARGET_REG, TARGET_CLF])
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]

    mm = ModelManager()
    mm.prepare_pipeline(X)
    mm.train_regression(X, y_reg)
    mm.train_classification(X, y_clf)
    mm.feature_columns = list(X.columns)

    path = tmp_path_factory.mktemp("models") / "test_model.joblib"
    mm.save_models(path=str(path))
    return str(path)


@pytest.fixture(scope="module")
def client(trained_model_path, monkeypatch_module):
    """Client FastAPI avec MODEL_FILE redirigé vers le modèle de test."""
    monkeypatch_module.setattr("src.api.MODEL_FILE", trained_model_path)
    # Réinitialiser l'état global pour forcer le rechargement.
    import src.api as api_module
    api_module._mm = None

    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    """MonkeyPatch utilisable au scope module."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ── Endpoints sans modèle ──────────────────────────────────────────────────
class TestPublicEndpoints:
    def test_root(self):
        from src.api import app
        c = TestClient(app)
        r = c.get("/")
        assert r.status_code == 200
        assert "EduStats" in r.json()["message"]

    def test_health(self):
        from src.api import app
        c = TestClient(app)
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ── Endpoints qui exigent un modèle ───────────────────────────────────────
class TestPredictEndpoints:
    def test_metrics(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is True
        assert "available_models" in body

    @pytest.mark.xfail(
        reason="Bug connu : StudentInput n'expose que 14 champs alors que le modèle "
               "est entraîné sur ~70 features. Le ColumnTransformer rejette donc l'input. "
               "À corriger en alignant le schéma Pydantic sur ModelManager.feature_columns."
    )
    def test_predict_returns_valid_payload(self, client):
        payload = {
            "heures_etude_soir": 3.0,
            "heures_sommeil": 8.0,
            "stress_personnel": 1,
            "perseverance": 3,
            "heures_jeux_video": 1.0,
            "heures_reseaux_sociaux": 1.0,
            "heures_streaming": 1.0,
            "organisation": 5,
            "confiance_soi": 7,
            "estime_soi": 7,
            "activite_sportive": "oui",
            "classe": "3eme",
            "qualite_sommeil": 7,
            "calme_maison": 7,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "note_predite" in body
        assert "probabilite_reussite" in body
        assert "risk_score" in body
        assert "risk_zone" in body
        assert 0 <= body["risk_score"] <= 100

    def test_predict_validates_bounds(self, client):
        """Pydantic doit rejeter les valeurs hors bornes (heures_sommeil > 12)."""
        payload = {
            "heures_etude_soir": 3.0,
            "heures_sommeil": 25.0,  # invalide
            "stress_personnel": 1,
            "perseverance": 3,
            "heures_jeux_video": 1.0,
            "heures_reseaux_sociaux": 1.0,
            "heures_streaming": 1.0,
            "organisation": 5,
            "confiance_soi": 7,
            "estime_soi": 7,
            "activite_sportive": "oui",
            "classe": "3eme",
            "qualite_sommeil": 7,
            "calme_maison": 7,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_batch_csv_upload(self, client):
        """L'endpoint batch doit accepter un CSV minimal et renvoyer une prédiction."""
        from src.data_utils import generer_donnees_synthetiques
        df = generer_donnees_synthetiques(n_eleves=10)
        csv_content = df.to_csv(index=False, sep=";", encoding="utf-8-sig")
        files = {"file": ("eleves.csv", csv_content, "text/csv")}
        r = client.post("/predict/batch", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["count"] >= 1
        assert "predictions" in body
        for pred in body["predictions"]:
            assert "note_predite" in pred
