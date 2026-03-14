"""
Tests unitaires pour le module src.models (y compris les réseaux de neurones).
"""
import sys
import os
import tempfile

# Permettre l'import du package src depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.data_utils import generer_donnees_synthetiques, nettoyer_donnees
from src.features import add_advanced_features, prenttoyer_horaires
from src.config import COLS_TO_DROP, TARGET_REG, TARGET_CLF
from src.models import ModelManager


@pytest.fixture(scope="module")
def prepared_data():
    """Prépare les données une seule fois pour tous les tests du module."""
    df = generer_donnees_synthetiques(n_eleves=80)
    df = nettoyer_donnees(df)
    df = prenttoyer_horaires(df)
    df = add_advanced_features(df)

    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=cols_drop + [TARGET_REG, TARGET_CLF])
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]
    return X, y_reg, y_clf


@pytest.fixture(scope="module")
def trained_manager(prepared_data):
    """Entraîne tous les modèles une seule fois."""
    X, y_reg, y_clf = prepared_data
    mm = ModelManager()
    mm.prepare_pipeline(X)
    mm.train_regression(X, y_reg)
    mm.train_classification(X, y_clf)
    mm.train_nn_regression(X, y_reg)
    mm.train_nn_classification(X, y_clf)
    return mm


class TestModelManagerPipeline:
    def test_prepare_pipeline(self, prepared_data):
        X, _, _ = prepared_data
        mm = ModelManager()
        preprocessor = mm.prepare_pipeline(X)
        assert preprocessor is not None
        assert mm.preprocessor is not None


class TestNNRegression:
    def test_train_nn_regression_returns_model(self, trained_manager):
        assert trained_manager.best_model_nn_reg is not None

    def test_nn_regression_predict(self, trained_manager, prepared_data):
        X, _, _ = prepared_data
        predictions = trained_manager.best_model_nn_reg.predict(X)
        assert len(predictions) == len(X)

    def test_nn_regression_predictions_are_numeric(self, trained_manager, prepared_data):
        X, _, _ = prepared_data
        predictions = trained_manager.best_model_nn_reg.predict(X)
        assert np.isfinite(predictions).all()


class TestNNClassification:
    def test_train_nn_classification_returns_model(self, trained_manager):
        assert trained_manager.best_model_nn_clf is not None

    def test_nn_classification_predict(self, trained_manager, prepared_data):
        X, _, _ = prepared_data
        predictions = trained_manager.best_model_nn_clf.predict(X)
        assert len(predictions) == len(X)

    def test_nn_classification_predictions_are_binary(self, trained_manager, prepared_data):
        X, _, _ = prepared_data
        predictions = trained_manager.best_model_nn_clf.predict(X)
        assert set(predictions).issubset({0, 1})

    def test_nn_classification_predict_proba(self, trained_manager, prepared_data):
        X, _, _ = prepared_data
        probas = trained_manager.best_model_nn_clf.predict_proba(X)
        assert probas.shape[0] == len(X)
        assert probas.shape[1] == 2


class TestSaveLoadModels:
    def test_save_and_load_models(self, trained_manager):
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            trained_manager.save_models(path=path)

            mm2 = ModelManager()
            result = mm2.load_models(path=path)

            assert result is True
            assert mm2.best_model_reg is not None
            assert mm2.best_model_clf is not None
            assert mm2.best_model_nn_reg is not None
            assert mm2.best_model_nn_clf is not None
        finally:
            os.unlink(path)

    def test_load_models_missing_file(self):
        mm = ModelManager()
        result = mm.load_models(path='/tmp/nonexistent_model.joblib')
        assert result is False
