"""
Entraînement des modèles, tuning et persistance.
"""
import numpy as np
import logging
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from src.config import MODEL_FILE

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.preprocessor = None
        self.best_model_reg = None
        self.best_model_clf = None

    def prepare_pipeline(self, X):
        """Définit le preprocesseur automatique."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ])
        return self.preprocessor

    def train_regression(self, X, y):
        """Entraîne et tune un modèle de régression."""
        logger.info("Entraînement du modèle de régression (XGBoost)...")
        pipeline = Pipeline(steps=[('pre', self.preprocessor), ('model', XGBRegressor(random_state=42))])

        param_dist = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=5, cv=3, random_state=42)
        search.fit(X, y)
        self.best_model_reg = search.best_estimator_
        logger.info(f"Meilleur score R2 régression : {search.best_score_:.4f}")
        return self.best_model_reg

    def train_classification(self, X, y):
        """Entraîne et tune un modèle de classification.

        Utilise class_weight='balanced' pour gérer le déséquilibre des classes.
        """
        logger.info("Entraînement du modèle de classification (Random Forest)...")
        pipeline = Pipeline(steps=[
            ('pre', self.preprocessor),
            ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        param_dist = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5]
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=5, cv=3, random_state=42)
        search.fit(X, y)
        self.best_model_clf = search.best_estimator_
        logger.info(f"Meilleure accuracy classification : {search.best_score_:.4f}")
        return self.best_model_clf

    def save_models(self, path=MODEL_FILE):
        """Sauvegarde les modèles sur disque."""
        joblib.dump({
            'reg': self.best_model_reg,
            'clf': self.best_model_clf
        }, path)
        logger.info(f"Modèles sauvegardés dans {path}")

    def load_models(self, path=MODEL_FILE):
        """Charge les modèles depuis le disque."""
        try:
            dict_models = joblib.load(path)
            self.best_model_reg = dict_models['reg']
            self.best_model_clf = dict_models['clf']
            logger.info("Modèles chargés avec succès.")
            return True
        except (FileNotFoundError, KeyError, Exception) as e:
            logger.warning(f"Impossible de charger les modèles depuis {path} : {e}")
            return False
