"""
Entraînement des modèles, tuning et persistance.
"""
import numpy as np
import logging
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor
from src.config import MODEL_FILE

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.preprocessor = None
        self.best_model_reg = None
        self.best_model_clf = None
        self.best_model_nn_reg = None
        self.best_model_nn_clf = None
        self.best_model_svm_reg = None
        self.best_model_svm_clf = None
        self.best_overall_reg = None
        self.best_overall_clf = None
        self.subject_models = {}

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

    def train_regression(self, X, y, subject_name=None):
        """Entraîne et tune un modèle de régression (XGBoost)."""
        if subject_name:
            logger.info(f"Entraînement du modèle de régression pour {subject_name}...")
        else:
            logger.info("Entraînement du modèle de régression (XGBoost)...")

        selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), threshold="0.5*mean")
        
        pipeline = Pipeline(steps=[
            ('pre', self.preprocessor), 
            ('select', selector),
            ('model', XGBRegressor(random_state=42))
        ])

        param_dist = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, random_state=42)
        search.fit(X, y)
        
        if subject_name:
            self.subject_models[subject_name] = search.best_estimator_
            return self.subject_models[subject_name]
        else:
            self.best_model_reg = search.best_estimator_
            logger.info(f"Meilleur score R2 régression : {search.best_score_:.4f}")
            return self.best_model_reg

    def train_classification(self, X, y):
        """Entraîne et tune un modèle de classification.

        Utilise class_weight='balanced' pour gérer le déséquilibre des classes.
        """
        logger.info("Entraînement du modèle de classification (Random Forest)...")
        selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="0.5*mean")

        pipeline = Pipeline(steps=[
            ('pre', self.preprocessor),
            ('select', selector),
            ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        param_dist = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5]
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, random_state=42)
        search.fit(X, y)
        self.best_model_clf = search.best_estimator_
        logger.info(f"Meilleure accuracy classification : {search.best_score_:.4f}")
        return self.best_model_clf

    def train_nn_regression(self, X, y):
        """Entraîne et tune un réseau de neurones pour la régression."""
        logger.info("Entraînement du modèle de régression (Réseau de Neurones)...")
        selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), threshold="0.5*mean")

        pipeline = Pipeline(steps=[
            ('pre', self.preprocessor),
            ('select', selector),
            ('model', MLPRegressor(random_state=42, max_iter=1000))
        ])

        param_dist = {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (100,)],
            'model__activation': ['relu', 'tanh'],
            'model__learning_rate_init': [0.001, 0.01],
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, random_state=42)
        search.fit(X, y)
        self.best_model_nn_reg = search.best_estimator_
        logger.info(f"Meilleur score R2 régression NN : {search.best_score_:.4f}")
        return self.best_model_nn_reg

    def train_nn_classification(self, X, y):
        """Entraîne et tune un réseau de neurones pour la classification."""
        logger.info("Entraînement du modèle de classification (Réseau de Neurones)...")
        selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="0.5*mean")

        pipeline = Pipeline(steps=[
            ('pre', self.preprocessor),
            ('select', selector),
            ('model', MLPClassifier(random_state=42, max_iter=1000))
        ])

        param_dist = {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (100,)],
            'model__activation': ['relu', 'tanh'],
            'model__learning_rate_init': [0.001, 0.01],
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, random_state=42)
        search.fit(X, y)
        self.best_model_nn_clf = search.best_estimator_
        logger.info(f"Meilleure accuracy classification NN : {search.best_score_:.4f}")
        return self.best_model_nn_clf

    def train_svm_regression(self, X, y):
        """Entraîne un modèle SVR (Support Vector Regression)."""
        logger.info("Entraînement du modèle de régression (SVM)...")
        selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), threshold="0.5*mean")

        pipeline = Pipeline(steps=[
            ('pre', self.preprocessor),
            ('select', selector),
            ('model', SVR())
        ])

        param_dist = {
            'model__C': [0.1, 1, 10],
            'model__epsilon': [0.01, 0.1, 0.5],
            'model__kernel': ['rbf', 'poly']
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, random_state=42)
        search.fit(X, y)
        self.best_model_svm_reg = search.best_estimator_
        logger.info(f"Meilleur score R2 régression SVM : {search.best_score_:.4f}")
        return self.best_model_svm_reg

    def train_svm_classification(self, X, y):
        """Entraîne un modèle SVC (Support Vector Classification)."""
        logger.info("Entraînement du modèle de classification (SVM)...")
        selector = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="0.5*mean")

        pipeline = Pipeline(steps=[
            ('pre', self.preprocessor),
            ('select', selector),
            ('model', SVC(probability=True, random_state=42))
        ])

        param_dist = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear']
        }

        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, random_state=42)
        search.fit(X, y)
        self.best_model_svm_clf = search.best_estimator_
        logger.info(f"Meilleure accuracy classification SVM : {search.best_score_:.4f}")
        return self.best_model_svm_clf

    def save_models(self, path=MODEL_FILE):
        """Sauvegarde les modèles sur disque."""
        joblib.dump({
            'reg': self.best_model_reg,
            'clf': self.best_model_clf,
            'nn_reg': self.best_model_nn_reg,
            'nn_clf': self.best_model_nn_clf,
            'svm_reg': self.best_model_svm_reg,
            'svm_clf': self.best_model_svm_clf,
            'best_reg': self.best_overall_reg,
            'best_clf': self.best_overall_clf,
            'subject_models': self.subject_models
        }, path)
        logger.info(f"Modèles sauvegardés dans {path}")

    def load_models(self, path=MODEL_FILE):
        """Charge les modèles depuis le disque."""
        try:
            dict_models = joblib.load(path)
            self.best_model_reg = dict_models['reg']
            self.best_model_clf = dict_models['clf']
            self.best_model_nn_reg = dict_models.get('nn_reg')
            self.best_model_nn_clf = dict_models.get('nn_clf')
            self.best_model_svm_reg = dict_models.get('svm_reg')
            self.best_model_svm_clf = dict_models.get('svm_clf')
            self.best_overall_reg = dict_models.get('best_reg')
            self.best_overall_clf = dict_models.get('best_clf')
            self.subject_models = dict_models.get('subject_models', {})
            logger.info("Modèles chargés avec succès.")
            return True
        except (FileNotFoundError, KeyError, Exception) as e:
            logger.warning(f"Impossible de charger les modèles depuis {path} : {e}")
            return False
