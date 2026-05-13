"""
Entraînement des modèles, tuning et persistance.
"""
import numpy as np
import pandas as pd
import logging
import joblib
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor
from src.config import MODEL_FILE

logger = logging.getLogger(__name__)


# ── Catalogue des modèles ───────────────────────────────────────────────────
# Chaque entrée décrit comment construire le modèle final, ses hyperparamètres
# à explorer, et la cible (régression ou classification). Cette structure
# centralisée remplace les 6 méthodes `train_*` quasi-identiques précédentes.
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    'reg': {
        'task': 'regression',
        'attr': 'best_model_reg',
        'label': 'XGBoost (régression)',
        'estimator': lambda: XGBRegressor(random_state=42),
        'param_dist': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2],
        },
    },
    'clf': {
        'task': 'classification',
        'attr': 'best_model_clf',
        'label': 'Random Forest (classification)',
        'estimator': lambda: RandomForestClassifier(random_state=42, class_weight='balanced'),
        'param_dist': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5],
        },
    },
    'nn_reg': {
        'task': 'regression',
        'attr': 'best_model_nn_reg',
        'label': 'MLP (régression)',
        'estimator': lambda: MLPRegressor(random_state=42, max_iter=1000),
        'param_dist': {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (100,)],
            'model__activation': ['relu', 'tanh'],
            'model__learning_rate_init': [0.001, 0.01],
        },
    },
    'nn_clf': {
        'task': 'classification',
        'attr': 'best_model_nn_clf',
        'label': 'MLP (classification)',
        'estimator': lambda: MLPClassifier(random_state=42, max_iter=1000),
        'param_dist': {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (100,)],
            'model__activation': ['relu', 'tanh'],
            'model__learning_rate_init': [0.001, 0.01],
        },
    },
    'svm_reg': {
        'task': 'regression',
        'attr': 'best_model_svm_reg',
        'label': 'SVR (régression)',
        'estimator': lambda: SVR(),
        'param_dist': {
            'model__C': [0.1, 1, 10],
            'model__epsilon': [0.01, 0.1, 0.5],
            'model__kernel': ['rbf', 'poly'],
        },
    },
    'svm_clf': {
        'task': 'classification',
        'attr': 'best_model_svm_clf',
        'label': 'SVC (classification)',
        'estimator': lambda: SVC(probability=True, random_state=42),
        'param_dist': {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear'],
        },
    },
}


class ModelManager:
    def __init__(self) -> None:
        self.preprocessor: Optional[ColumnTransformer] = None
        self.best_model_reg = None
        self.best_model_clf = None
        self.best_model_nn_reg = None
        self.best_model_nn_clf = None
        self.best_model_svm_reg = None
        self.best_model_svm_clf = None
        self.best_overall_reg = None
        self.best_overall_clf = None
        self.subject_models: Dict[str, Any] = {}
        self.feature_columns: Optional[list] = None  # Persisté pour la reconstruction après reload
        self.cv_scores: Dict[str, float] = {}  # Score CV de chaque modèle entraîné

    def prepare_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Définit le preprocesseur automatique."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        # L'imputation doit être DANS le pipeline pour être apprise sur le seul
        # train set et appliquée ensuite au test set avec les mêmes statistiques.
        # Ne jamais imputer avant train_test_split (fuite de données).
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ])
        return self.preprocessor

    def _build_pipeline(self, task: str, estimator: Any) -> Pipeline:
        """Construit le pipeline complet : preprocessing → sélection → modèle."""
        if task == 'classification':
            selector_base = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            selector_base = RandomForestRegressor(n_estimators=50, random_state=42)

        return Pipeline(steps=[
            ('pre', self.preprocessor),
            ('select', SelectFromModel(selector_base, threshold='0.5*mean')),
            ('model', estimator),
        ])

    def _train(self, key: str, X: pd.DataFrame, y: pd.Series,
               n_iter: int = 10, n_splits: int = 5) -> Any:
        """
        Entraîne un modèle du catalogue par recherche d'hyperparamètres.
        Utilise StratifiedKFold pour la classification (évite les plis sans échec).
        """
        spec = MODEL_CATALOG[key]
        logger.info(f"Entraînement : {spec['label']}")

        pipeline = self._build_pipeline(spec['task'], spec['estimator']())

        if spec['task'] == 'classification':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scoring = 'f1_weighted'
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scoring = 'r2'

        search = RandomizedSearchCV(
            pipeline, spec['param_dist'],
            n_iter=n_iter, cv=cv, scoring=scoring, random_state=42,
        )
        search.fit(X, y)

        best = search.best_estimator_
        setattr(self, spec['attr'], best)
        self.cv_scores[key] = float(search.best_score_)
        logger.info(f"  → Meilleur score CV ({scoring}) : {search.best_score_:.4f}")
        return best

    # ── API publique compatible ascendante ────────────────────────────────
    def train_regression(self, X: pd.DataFrame, y: pd.Series,
                         subject_name: Optional[str] = None) -> Any:
        """Entraîne XGBoost en régression. Si subject_name est fourni, stocké dans subject_models."""
        if subject_name:
            logger.info(f"Entraînement par matière : {subject_name}")
            pipeline = self._build_pipeline('regression', MODEL_CATALOG['reg']['estimator']())
            search = RandomizedSearchCV(
                pipeline, MODEL_CATALOG['reg']['param_dist'],
                n_iter=10, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring='r2', random_state=42,
            )
            search.fit(X, y)
            self.subject_models[subject_name] = search.best_estimator_
            return self.subject_models[subject_name]
        return self._train('reg', X, y)

    def train_classification(self, X: pd.DataFrame, y: pd.Series) -> Any:
        return self._train('clf', X, y)

    def train_nn_regression(self, X: pd.DataFrame, y: pd.Series) -> Any:
        return self._train('nn_reg', X, y)

    def train_nn_classification(self, X: pd.DataFrame, y: pd.Series) -> Any:
        return self._train('nn_clf', X, y)

    def train_svm_regression(self, X: pd.DataFrame, y: pd.Series) -> Any:
        return self._train('svm_reg', X, y)

    def train_svm_classification(self, X: pd.DataFrame, y: pd.Series) -> Any:
        return self._train('svm_clf', X, y)

    def train_all(self, X: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series,
                  include_svm: bool = False) -> Dict[str, Any]:
        """
        Entraîne tous les modèles du catalogue et sélectionne le meilleur
        par tâche selon le score CV. Met à jour best_overall_reg/clf.
        """
        keys = ['reg', 'clf', 'nn_reg', 'nn_clf']
        if include_svm:
            keys += ['svm_reg', 'svm_clf']

        for key in keys:
            y = y_reg if MODEL_CATALOG[key]['task'] == 'regression' else y_clf
            self._train(key, X, y)

        # Sélection du meilleur modèle par tâche
        reg_keys = [k for k in keys if MODEL_CATALOG[k]['task'] == 'regression']
        clf_keys = [k for k in keys if MODEL_CATALOG[k]['task'] == 'classification']
        if reg_keys:
            best_reg_key = max(reg_keys, key=lambda k: self.cv_scores.get(k, -np.inf))
            self.best_overall_reg = getattr(self, MODEL_CATALOG[best_reg_key]['attr'])
            logger.info(f"Meilleur modèle régression : {MODEL_CATALOG[best_reg_key]['label']}")
        if clf_keys:
            best_clf_key = max(clf_keys, key=lambda k: self.cv_scores.get(k, -np.inf))
            self.best_overall_clf = getattr(self, MODEL_CATALOG[best_clf_key]['attr'])
            logger.info(f"Meilleur modèle classification : {MODEL_CATALOG[best_clf_key]['label']}")

        return self.cv_scores

    def save_models(self, path: str = MODEL_FILE) -> None:
        """Sauvegarde les modèles et les métadonnées sur disque."""
        joblib.dump({
            'reg': self.best_model_reg,
            'clf': self.best_model_clf,
            'nn_reg': self.best_model_nn_reg,
            'nn_clf': self.best_model_nn_clf,
            'svm_reg': self.best_model_svm_reg,
            'svm_clf': self.best_model_svm_clf,
            'best_reg': self.best_overall_reg,
            'best_clf': self.best_overall_clf,
            'subject_models': self.subject_models,
            'feature_columns': self.feature_columns,
            'cv_scores': self.cv_scores,
        }, path)
        logger.info(f"Modèles sauvegardés dans {path}")

    def load_models(self, path: str = MODEL_FILE) -> bool:
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
            self.feature_columns = dict_models.get('feature_columns')
            self.cv_scores = dict_models.get('cv_scores', {})
            logger.info("Modèles chargés avec succès.")
            return True
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Impossible de charger les modèles depuis {path} : {e}")
            return False
        except Exception as e:
            logger.warning(f"Erreur inattendue au chargement des modèles depuis {path} : {e}")
            return False
