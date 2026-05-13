"""
Tests de non-régression sur les fuites de données.

Vérifie que :
- Les colonnes brutes de notes (GRADE_COLUMNS) ne se retrouvent pas dans X
  après preprocessing.
- La cible de classification (TARGET_CLF) n'est pas présente dans X.
- Le préprocesseur du ModelManager n'utilise pas la cible.
- Les features avancées ne fuitent pas la note via une feature dérivée
  trivialement bijective.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import generer_donnees_synthetiques, nettoyer_donnees
from src.features import nettoyer_horaires, add_advanced_features
from src.config import COLS_TO_DROP, TARGET_REG, TARGET_CLF, GRADE_COLUMNS, ID_COLUMNS
from src.models import ModelManager


@pytest.fixture(scope="module")
def df_prepared():
    df = generer_donnees_synthetiques(n_eleves=80)
    df = nettoyer_donnees(df)
    df = nettoyer_horaires(df)
    df = add_advanced_features(df)
    return df


def _build_X(df: pd.DataFrame) -> pd.DataFrame:
    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_drop + [TARGET_REG, TARGET_CLF])


class TestNoTargetLeakage:
    def test_targets_not_in_X(self, df_prepared):
        X = _build_X(df_prepared)
        assert TARGET_REG not in X.columns, "La cible régression ne doit pas être dans X."
        assert TARGET_CLF not in X.columns, "La cible classification ne doit pas être dans X."

    def test_grade_columns_excluded_from_X(self, df_prepared):
        X = _build_X(df_prepared)
        leaking = [c for c in GRADE_COLUMNS if c in X.columns]
        assert leaking == [], (
            f"Les notes brutes {leaking} doivent être exclues : "
            "elles permettent de reconstituer note_moyenne (fuite directe)."
        )

    def test_id_columns_excluded_from_X(self, df_prepared):
        X = _build_X(df_prepared)
        leaking = [c for c in ID_COLUMNS if c in X.columns]
        assert leaking == [], (
            f"Les colonnes d'identité {leaking} doivent être exclues "
            "(non-prédictives et risque RGPD)."
        )

    def test_horaires_textuels_exclus_du_modele(self, df_prepared):
        """heure_coucher / heure_lever sont dans COLS_TO_DROP : seuls leurs
        équivalents numériques (_num) doivent rester."""
        X = _build_X(df_prepared)
        for col in ['heure_coucher', 'heure_lever']:
            assert col not in X.columns, f"{col} ne doit pas être dans X."


class TestNoTrivialFeatureLeak:
    def test_no_perfect_correlation_with_target(self, df_prepared):
        """Aucune feature numérique ne doit être parfaitement corrélée à la cible
        (corrélation = ±1.0), ce qui révèlerait une fuite déterministe."""
        X = _build_X(df_prepared)
        y = df_prepared[TARGET_REG]
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if X[col].nunique() <= 1:
                continue
            corr = np.corrcoef(X[col].fillna(0).values, y.values)[0, 1]
            assert abs(corr) < 0.999, (
                f"Feature '{col}' corrélée à {corr:.3f} avec la cible — "
                "fuite probable."
            )


class TestPipelineDoesNotSeeTarget:
    def test_preprocessor_columns_exclude_target(self, df_prepared):
        """Le ColumnTransformer construit par prepare_pipeline ne doit
        contenir aucune référence aux colonnes cibles."""
        X = _build_X(df_prepared)
        mm = ModelManager()
        pre = mm.prepare_pipeline(X)
        all_cols_in_pre = []
        for _, _, cols in pre.transformers:
            all_cols_in_pre.extend(cols)
        assert TARGET_REG not in all_cols_in_pre
        assert TARGET_CLF not in all_cols_in_pre
        for grade in GRADE_COLUMNS:
            assert grade not in all_cols_in_pre, (
                f"Le préprocesseur référence la note brute '{grade}'."
            )


class TestImputationAfterSplit:
    """Vérifie que l'imputation statistique est déléguée au pipeline sklearn
    (post-split) et non appliquée globalement avant le split dans nettoyer_donnees."""

    def test_nettoyer_donnees_preserve_les_nan(self):
        """nettoyer_donnees ne doit plus imputer les valeurs manquantes numériques."""
        df = generer_donnees_synthetiques(n_eleves=40)
        df.loc[0, 'heures_etude_soir'] = np.nan
        cleaned = nettoyer_donnees(df.copy())
        assert cleaned['heures_etude_soir'].isna().any(), (
            "nettoyer_donnees ne doit pas imputer les NaN numériques — "
            "l'imputation est déléguée au pipeline sklearn (post-split)."
        )

    def test_pipeline_contient_imputer_numerique(self, df_prepared):
        """La branche numérique du pipeline doit contenir un SimpleImputer."""
        from sklearn.impute import SimpleImputer
        X = _build_X(df_prepared)
        mm = ModelManager()
        mm.prepare_pipeline(X)
        # Accès aux transformers non fittés via .transformers (liste de tuples).
        tr_dict = {name: tr for name, tr, _ in mm.preprocessor.transformers}
        num_pipe = tr_dict['num']
        assert 'imputer' in num_pipe.named_steps, (
            "prepare_pipeline doit inclure un SimpleImputer dans la branche numérique."
        )
        assert isinstance(num_pipe.named_steps['imputer'], SimpleImputer)
        assert num_pipe.named_steps['imputer'].strategy == 'median'

    def test_pipeline_contient_imputer_categoriel(self, df_prepared):
        """La branche catégorielle du pipeline doit contenir un SimpleImputer."""
        from sklearn.impute import SimpleImputer
        X = _build_X(df_prepared)
        mm = ModelManager()
        mm.prepare_pipeline(X)
        tr_dict = {name: tr for name, tr, _ in mm.preprocessor.transformers}
        cat_pipe = tr_dict['cat']
        assert 'imputer' in cat_pipe.named_steps, (
            "prepare_pipeline doit inclure un SimpleImputer dans la branche catégorielle."
        )
        assert isinstance(cat_pipe.named_steps['imputer'], SimpleImputer)
        assert cat_pipe.named_steps['imputer'].strategy == 'most_frequent'

    def test_pipeline_gere_nan_du_test_set(self, df_prepared):
        """Un NaN introduit dans le test set doit être géré par l'imputer
        du pipeline entraîné sur le train set (sans erreur et sans voir le test)."""
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor
        X = _build_X(df_prepared)
        y = df_prepared[TARGET_REG]
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        feat = X_train.select_dtypes(include=[np.number]).columns[0]
        X_test_nan = X_test.copy()
        X_test_nan.iloc[0, X_test_nan.columns.get_loc(feat)] = np.nan

        mm = ModelManager()
        mm.prepare_pipeline(X_train)
        pipeline = mm._build_pipeline('regression', XGBRegressor(random_state=42, n_estimators=10))
        pipeline.fit(X_train, y_train)

        # La prédiction ne doit pas lever d'exception et doit renvoyer un tableau valide.
        preds = pipeline.predict(X_test_nan)
        assert len(preds) == len(X_test_nan)
        assert not np.isnan(preds).any(), "Les prédictions ne doivent pas contenir de NaN."
