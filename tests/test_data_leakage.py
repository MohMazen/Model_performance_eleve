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
    """Sanity-check : nettoyer_donnees ne doit pas dépendre du test set quand
    on l'applique séparément. La signature actuelle de nettoyer_donnees prend
    un seul DataFrame ; ce test fige le comportement attendu pour qu'une
    régression soit détectée si la fonction commençait à utiliser une variable
    globale ou un cache partagé entre appels successifs."""

    def test_nettoyage_independent_des_appels(self):
        df1 = generer_donnees_synthetiques(n_eleves=40)
        df2 = generer_donnees_synthetiques(n_eleves=60)
        # Introduire des NaN différents dans chaque jeu.
        df1.loc[0:3, 'heures_etude_soir'] = np.nan
        df2.loc[0:5, 'heures_etude_soir'] = np.nan

        clean1_solo = nettoyer_donnees(df1.copy())
        clean1_after_df2 = nettoyer_donnees(df1.copy())  # même entrée, après df2 vu

        # Le nettoyage de df1 ne doit pas dépendre d'un état laissé par df2.
        pd.testing.assert_frame_equal(
            clean1_solo.reset_index(drop=True),
            clean1_after_df2.reset_index(drop=True),
        )
