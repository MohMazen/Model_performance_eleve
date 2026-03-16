"""
Tests unitaires pour le module src.features.
"""
import sys
import os

# Permettre l'import du package src depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.features import parse_heure, add_advanced_features, prenttoyer_horaires
from src.data_utils import generer_donnees_synthetiques, nettoyer_donnees


class TestParseHeure:
    def test_format_heure_standard(self):
        assert parse_heure("22h30") == pytest.approx(22.5)

    def test_format_heure_zero_minutes(self):
        assert parse_heure("6h00") == pytest.approx(6.0)

    def test_format_heure_quinze_minutes(self):
        assert parse_heure("8h15") == pytest.approx(8.25)

    def test_valeur_invalide_retourne_zero(self):
        assert parse_heure("invalide") == pytest.approx(0.0)

    def test_valeur_none_retourne_zero(self):
        assert parse_heure(None) == pytest.approx(0.0)

    def test_valeur_float(self):
        assert parse_heure(7.5) == pytest.approx(7.5)


class TestAddAdvancedFeatures:
    @pytest.fixture
    def df_base(self):
        """DataFrame minimal pour les tests de features."""
        return generer_donnees_synthetiques(n_eleves=50)

    def test_colonne_score_equilibre_creee(self, df_base):
        df_result = add_advanced_features(df_base)
        assert 'score_equilibre' in df_result.columns

    def test_colonne_stress_total_creee(self, df_base):
        df_result = add_advanced_features(df_base)
        assert 'stress_total' in df_result.columns

    def test_colonne_reussite_creee(self, df_base):
        df_result = add_advanced_features(df_base)
        assert 'reussite' in df_result.columns

    def test_reussite_binaire(self, df_base):
        df_result = add_advanced_features(df_base)
        assert set(df_result['reussite'].unique()).issubset({0, 1})

    def test_score_equilibre_positif(self, df_base):
        df_result = add_advanced_features(df_base)
        assert (df_result['score_equilibre'] >= 0).all()

    def test_donnees_originales_non_modifiees(self, df_base):
        cols_avant = set(df_base.columns)
        add_advanced_features(df_base)
        assert set(df_base.columns) == cols_avant


class TestNettoyerDonnees:
    def test_pas_de_nan_apres_nettoyage(self):
        df = generer_donnees_synthetiques(n_eleves=100)
        # Introduire des NaN
        df.loc[0:5, 'heures_etude_soir'] = np.nan
        df.loc[3:7, 'classe'] = np.nan
        df_clean = nettoyer_donnees(df)
        assert df_clean.isnull().sum().sum() == 0

    def test_retourne_none_si_entree_none(self):
        assert nettoyer_donnees(None) is None

    def test_shape_preserve(self):
        df = generer_donnees_synthetiques(n_eleves=50)
        df_clean = nettoyer_donnees(df)
        assert df_clean.shape == df.shape
