"""
Tests pour l'import de notes détaillées au format long (ÉcoleDirecte / Pronote).

Vérifie que :
- L'encodage cp1252 (typique d'ÉcoleDirecte) est correctement décodé
- Les noms contenant une virgule interne sont préservés
- Les notes sur /10 (Contrôles) sont normalisées vers /20
- La ligne 39 (Étude de texte / Contrôle 6/10 coeff 1) est intégrée à la moyenne
- Les Examens et Contrôles coexistent (pas de dé-duplication sur le libellé)
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.import_notes_detaillees import (
    aggreger_par_matiere,
    charger_notes_detaillees,
    normaliser_note_sur_20,
    parser_note,
    pivoter_notes_wide,
    preparer_notes_detaillees,
)


# ── Données de test reproduisant exactement le cas utilisateur ────────────────
CSV_ARABE = (
    'index,nom_eleve,date,trimestre,matiere,prof,libelle,type,note,coeff,extra\n'
    '1,"Ali Seghir Abdelkader,Ahmed","Mercredi 03 décembre 2025","1er Trimestre","Arabe","-","Dictée","Examen","19 / 20","2.00",""\n'
    '2,"Ali Seghir Abdelkader,Ahmed","Mercredi 03 décembre 2025","1er Trimestre","Arabe","-","Écriture","Examen","16 / 20","2.00",""\n'
    '3,"Ali Seghir Abdelkader,Ahmed","Mercredi 03 décembre 2025","1er Trimestre","Arabe","-","Étude de texte","Examen","10 / 20","2.00",""\n'
    '4,"Ali Seghir Abdelkader,Ahmed","Mercredi 03 décembre 2025","1er Trimestre","Arabe","-","Expression ecrite","Examen","16 / 20","2.00",""\n'
    '5,"Ali Seghir Abdelkader,Ahmed","Mercredi 03 décembre 2025","1er Trimestre","Arabe","-","Quawa3ed","Examen","15 / 20","2.00",""\n'
    '39,"Ali Seghir Abdelkader,Ahmed","Jeudi 30 octobre 2025","1er Trimestre","Arabe","-","Étude de texte","Contrôle","6 / 10","1.00",""\n'
)


# ═════════════════════════════════════════════════════════════════════════════
# Parsers unitaires
# ═════════════════════════════════════════════════════════════════════════════
class TestParserNote:
    def test_note_sur_20_standard(self):
        assert parser_note("19 / 20") == (19.0, 20.0)

    def test_note_sur_10_controle(self):
        assert parser_note("6 / 10") == (6.0, 10.0)

    def test_note_avec_virgule_decimale(self):
        assert parser_note("15,5 / 20") == (15.5, 20.0)

    def test_note_avec_point_decimal(self):
        assert parser_note("15.5 / 20") == (15.5, 20.0)

    def test_note_non_parsable(self):
        n, d = parser_note("Abs")
        assert np.isnan(n) and np.isnan(d)

    def test_note_denominateur_zero(self):
        n, d = parser_note("5 / 0")
        assert np.isnan(n) and np.isnan(d)

    def test_note_non_string(self):
        n, d = parser_note(None)
        assert np.isnan(n) and np.isnan(d)


class TestNormaliserNoteSur20:
    def test_note_deja_sur_20(self):
        assert normaliser_note_sur_20("19 / 20") == 19.0

    def test_note_sur_10_convertie(self):
        # 6/10 = 12/20 — c'est le cœur du bug utilisateur
        assert normaliser_note_sur_20("6 / 10") == 12.0

    def test_note_sur_100(self):
        assert normaliser_note_sur_20("85 / 100") == 17.0


# ═════════════════════════════════════════════════════════════════════════════
# Chargement CSV
# ═════════════════════════════════════════════════════════════════════════════
class TestChargerNotesDetaillees:
    def test_charge_csv_utf8(self):
        df = charger_notes_detaillees(io.StringIO(CSV_ARABE))
        assert len(df) == 6  # 5 Examens + 1 Contrôle → 6 lignes
        assert 'note' in df.columns

    def test_preserve_nom_avec_virgule_interne(self):
        """Le nom 'Ali Seghir Abdelkader,Ahmed' contient une virgule interne
        et doit être préservé grâce au CSV quoting."""
        df = charger_notes_detaillees(io.StringIO(CSV_ARABE))
        noms = df['nom_eleve'].unique().tolist()
        assert noms == ["Ali Seghir Abdelkader,Ahmed"]

    def test_cascade_encodage_cp1252(self, tmp_path):
        """ÉcoleDirecte exporte typiquement en CP1252 — vérifier le fallback."""
        chemin = tmp_path / "notes_cp1252.csv"
        chemin.write_bytes(CSV_ARABE.encode('cp1252'))
        df = charger_notes_detaillees(str(chemin))
        # « décembre » et « Contrôle » sont correctement décodés
        assert any("décembre" in str(v) for v in df['date'].values)
        assert any("Contrôle" in str(v) for v in df['type'].values)


# ═════════════════════════════════════════════════════════════════════════════
# Préparation et agrégation — le cœur du bug utilisateur
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def df_prepare():
    df_raw = charger_notes_detaillees(io.StringIO(CSV_ARABE))
    return preparer_notes_detaillees(df_raw)


class TestPreparerNotes:
    def test_toutes_les_lignes_conservees(self, df_prepare):
        """Aucune ligne ne doit être écartée — Examens ET Contrôles."""
        assert len(df_prepare) == 6

    def test_ligne_controle_note_normalisee(self, df_prepare):
        """La ligne 39 (Contrôle 6/10) doit devenir 12/20."""
        ligne_controle = df_prepare[df_prepare['type_evaluation'] == 'Contrôle']
        assert len(ligne_controle) == 1
        assert float(ligne_controle['note_sur_20'].iloc[0]) == 12.0
        assert float(ligne_controle['coeff_num'].iloc[0]) == 1.0

    def test_examen_et_controle_meme_libelle_coexistent(self, df_prepare):
        """Les deux 'Étude de texte' (Examen et Contrôle) doivent coexister."""
        etudes = df_prepare[df_prepare['libelle'] == 'Étude de texte']
        assert len(etudes) == 2
        assert set(etudes['type_evaluation']) == {'Examen', 'Contrôle'}


class TestAggregation:
    def test_moyenne_ponderee_inclut_le_controle(self, df_prepare):
        """
        Vérifie le calcul complet avec la ligne 39 :
          (19+16+10+16+15) × 2 + 12 × 1 = 164
          Coeffs totaux : 5×2 + 1 = 11
          Moyenne = 164 / 11 = 14.909... ≈ 14.91
        """
        agg = aggreger_par_matiere(df_prepare)
        assert len(agg) == 1  # 1 élève × 1 matière
        row = agg.iloc[0]
        assert row['nom_eleve'] == "Ali Seghir Abdelkader,Ahmed"
        assert row['matiere'] == "Arabe"
        assert row['nb_notes'] == 6, "6 notes attendues (5 examens + 1 contrôle)"
        assert row['moyenne_ponderee'] == pytest.approx(14.91, abs=0.01)

    def test_moyenne_change_si_on_exclut_le_controle(self, df_prepare):
        """Contre-preuve : sans la ligne 39, la moyenne serait 15.20."""
        df_sans_controle = df_prepare[df_prepare['type_evaluation'] != 'Contrôle']
        agg = aggreger_par_matiere(df_sans_controle)
        assert agg.iloc[0]['moyenne_ponderee'] == pytest.approx(15.20, abs=0.01)
        assert agg.iloc[0]['nb_notes'] == 5


class TestPivotWide:
    def test_pivot_produit_note_arabe(self, df_prepare):
        agg = aggreger_par_matiere(df_prepare)
        wide = pivoter_notes_wide(agg)
        assert 'note_arabe' in wide.columns
        assert wide['note_arabe'].iloc[0] == pytest.approx(14.91, abs=0.01)
