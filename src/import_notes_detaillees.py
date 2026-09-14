"""
Import de notes détaillées au format long (export ÉcoleDirecte / Pronote).

Format d'entrée attendu (1 ligne = 1 note d'évaluation) :
    index, nom_eleve, date, trimestre, matiere, [prof], libelle, type,
    note ("X / Y"), coefficient, [extra]

Objectif : produire une note pondérée par (élève, matière) qui intègre TOUS
les types d'évaluation (Examens ET Contrôles) et normalise les notes sur /20
quel que soit le barème d'origine.

Points de vigilance corrigés :
- Encodage : cascade utf-8-sig → cp1252 → latin-1 (ÉcoleDirecte exporte en cp1252)
- Types "Contrôle" NON filtrés (conservés avec leur coefficient)
- Notes /10 (ou tout autre barème) normalisées vers /20
- Pas de dé-duplication sur le libellé (Examen et Contrôle du même chapitre coexistent)
- Noms d'élèves contenant une virgule (ex. "Ali Seghir Abdelkader,Ahmed") préservés
  grâce au parseur CSV Python standard (quoted fields).
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ENCODAGES_CANDIDATS = ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1')


def _detecter_col(df: pd.DataFrame, motifs: Tuple[str, ...]) -> Optional[str]:
    """Retourne la première colonne dont le nom contient l'un des motifs."""
    for col in df.columns:
        col_low = str(col).lower()
        if any(m in col_low for m in motifs):
            return col
    return None


def charger_notes_detaillees(
    chemin_ou_buffer,
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """
    Charge un CSV long-format avec cascade d'encodages.

    Parameters
    ----------
    chemin_ou_buffer : chemin fichier OU objet fichier-like OU StringIO
    sep              : séparateur (None → auto-détection csv.Sniffer)
    """
    dernier_err: Optional[Exception] = None

    for encodage in _ENCODAGES_CANDIDATS:
        try:
            if hasattr(chemin_ou_buffer, 'seek'):
                chemin_ou_buffer.seek(0)
            df = pd.read_csv(
                chemin_ou_buffer,
                sep=sep,
                engine='python',
                encoding=encodage,
                dtype=str,
                keep_default_na=False,
                na_values=['', 'NaN', 'nan', 'N/A', 'null'],
            )
            df.columns = [str(c).strip() for c in df.columns]
            logger.info(
                "Fichier chargé (encodage=%s, %d ligne(s), %d colonne(s))",
                encodage, len(df), len(df.columns),
            )
            return df
        except UnicodeDecodeError as e:
            logger.debug("Encodage %s a échoué : %s", encodage, e)
            dernier_err = e
            continue

    raise ValueError(
        f"Aucun encodage parmi {_ENCODAGES_CANDIDATS} n'a fonctionné : {dernier_err}"
    )


def parser_note(note_str) -> Tuple[float, float]:
    """
    Extrait (numérateur, dénominateur) d'une note format 'X / Y'.
    Retourne (NaN, NaN) si non parsable.
    """
    if not isinstance(note_str, str):
        return (np.nan, np.nan)
    m = re.match(r'\s*(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)', note_str)
    if not m:
        return (np.nan, np.nan)
    try:
        num = float(m.group(1).replace(',', '.'))
        denom = float(m.group(2).replace(',', '.'))
    except ValueError:
        return (np.nan, np.nan)
    if denom == 0:
        return (np.nan, np.nan)
    return (num, denom)


def normaliser_note_sur_20(note_str) -> float:
    """Convertit 'X / Y' en note sur 20. NaN si non parsable."""
    num, denom = parser_note(note_str)
    if np.isnan(num) or np.isnan(denom):
        return np.nan
    return round(num * 20.0 / denom, 2)


def preparer_notes_detaillees(
    df: pd.DataFrame,
    col_nom: Optional[str] = None,
    col_matiere: Optional[str] = None,
    col_note: Optional[str] = None,
    col_coeff: Optional[str] = None,
    col_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ajoute les colonnes calculées `note_sur_20` et `coeff_num`.
    Ne filtre AUCUNE ligne (les Contrôles restent, les libellés dupliqués aussi).
    """
    df = df.copy()

    col_nom     = col_nom     or _detecter_col(df, ('nom', 'élève', 'eleve', 'student'))
    col_matiere = col_matiere or _detecter_col(df, ('matière', 'matiere', 'subject', 'discipline'))
    col_note    = col_note    or _detecter_col(df, ('note',))
    col_coeff   = col_coeff   or _detecter_col(df, ('coeff', 'coef', 'pond'))
    col_type    = col_type    or _detecter_col(df, ('type', 'catégorie', 'categorie', 'evaluation'))

    for label, col in [('nom', col_nom), ('matiere', col_matiere),
                       ('note', col_note), ('coeff', col_coeff)]:
        if col is None:
            raise ValueError(f"Colonne '{label}' introuvable — colonnes disponibles : {list(df.columns)}")

    df['note_sur_20'] = df[col_note].apply(normaliser_note_sur_20)
    df['coeff_num'] = pd.to_numeric(
        df[col_coeff].astype(str).str.replace(',', '.').str.strip(),
        errors='coerce',
    ).fillna(1.0)

    n_rejetees = int(df['note_sur_20'].isna().sum())
    if n_rejetees:
        logger.warning("%d note(s) non parsable(s) — format attendu 'X / Y'.", n_rejetees)

    renommage = {col_nom: 'nom_eleve', col_matiere: 'matiere'}
    if col_type:
        renommage[col_type] = 'type_evaluation'
    df = df.rename(columns=renommage)

    return df


def aggreger_par_matiere(df_notes: pd.DataFrame) -> pd.DataFrame:
    """
    Moyenne pondérée par (nom_eleve, matiere), INCLUANT tous les types d'évaluation.

    Formule : sum(note_sur_20 * coeff) / sum(coeff)
    """
    df_ok = df_notes.dropna(subset=['note_sur_20']).copy()
    df_ok['produit'] = df_ok['note_sur_20'] * df_ok['coeff_num']

    grouped = df_ok.groupby(['nom_eleve', 'matiere']).agg(
        somme_produits=('produit', 'sum'),
        somme_coeffs=('coeff_num', 'sum'),
        nb_notes=('note_sur_20', 'count'),
    )
    grouped['moyenne_ponderee'] = (grouped['somme_produits'] / grouped['somme_coeffs']).round(2)
    return grouped.reset_index()[['nom_eleve', 'matiere', 'moyenne_ponderee', 'nb_notes']]


def pivoter_notes_wide(df_agg: pd.DataFrame, prefixe: str = 'note_') -> pd.DataFrame:
    """
    Transforme le format long agrégé en format wide compatible EduStats :
      note_arabe, note_maths, note_francais, ...
    """
    def _slug(s: str) -> str:
        s = str(s).lower().strip()
        for old, new in [('é', 'e'), ('è', 'e'), ('ê', 'e'), ('à', 'a'),
                         ('ô', 'o'), ('û', 'u'), ('ç', 'c'), ('î', 'i'), ('ï', 'i')]:
            s = s.replace(old, new)
        return re.sub(r'\s+', '_', s)

    df_wide = df_agg.pivot(index='nom_eleve', columns='matiere', values='moyenne_ponderee')
    df_wide.columns = [f"{prefixe}{_slug(c)}" for c in df_wide.columns]
    return df_wide.reset_index()
