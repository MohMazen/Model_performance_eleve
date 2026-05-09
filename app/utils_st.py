import streamlit as st
from typing import List, Optional
import pandas as pd


def _get(key, default=None):
    return st.session_state.get(key, default)


def _set(key, value):
    st.session_state[key] = value


# ── Wrappers avec cache Streamlit ─────────────────────────────────────────
# Le cache @st.cache_data appartient à la couche UI, pas au backend.
# Ces wrappers encapsulent les fonctions backend lourdes avec le cache Streamlit
# de sorte que src/data_utils.py reste un module Python pur et testable.

@st.cache_data(show_spinner=False)
def cached_generer_donnees(n_eleves: int, classes_selectionnees: Optional[List[str]] = None) -> pd.DataFrame:
    """Wrapper caché de generer_donnees_synthetiques."""
    from src.data_utils import generer_donnees_synthetiques
    return generer_donnees_synthetiques(n_eleves, classes_selectionnees)


@st.cache_data(show_spinner=False)
def cached_charger_donnees(chemin: str) -> Optional[pd.DataFrame]:
    """Wrapper caché de charger_donnees."""
    from src.data_utils import charger_donnees
    return charger_donnees(chemin)


@st.cache_data(show_spinner=False)
def cached_nettoyer_donnees(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Wrapper caché de nettoyer_donnees."""
    from src.data_utils import nettoyer_donnees
    return nettoyer_donnees(df)
