import streamlit as st

def _get(key, default=None):
    return st.session_state.get(key, default)

def _set(key, value):
    st.session_state[key] = value
