
import os
import sys
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.temporal import TemporalAnalyzer, generer_donnees_multi_periodes
from src.config import TARGET_REG, TARGET_CLF
from app.utils_st import _get, _set

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("📈 Suivi Temporel")
st.markdown("Suivez l'évolution des performances sur plusieurs périodes.")

tab_upload, tab_synth = st.tabs(["📁 Charger des périodes", "🔄 Générer des données multi-périodes"])

with tab_synth:
    st.subheader("Générer des données synthétiques multi-périodes")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_eleves_t = st.number_input("Nombre d'élèves", 30, 500, 100, 10, key="temp_n")
    with col_s2:
        n_periodes = st.number_input("Nombre de périodes (trimestres)", 2, 6, 3, 1, key="temp_p")

    if st.button("🔄 Générer", key="btn_gen_temp"):
        with st.spinner("Génération multi-périodes…"):
            df_multi = generer_donnees_multi_periodes(int(n_eleves_t), int(n_periodes))
            _set("df_multi", df_multi)
            st.success(f"✅ {len(df_multi)} lignes générées sur {n_periodes} périodes.")

with tab_upload:
    st.subheader("Charger plusieurs fichiers CSV")
    uploaded_files = st.file_uploader("Fichiers CSV (un par période)", type=["csv"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) >= 2:
        frames = []
        for i, f in enumerate(uploaded_files):
            try:
                df_up = pd.read_csv(f, sep=None, engine='python', encoding='utf-8-sig')
                df_up.columns = [c.lower() for c in df_up.columns]
                label = f"Période {i+1} ({f.name})"
                frames.append((df_up, label))
            except Exception as e:
                st.error(f"Erreur pour {f.name} : {e}")

        if frames and st.button("📥 Fusionner les périodes"):
            ta = TemporalAnalyzer()
            df_multi = ta.merge_uploaded_periods(frames)
            _set("df_multi", df_multi)
            st.success(f"✅ {len(df_multi)} lignes fusionnées sur {len(frames)} périodes.")
    elif uploaded_files:
        st.info("Veuillez charger au moins 2 fichiers CSV pour le suivi temporel.")

df_multi = _get("df_multi")
if df_multi is None:
    st.info("Générez ou chargez des données multi-périodes pour commencer.")
    st.stop()

st.markdown("---")
target_reg = _get("target_reg", TARGET_REG)
id_col = "nom"

# Détection auto de la colonne ID
if id_col not in df_multi.columns:
    possible_ids = [c for c in df_multi.columns if c.lower() in ['nom', 'id', 'eleve', 'student']]
    id_col = possible_ids[0] if possible_ids else df_multi.columns[0]

ta = TemporalAnalyzer(id_col=id_col, target=target_reg)

# Résumé par période
st.subheader("📊 Résumé par période")
if target_reg in df_multi.columns:
    summary = ta.get_period_summary(df_multi)
    st.dataframe(summary, use_container_width=True)

    # Graphique d'évolution de la moyenne
    fig_evo = px.bar(
        summary.reset_index(), x="periode", y="moyenne",
        color="taux_reussite", color_continuous_scale="RdYlGn",
        title="Évolution de la moyenne par période",
        labels={"moyenne": "Moyenne", "periode": "Période", "taux_reussite": "Taux réussite (%)"},
        text="moyenne",
    )
    fig_evo.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_evo, use_container_width=True)

# Tendances
st.markdown("---")
st.subheader("📈 Tendances individuelles")

if st.button("🔍 Calculer les tendances"):
    with st.spinner("Analyse des trajectoires…"):
        try:
            trends = ta.compute_trends(df_multi)
            _set("trends", trends)
            st.success(f"✅ Tendances calculées pour {len(trends)} élèves.")
        except Exception as e:
            st.error(f"Erreur : {e}")

trends = _get("trends")
if trends is not None:
    # Compteurs de tendances
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        n_prog = len(trends[trends["trend_label"].str.contains("Progression")])
        st.metric("↗️ En progression", n_prog)
    with col_t2:
        n_stable = len(trends[trends["trend_label"].str.contains("Stable")])
        st.metric("➡️ Stables", n_stable)
    with col_t3:
        n_reg = len(trends[trends["trend_label"].str.contains("Régression")])
        st.metric("↘️ En régression", n_reg)

    # Tableau des tendances
    display_cols_t = [c for c in [id_col, "trend_label", "slope", "first_value", "last_value", "delta", "n_periods"] if c in trends.columns]
    st.dataframe(
        trends[display_cols_t].sort_values("slope", ascending=True),
        use_container_width=True,
    )

    # Scatter des pentes
    fig_slope = px.histogram(
        trends, x="slope", color="trend_label",
        color_discrete_map={"↗️ Progression": "#4caf50", "➡️ Stable": "#ffc107", "↘️ Régression": "#f44336"},
        title="Distribution des pentes de tendance",
        labels={"slope": "Pente (points/période)", "trend_label": "Tendance"},
        nbins=20,
    )
    st.plotly_chart(fig_slope, use_container_width=True)

    # Détection de décrochage
    st.markdown("---")
    st.subheader("🚨 Détection de décrochage")
    slope_thresh = st.slider("Seuil de pente (points/période)", -3.0, 0.0, -1.0, 0.1)
    at_risk = ta.detect_dropout_risk(trends, slope_threshold=slope_thresh)

    if at_risk.empty:
        st.success("🎉 Aucun élève en situation de décrochage détecté.")
    else:
        st.error(f"**{len(at_risk)} élève(s)** en risque de décrochage.")
        st.dataframe(at_risk[display_cols_t], use_container_width=True)

    # Trajectoire individuelle
    st.markdown("---")
    st.subheader("📉 Trajectoire individuelle")
    students = df_multi[id_col].unique().tolist()
    sel_student = st.selectbox("Sélectionnez un élève", students[:50], key="traj_student")

    if sel_student:
        student_data = df_multi[df_multi[id_col] == sel_student].sort_values("periode_order")
        if target_reg in student_data.columns:
            fig_traj = px.line(
                student_data, x="periode", y=target_reg,
                title=f"Trajectoire de {sel_student}",
                markers=True,
                labels={target_reg: "Note moyenne", "periode": "Période"},
            )
            fig_traj.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Seuil réussite")
            st.plotly_chart(fig_traj, use_container_width=True)

# ── Courbe de prédictibilité progressive ────────────────────────────────────
st.markdown("---")
st.subheader("📊 Courbe de prédictibilité progressive")
st.markdown(
    "Inspiré de Muresan et al. (2026) : comment la précision du modèle évolue-t-elle "
    "selon la quantité de données disponibles ? Chaque point correspond à un modèle "
    "entraîné sur les données agrégées des *k* premières périodes."
)

periods_available = sorted(df_multi["periode_order"].unique())

if len(periods_available) < 2:
    st.info("Au moins 2 périodes sont nécessaires pour tracer la courbe de prédictibilité.")
elif TARGET_CLF not in df_multi.columns:
    st.info(f"Colonne cible '{TARGET_CLF}' absente des données — courbe non disponible.")
else:
    if st.button("📈 Calculer la courbe de prédictibilité", key="btn_pred_curve"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_validate, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        num_cols = df_multi.select_dtypes(include="number").columns.tolist()
        feature_cols = [
            c for c in num_cols
            if c not in ("periode_order", target_reg, TARGET_CLF)
        ]

        clf_pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, random_state=42),
        )

        curve_rows = []
        n_total = len(periods_available)

        with st.spinner(f"Calcul sur {n_total} période(s)…"):
            for k in range(1, n_total + 1):
                sub = df_multi[df_multi["periode_order"] < k]
                agg_feats = sub.groupby(id_col)[feature_cols].mean()
                y_per_student = sub.groupby(id_col)[TARGET_CLF].last()

                common = agg_feats.index.intersection(y_per_student.index)
                X_k = agg_feats.loc[common]
                y_k = y_per_student.loc[common]

                n_classes = len(y_k.unique())
                min_class = int(y_k.value_counts().min()) if n_classes >= 2 else 0

                if len(X_k) >= 10 and n_classes >= 2 and min_class >= 2:
                    n_splits = max(2, min(5, min_class))
                    cv_obj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    try:
                        scores = cross_validate(
                            clf_pipe, X_k, y_k, cv=cv_obj,
                            scoring=["f1", "accuracy"],
                        )
                        curve_rows.append({
                            "periode": f"P{k}",
                            "pct_disponible": round(k / n_total * 100, 1),
                            "f1": round(float(scores["test_f1"].mean()), 3),
                            "accuracy": round(float(scores["test_accuracy"].mean()), 3),
                            "n_eleves": len(X_k),
                        })
                    except Exception as exc:
                        logger.warning("Erreur période %d : %s", k, exc)

        if curve_rows:
            df_curve = pd.DataFrame(curve_rows)
            _set("predictability_curve", df_curve)

            df_melted = df_curve.melt(
                id_vars=["periode", "pct_disponible", "n_eleves"],
                value_vars=["f1", "accuracy"],
                var_name="Métrique", value_name="Score",
            )
            fig_curve = px.line(
                df_melted, x="pct_disponible", y="Score", color="Métrique",
                markers=True,
                title="Prédictibilité progressive selon les périodes disponibles",
                labels={
                    "pct_disponible": "% données disponibles",
                    "Score": "Score (CV stratifié)",
                },
                color_discrete_map={"f1": "#2196f3", "accuracy": "#4caf50"},
            )
            fig_curve.update_xaxes(ticksuffix="%", range=[0, 105])
            fig_curve.update_yaxes(range=[0, 1.05])
            fig_curve.add_hline(
                y=0.7, line_dash="dot", line_color="orange",
                annotation_text="Seuil acceptable (0.70)",
            )
            st.plotly_chart(fig_curve, use_container_width=True)
            st.dataframe(df_curve, use_container_width=True)

            if len(df_curve) >= 2:
                f1_first = df_curve["f1"].iloc[0]
                f1_last = df_curve["f1"].iloc[-1]
                gain = f1_last - f1_first
                st.info(
                    f"📈 Gain de F1 : **{f1_first:.3f}** (période 1) → "
                    f"**{f1_last:.3f}** (toutes périodes) — gain total : **{gain:+.3f}**."
                )
        else:
            st.warning("Données insuffisantes pour calculer la courbe (< 10 élèves ou une seule classe).")
