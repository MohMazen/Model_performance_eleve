
import os
import sys
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.clustering import StudentProfiler
from src.config import TARGET_REG
from app.utils_st import _get, _set

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("🎯 Profils d'Élèves")
st.markdown("Segmentation comportementale par clustering non-supervisé.")
st.info(
    "**Usage pédagogique** : les groupes ci-dessous sont des regroupements "
    "statistiques basés sur des tendances comportementales agrégées. Ils décrivent "
    "des patterns collectifs observés dans les données, **non des catégories "
    "figées ni des jugements individuels**. Ces profils sont destinés aux "
    "équipes pédagogiques pour orienter un accompagnement collectif, pas à être "
    "communiqués directement aux élèves ou aux familles."
)

df_clean = _get("df_clean")
df_feat = _get("df_feat")

if df_clean is None:
    st.warning("⚠️ Veuillez d'abord Nettoyer les données (Page 2) pour continuer.")
    st.stop()
elif df_feat is None:
    st.warning("⚠️ Veuillez effectuer le Feature Engineering (Page 2) pour continuer.")
    st.stop()

# Paramètres
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    method = st.selectbox("Algorithme", ["kmeans", "dbscan"], index=0)
with col_p2:
    if method == "kmeans":
        auto_k = st.checkbox("Détection automatique du nombre de clusters", value=True)
        if not auto_k:
            n_clusters = st.slider("Nombre de clusters", 2, 8, 4)
        else:
            n_clusters = None
    else:
        n_clusters = None
with col_p3:
    viz_method = st.selectbox("Visualisation 2D", ["pca", "tsne"], index=0)

if st.button("🎯 Lancer le clustering", type="primary"):
    with st.spinner("Segmentation en cours…"):
        try:
            profiler = StudentProfiler()
            labels, X_used = profiler.fit_clusters(df_feat, method=method, n_clusters=n_clusters)
            profiles = profiler.describe_clusters(df_feat, labels)
            names = profiler.generate_cluster_names(profiles)
            coords_2d = profiler.reduce_dimensions(df_feat, method=viz_method)

            _set("cluster_labels", labels)
            _set("cluster_profiles", profiles)
            _set("cluster_names", names)
            _set("cluster_coords", coords_2d)
            _set("cluster_profiler", profiler)
            st.success(f"✅ {len(set(labels) - {-1})} profils identifiés.")
        except Exception as e:
            st.error(f"Erreur : {e}")
            logger.error(f"Erreur clustering : {e}", exc_info=True)

labels = _get("cluster_labels")
profiles = _get("cluster_profiles")
names = _get("cluster_names")
coords_2d = _get("cluster_coords")

if labels is None or profiles is None:
    st.info("Cliquez sur 'Lancer le clustering' pour démarrer.")
    st.stop()

# Scatter 2D interactif
st.markdown("---")
st.subheader("🗺️ Carte des profils (projection 2D)")

df_viz = pd.DataFrame(coords_2d, columns=["Dim1", "Dim2"])
df_viz["Cluster"] = labels
df_viz["Profil"] = [names.get(l, f"Profil {l}") for l in labels]
if "nom" in df_feat.columns:
    df_viz["Élève"] = df_feat["nom"].values
target_reg = _get("target_reg", TARGET_REG)
if target_reg in df_feat.columns:
    df_viz["Note"] = df_feat[target_reg].values

hover_data = ["Élève", "Note"] if "Élève" in df_viz.columns else ["Note"] if "Note" in df_viz.columns else None

fig_scatter = px.scatter(
    df_viz, x="Dim1", y="Dim2", color="Profil",
    hover_data=hover_data,
    title=f"Projection {viz_method.upper()} — Profils d'élèves",
    labels={"Dim1": f"Composante 1", "Dim2": f"Composante 2"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

# Résumé par cluster
st.markdown("---")
st.subheader("📊 Résumé des profils")

cols_per_row = min(len(profiles), 3)
cols = st.columns(cols_per_row)

for i, (cid, profile) in enumerate(sorted(profiles.items())):
    with cols[i % cols_per_row]:
        name = names.get(cid, f"Profil {cid}")
        st.markdown(f"### {name}")
        st.metric("Effectif", f"{profile['size']} ({profile['pct']}%)")
        if profile.get("note_moyenne") is not None:
            st.metric("Moyenne", f"{profile['note_moyenne']}/20")

        st.markdown("**Caractéristiques dominantes :**")
        for feat, z in profile["dominant_features"][:4]:
            direction = "🔼" if z > 0 else "🔽"
            feat_label = feat.replace("_", " ").title()
            st.caption(f"{direction} {feat_label} (z={z:+.2f})")

# Radar charts
st.markdown("---")
st.subheader("🕸️ Radar comparatif")

profiler = _get("cluster_profiler")
if profiler and profiler.feature_names:
    radar_features = profiler.feature_names[:8]
    fig_radar = go.Figure()

    for cid, profile in sorted(profiles.items()):
        name = names.get(cid, f"Profil {cid}")
        values = [profile["z_scores"].get(f, 0) for f in radar_features]
        values.append(values[0])  # Fermer le polygone
        labels_r = [f.replace("_", " ").title()[:15] for f in radar_features]
        labels_r.append(labels_r[0])

        fig_radar.add_trace(go.Scatterpolar(r=values, theta=labels_r, fill='toself', name=name, opacity=0.6))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
        title="Comparaison des profils (Z-scores)",
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Attribution par élève
st.markdown("---")
st.subheader("👤 Attribution individuelle")
st.caption(
    "⚠️ Ce tableau associe chaque élève au groupe le plus proche dans l'espace "
    "des features. Cette association est probabiliste et provisoire — elle ne "
    "définit pas l'élève."
)

df_attrib = df_feat.copy()
df_attrib["Profil"] = [names.get(l, f"Profil {l}") for l in labels]
display_cols_c = [c for c in ["nom", "prenom", "classe", "Profil"] if c in df_attrib.columns]
if target_reg in df_attrib.columns:
    display_cols_c.append(target_reg)

filter_profil = st.multiselect("Filtrer par profil", list(names.values()), default=list(names.values()))
df_filtered = df_attrib[df_attrib["Profil"].isin(filter_profil)]
st.dataframe(df_filtered[display_cols_c].reset_index(drop=True), use_container_width=True)
