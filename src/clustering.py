"""
Clustering automatique des profils d'élèves.
Segmentation non-supervisée et profilage descriptif.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Noms de groupes comportementaux — descriptifs et neutres.
# Ces étiquettes décrivent des tendances statistiques agrégées, pas des
# caractéristiques individuelles permanentes. Elles ne doivent pas être
# communiquées directement aux élèves ou aux familles.
ARCHETYPE_RULES = [
    {"condition": lambda p: p.get("heures_etude_soir", 0) > 0.5 and p.get("stress_total", 0) > 0.5,
     "name": "📖 Groupe Étude Soutenue"},
    {"condition": lambda p: p.get("score_equilibre", 0) > 0.5 and p.get("indice_motivation", 0) > 0.5,
     "name": "⚖️ Groupe Équilibre Élevé"},
    {"condition": lambda p: p.get("temps_ecrans_total", 0) > 0.5 and p.get("heures_etude_soir", 0) < -0.3,
     "name": "📱 Groupe Temps d'Écran Élevé"},
    {"condition": lambda p: p.get("perseverance", 0) > 0.3 and p.get("confiance_soi", 0) > 0.3,
     "name": "💪 Groupe Persévérance & Confiance"},
    {"condition": lambda p: p.get("heures_sommeil", 0) < -0.3 and p.get("stress_total", 0) > 0.3,
     "name": "😴 Groupe Sommeil Court"},
]


class StudentProfiler:
    """Segmentation non-supervisée des profils d'élèves."""

    def __init__(self) -> None:
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Any] = None
        self.labels: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sélectionne les features numériques pertinentes pour le clustering."""
        preferred = [
            'heures_etude_soir', 'heures_sommeil', 'stress_total', 'score_equilibre',
            'temps_ecrans_total', 'ratio_etude_ecrans', 'indice_motivation',
            'perseverance', 'organisation', 'confiance_soi', 'estime_soi',
            'heures_jeux_video', 'heures_reseaux_sociaux', 'heures_streaming',
            'qualite_sommeil', 'heures_activite_physique', 'calme_maison',
        ]
        available = [c for c in preferred if c in df.columns]
        if len(available) < 3:
            available = df.select_dtypes(include=[np.number]).columns.tolist()[:15]
        self.feature_names = available
        return df[available].copy()

    def find_optimal_k(self, X_scaled: np.ndarray, k_range: Tuple[int, int] = (2, 8)) -> int:
        """Détermine le nombre optimal de clusters via silhouette score."""
        best_k, best_score = 2, -1
        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            logger.info(f"  k={k} → silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k
        logger.info(f"Nombre optimal de clusters : {best_k} (silhouette={best_score:.3f})")
        return best_k

    def fit_clusters(self, df: pd.DataFrame, method: str = 'kmeans',
                     n_clusters: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """Applique le clustering et retourne (labels, X_features_utilisées)."""
        logger.info(f"Clustering des profils d'élèves (méthode={method})…")
        X_raw = self._select_features(df)
        X_raw = X_raw.fillna(X_raw.median())

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        if method == 'kmeans':
            if n_clusters is None:
                n_clusters = self.find_optimal_k(X_scaled)
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            self.model = DBSCAN(eps=1.5, min_samples=5)
        else:
            raise ValueError(f"Méthode inconnue : {method}")

        self.labels = self.model.fit_predict(X_scaled)
        logger.info(f"Clustering terminé : {len(set(self.labels))} clusters trouvés.")
        return self.labels, X_raw

    def describe_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Génère un profil descriptif pour chaque cluster."""
        X_raw = df[self.feature_names].copy() if self.feature_names else df.select_dtypes(include=[np.number])
        profiles = {}
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            cluster_data = X_raw[mask]
            global_means = X_raw.mean()
            cluster_means = cluster_data.mean()
            # Z-scores normalisés par rapport à la moyenne globale
            global_stds = X_raw.std().replace(0, 1)
            z_scores = ((cluster_means - global_means) / global_stds).to_dict()

            # Caractéristiques dominantes (|z| > 0.3)
            dominant = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            dominant_features = [(k, round(v, 2)) for k, v in dominant[:5]]

            # Note moyenne si disponible
            note_moy = df.loc[mask, 'note_moyenne'].mean() if 'note_moyenne' in df.columns else None

            profiles[cluster_id] = {
                "size": int(mask.sum()),
                "pct": round(mask.sum() / len(labels) * 100, 1),
                "means": cluster_means.round(2).to_dict(),
                "z_scores": {k: round(v, 2) for k, v in z_scores.items()},
                "dominant_features": dominant_features,
                "note_moyenne": round(note_moy, 2) if note_moy is not None else None,
            }
        return profiles

    def generate_cluster_names(self, profiles: Dict[int, Dict]) -> Dict[int, str]:
        """Attribue des noms parlants aux clusters."""
        names = {}
        for cluster_id, profile in profiles.items():
            z = profile.get("z_scores", {})
            named = False
            for rule in ARCHETYPE_RULES:
                if rule["condition"](z):
                    names[cluster_id] = rule["name"]
                    named = True
                    break
            if not named:
                top_feat = profile.get("dominant_features", [])
                if top_feat:
                    feat_name = top_feat[0][0].replace("_", " ").title()
                    direction = "+" if top_feat[0][1] > 0 else "-"
                    names[cluster_id] = f"🔹 Profil {feat_name} ({direction})"
                else:
                    names[cluster_id] = f"🔹 Profil {cluster_id + 1}"
        return names

    def reduce_dimensions(self, df: pd.DataFrame, method: str = 'pca') -> np.ndarray:
        """Réduction dimensionnelle pour visualisation 2D."""
        X_raw = df[self.feature_names].copy() if self.feature_names else df.select_dtypes(include=[np.number])
        X_raw = X_raw.fillna(X_raw.median())
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_raw)
        else:
            X_scaled = StandardScaler().fit_transform(X_raw)

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1))

        return reducer.fit_transform(X_scaled)
