"""
Génération de rapports et visualisations.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.config import REPORT_FILE

logger = logging.getLogger(__name__)


def generer_visualisations(df, buf=None):
    """Génère des graphiques de base.

    Parameters
    ----------
    df : pd.DataFrame
    buf : io.BytesIO, optional
        Si fourni, sauvegarde le graphique dans ce buffer.
    """
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['note_moyenne'], kde=True, color='teal', ax=ax)
    ax.set_title("Distribution des Notes Moyennes")
    ax.set_xlabel("Moyenne / 20")

    if buf is not None:
        fig.savefig(buf, format='png', bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def generer_rapport_markdown(df, metrics_reg, metrics_clf, path=REPORT_FILE,
                             metrics_nn_reg=None, metrics_nn_clf=None, model_name=None):
    """Génère le rapport final en Markdown.

    Parameters
    ----------
    df : pd.DataFrame
    metrics_reg : dict
    metrics_clf : dict
    path : str or None
        Chemin de sauvegarde. Si None, le rapport n'est pas écrit sur disque
        (utile pour le dashboard Streamlit).
    metrics_nn_reg : dict, optional
        Métriques du réseau de neurones en régression.
    metrics_nn_clf : dict, optional
        Métriques du réseau de neurones en classification.
    model_name : str, optional
        Nom personnalisé du modèle.

    Returns
    -------
    str : Contenu du rapport Markdown.
    """
    logger.info(f"Génération du rapport : {path}")

    moyenne_gen = df['note_moyenne'].mean()
    nb_eleves = len(df)
    nb_echec = (df['note_moyenne'] < 10).sum()

    contenu = f"""# RAPPORT D'ANALYSE SCOLAIRE AVANCÉ
============================================================

## 1. RÉSUMÉ EXÉCUTIF
- **Effectif** : {nb_eleves} élèves
- **Moyenne Générale** : {moyenne_gen:.2f}/20
- **Élèves sous le seuil (10/20)** : {nb_echec} ({nb_echec/nb_eleves*100:.1f}%)
"""
    if model_name:
        contenu += f"- **Nom du modèle** : {model_name}\n"

    contenu += f"""
## 2. PERFORMANCE DES MODÈLES
### Régression (Prédiction de la note)
- **Modèle** : XGBoost Tuned
- **R² Score** : {metrics_reg.get('r2', 'N/A'):.3f}
- **MAE** : {metrics_reg.get('mae', 'N/A'):.3f}
- **RMSE** : {metrics_reg.get('rmse', 'N/A'):.3f}

### Classification (Prédiction de la réussite)
- **Modèle** : Random Forest Tuned (class_weight=balanced)
- **Accuracy** : {metrics_clf.get('accuracy', 'N/A'):.1f}%
- **F1-Score** : {metrics_clf.get('f1', 'N/A'):.3f}
- **Precision** : {metrics_clf.get('precision', 'N/A'):.3f}
- **Recall** : {metrics_clf.get('recall', 'N/A'):.3f}
"""

    if metrics_nn_reg is not None:
        contenu += f"""
### Réseau de Neurones – Régression
- **Modèle** : MLPRegressor Tuned
- **R² Score** : {metrics_nn_reg.get('r2', 'N/A'):.3f}
- **MAE** : {metrics_nn_reg.get('mae', 'N/A'):.3f}
- **RMSE** : {metrics_nn_reg.get('rmse', 'N/A'):.3f}
"""

    if metrics_nn_clf is not None:
        contenu += f"""
### Réseau de Neurones – Classification
- **Modèle** : MLPClassifier Tuned
- **Accuracy** : {metrics_nn_clf.get('accuracy', 'N/A'):.1f}%
- **F1-Score** : {metrics_nn_clf.get('f1', 'N/A'):.3f}
- **Precision** : {metrics_nn_clf.get('precision', 'N/A'):.3f}
- **Recall** : {metrics_nn_clf.get('recall', 'N/A'):.3f}
"""

    contenu += """
## 3. ANALYSE DE L'EXPLICABILITÉ (SHAP)
*L'analyse SHAP a identifié les facteurs les plus influents sur la performance individuelle.*
- *Voir graphiques générés pour le détail des impacts.*

## 4. RECOMMANDATIONS
1. Focus sur les élèves avec un faible **Score d'Équilibre**.
2. Intervention préventive pour les tensions détectées via l'interaction **Stress x Absences**.

---
*Généré par le Système d'Analyse Scolaire v2.0*
"""
    if path is not None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contenu)
    return contenu
