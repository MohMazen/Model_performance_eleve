"""
Génération de rapports et visualisations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from config import REPORT_FILE

logger = logging.getLogger(__name__)

def generer_visualisations(df):
    """Génère des graphiques de base."""
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution de la moyenne
    plt.figure(figsize=(10, 6))
    sns.histplot(df['note_moyenne'], kde=True, color='teal')
    plt.title("Distribution des Notes Moyennes")
    plt.xlabel("Moyenne / 20")
    plt.show()

def generer_rapport_markdown(df, metrics_reg, metrics_clf, path=REPORT_FILE):
    """Génère le rapport final en Markdown."""
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

## 2. PERFORMANCE DES MODÈLES
### Régression (Prédiction de la note)
- **Modèle** : XGBoost Tuned
- **R² Score** : {metrics_reg.get('r2', 'N/A'):.3f}
- **MAE** : {metrics_reg.get('mae', 'N/A'):.3f}

### Classification (Prédiction de la réussite)
- **Modèle** : Random Forest Tuned
- **Accuracy** : {metrics_clf.get('accuracy', 'N/A'):.1f}%
- **F1-Score** : {metrics_clf.get('f1', 'N/A'):.3f}

## 3. ANALYSE DE L'EXPLICABILITÉ (SHAP)
*L'analyse SHAP a identifié les facteurs les plus influents sur la performance individuelle.*
- *Voir graphiques générés pour le détail des impacts.*

## 4. RECOMMANDATIONS
1. Focus sur les élèves avec un faible **Score d'Équilibre**.
2. Intervention préventive pour les tensions détectées via l'interaction **Stress x Absences**.

---
*Généré par le Système d'Analyse Scolaire v2.0*
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(contenu)
    return path
