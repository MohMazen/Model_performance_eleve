"""
Génération de rapports et visualisations.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
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
                             metrics_nn_reg=None, metrics_nn_clf=None,
                             metrics_svm_reg=None, metrics_svm_clf=None,
                             selected_features=None, model_name=None,
                             target_col=None, threshold=None, grade_cols=None):
    """
    Génère un rapport final extrêmement détaillé incluant les statistiques
    de cohorte, le prétraitement et le benchmark des modèles.
    """
    logger.info(f"Génération du rapport haute définition : {path}")

    from src.config import GRADE_COLUMNS, TARGET_REG, SEUIL_REUSSITE
    
    # Utiliser les valeurs fournies ou les valeurs par défaut de la configuration
    target_col = target_col if target_col is not None else TARGET_REG
    threshold = threshold if threshold is not None else SEUIL_REUSSITE
    grade_cols = grade_cols if grade_cols is not None else GRADE_COLUMNS
    
    # 1. Statistiques Globales
    nb_eleves = len(df)
    moyenne_gen = df[target_col].mean() if target_col in df.columns else 0
    nb_reussite = (df[target_col] >= threshold).sum() if target_col in df.columns else 0
    nb_echec = nb_eleves - nb_reussite
    taux_reussite = (nb_reussite / nb_eleves) * 100 if nb_eleves > 0 else 0

    # 2. Statistiques par matière
    stats_matieres = "| Matière | Moyenne | Écart-Type | Min | Max |\n| :--- | :---: | :---: | :---: | :---: |\n"
    for col in grade_cols:
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            mini = df[col].min()
            maxi = df[col].max()
            label = str(col).replace('note_', '').capitalize()
            stats_matieres += f"| {label} | {m:.2f} | {s:.2f} | {mini:.1f} | {maxi:.1f} |\n"

    # 3. Préparation du contenu
    contenu = f"""# 🎓 RAPPORT D'ANALYSE SCOLAIRE DÉTAILLÉ
============================================================

## 📊 1. SYNTHÈSE DE LA COHORTE
Ce rapport présente les résultats de l'analyse effectuée sur une cohorte de **{nb_eleves} élèves**.

### Indicateurs Clés :
- **Moyenne Générale de la cohorte ({target_col})** : `{moyenne_gen:.2f}`
- **Taux de Réussite global (>= {threshold})** : `{taux_reussite:.1f}%`
- **Élèves au-dessus du seuil** : {nb_reussite}
- **Élèves en difficulté** : {nb_echec}

### Zoom par Matière / Variables :
{stats_matieres}

---

## 🛠️ 2. PRÉTRAITEMENT & SÉLECTION DE VARIABLES
Le processus d'analyse a inclus une phase de nettoyage des données et d'ingénierie des variables (création de scores d'équilibre, calcul du stress total, etc.).

"""
    if selected_features:
        contenu += f"""### Variables sélectionnées par l'IA :
L'algorithme de sélection a retenu les **{len(selected_features)} variables** les plus prédictives pour optimiser les performances :
`{", ".join(selected_features)}`

"""
    else:
        contenu += "*Note : Toutes les variables disponibles ont été utilisées pour l'entraînement.*\n\n"

    contenu += f"""
---

## 🚀 3. BENCHMARK DES MODÈLES (PERFORMANCE)
Nous avons comparé plusieurs architectures d'IA pour identifier la plus précise.

### 🔢 Régression (Prédiction de '{target_col}')
*Objectif : Estimer la valeur future de '{target_col}'.*

| Modèle | R² Score | MAE (Erreur moyenne) | RMSE |
| :--- | :---: | :---: | :---: |
| **XGBoost (Référence)** | {metrics_reg.get('r2', 0):.3f} | {metrics_reg.get('mae', 0):.3f} | {metrics_reg.get('rmse', 0):.3f} |
"""
    if metrics_nn_reg:
        contenu += f"| **Réseau de Neurones** | {metrics_nn_reg.get('r2', 0):.3f} | {metrics_nn_reg.get('mae', 0):.3f} | {metrics_nn_reg.get('rmse', 0):.3f} |\n"
    if metrics_svm_reg:
        contenu += f"| **SVM (SVR)** | {metrics_svm_reg.get('r2', 0):.3f} | {metrics_svm_reg.get('mae', 0):.3f} | {metrics_svm_reg.get('rmse', 0):.3f} |\n"

    contenu += f"""
### 🏆 Classification (Réussite vs Échec)
*Objectif : Prédire si l'élève franchira le seuil de {threshold}.*

| Modèle | Accuracy | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | {metrics_clf.get('accuracy', 0):.1f}% | {metrics_clf.get('f1', 0):.3f} | {metrics_clf.get('precision', 0):.3f} | {metrics_clf.get('recall', 0):.3f} |
"""
    if metrics_nn_clf:
        contenu += f"| **Réseau de Neurones** | {metrics_nn_clf.get('accuracy', 0):.1f}% | {metrics_nn_clf.get('f1', 0):.3f} | {metrics_nn_clf.get('precision', 0):.3f} | {metrics_nn_clf.get('recall', 0):.3f} |\n"
    if metrics_svm_clf:
        contenu += f"| **SVM (SVC)** | {metrics_svm_clf.get('accuracy', 0):.1f}% | {metrics_svm_clf.get('f1', 0):.3f} | {metrics_svm_clf.get('precision', 0):.3f} | {metrics_svm_clf.get('recall', 0):.3f} |\n"

    contenu += f"""
> **Modèle sélectionné** : {model_name if model_name else "Performances croisées"}

---

## 📈 4. ANALYSE D'EXPLICABILITÉ (SHAP)
L'intelligence artificielle indique que les facteurs suivants sont les leviers majeurs de performance pour cette cohorte :
1. Variables avec un grand impact positif (Facteurs de Réussite).
2. Variables augmentant le risque d'échec (Facteurs d'Échec).
3. L'importance de chaque facteur dépend de la SHAP summary générée.

---

## 💡 5. RECOMMANDATIONS ET ACTIONS
Sur la base des prédictions, voici les préconisations suggérées :
- **Monitorage actif** : Mettre en place un suivi particulier pour les élèves dont la probabilité de réussite est inférieure à 60% ou dont la note prédite est < 10.
- **Orientation Préventive** : Utiliser les prédictions par matière pour rediriger les élèves vers des séances de soutien spécifiques avant les examens.
- **Sensibilisation à l'équilibre** : Le score d'équilibre étant un facteur clé, des ateliers sur l'organisation du temps et le sommeil pourraient réduire les risques d'échec.

---
*Rapport généré par l'IA Antigravity v2.1 — Page générée le {datetime.now().strftime('%d/%m/%Y')}*
"""
    if path is not None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(contenu)
        except Exception as e:
            logger.error(f"Erreur lors de l'écriture du rapport sur disque : {e}")

    return contenu
