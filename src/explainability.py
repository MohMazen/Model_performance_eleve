import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Dictionnaire de traduction des variables
TRADUCTIONS = {
    'heures_etude_soir': 'Heures d\'étude (soir)',
    'interet_maths': 'Intérêt pour les Maths',
    'heures_sommeil': 'Heures de sommeil',
    'stress_1': 'Niveau de stress 1',
    'stress_2': 'Niveau de stress 2',
    'absences': "Nombre d'absences",
    'heures_jeux_video': "Jeux Vidéo",
    'score_equilibre': "Équilibre Vie/Études",
    'stress_total': 'Stress Total',
    'genre_m': 'Genre : Garçon',
    'genre_f': 'Genre : Fille',
    'activite_sportive_oui': 'Pratique du sport',
    'activite_sportive_non': 'Pas de sport'
}


def generate_shap_analysis(model_pipeline, X_sample, buf=None):
    """
    Génère une analyse d'importance des facteurs en français.
    Gère les pipelines avec ou sans étape de sélection de variables.
    Utilise TreeExplainer pour les modèles à base d'arbres (XGBoost, RF).
    """
    logger.info("Calcul des valeurs SHAP...")

    try:
        preprocessor = model_pipeline.named_steps['pre']
        model = model_pipeline.named_steps['model']

        # 1. Appliquer le preprocessing
        X_transformed = preprocessor.transform(X_sample)

        # 2. Récupérer les noms de toutes les features après preprocessing
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        num_feature_names = preprocessor.transformers_[0][2]
        all_feature_names = list(num_feature_names) + list(cat_feature_names)

        # 3. Si le pipeline contient un sélecteur, filtrer les données et les noms
        if 'select' in model_pipeline.named_steps:
            selector = model_pipeline.named_steps['select']
            X_transformed = selector.transform(X_transformed)
            selected_mask = selector.get_support()
            all_feature_names = [name for name, sel in zip(all_feature_names, selected_mask) if sel]

        # 4. Application de la traduction
        feature_names_fr = [TRADUCTIONS.get(name, name) for name in all_feature_names]

        # 5. Choisir le bon type d'explainer selon le modèle
        model_type = type(model).__name__
        logger.info(f"Type de modèle détecté pour SHAP : {model_type}")

        # Conversion en dense si nécessaire (sparse matrix → numpy array)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        elif hasattr(X_transformed, 'todense'):
            X_transformed = np.asarray(X_transformed.todense())

        try:
            # TreeExplainer pour les modèles arborescents
            if model_type in ('XGBRegressor', 'XGBClassifier', 'RandomForestRegressor', 'RandomForestClassifier'):
                explainer = shap.TreeExplainer(model)
            else:
                # Pour SVM, MLP et autres : KernelExplainer avec model.predict
                bg_size = min(20, X_transformed.shape[0])
                explainer = shap.KernelExplainer(model.predict, X_transformed[:bg_size])
        except Exception as e_explainer:
            logger.warning(f"Explainer spécialisé échoué ({e_explainer}), fallback sur Explainer générique.")
            # Fallback : passer model.predict (fonction callable) et non l'objet model
            explainer = shap.Explainer(model.predict, X_transformed)

        shap_values = explainer(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names_fr,
            plot_type="bar",
            show=False
        )

        plt.title("Importance des facteurs de réussite (Analyse IA)", fontsize=14, pad=20)
        plt.xlabel("Impact moyen sur la note de l'élève", fontsize=12)
        plt.ylabel("Facteurs analysés", fontsize=12)
        plt.tight_layout()

        if buf is not None:
            plt.savefig(buf, format='png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        logger.info("Analyse SHAP terminée avec succès.")
        return shap_values
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse SHAP : {e}", exc_info=True)
        plt.close('all')
        return None
