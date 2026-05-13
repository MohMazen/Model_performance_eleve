import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Dictionnaire de traduction des variables (aligné Questionnaire.html)
TRADUCTIONS = {
    'heures_etude_soir': "Heures d'étude (soir)",
    'motivation_maths': "Motivation pour les Maths",
    'motivation_francais': "Motivation pour le Français",
    'motivation_hgemc': "Motivation pour HG-EMC",
    'motivation_anglais': "Motivation pour l'Anglais",
    'motivation_arabe': "Motivation pour l'Arabe",
    'motivation_sciences': "Motivation pour les Sciences",
    'motivation_eps': "Motivation pour l'EPS",
    'motivation_enseignement_scientifique': "Motivation pour l'enseignement scientifique",
    'motivation_famille': "Motivation : famille",
    'motivation_recompenses': "Motivation : récompenses",
    'heures_sommeil': "Heures de sommeil",
    'stress_personnel': "Niveau de stress personnel",
    'stress1': "Stress (item 1)",
    'stress2': "Stress (item 2)",
    'perseverance': "Persévérance",
    'grit1': "Grit (item 1)",
    'grit2': "Grit (item 2)",
    'grit3': "Grit (item 3)",
    'absences': "Nombre d'absences",
    'heures_jeux_video': "Jeux Vidéo",
    'score_equilibre': "Équilibre Vie/Études",
    'stress_total': "Stress Total",
    'genre_m': "Genre : Garçon",
    'genre_f': "Genre : Fille",
    'activite_sportive_oui': "Pratique du sport",
    'activite_sportive_non': "Pas de sport",
    'temps_ecrans_total': "Temps d'écran total",
    'ratio_etude_ecrans': "Ratio Étude/Écrans",
    'indice_motivation': "Indice de Motivation",
    'organisation': "Capacité d'organisation",
    'confiance_soi': "Confiance en soi",
    'estime_soi': "Estime de soi",
    'qualite_tuteur': "Qualité du tuteur",
    'qualite_mentor': "Qualité du mentor",
    'temps_libre': "Temps libre quotidien",
    'heures_garde_freres_soeurs': "Garde frères/sœurs (h/sem)",
    # Compatibilité ascendante : ancien nom `interet_*` toujours toléré.
    'interet_maths': "Motivation pour les Maths",
    'interet_francais': "Motivation pour le Français",
}


def generate_shap_failure_analysis(model_pipeline: Any, X_sample: pd.DataFrame,
                                   y_true: Optional[pd.Series] = None,
                                   seuil: float = 10.0,
                                   buf: Optional[Any] = None,
                                   use_predictions: bool = True) -> Optional[Any]:
    """
    Génère une analyse SHAP spécifique aux élèves prédits en échec.

    Par défaut (use_predictions=True), filtre les élèves selon la PRÉDICTION
    du modèle, pas la vérité terrain — c'est ce qu'on veut pour expliquer
    le comportement du modèle. Si use_predictions=False, le filtrage se fait
    sur y_true (ancien comportement, gardé pour compatibilité).

    Le fallback KernelExplainer utilise désormais un échantillon de
    background représentatif (jusqu'à 20 lignes via shap.sample) au lieu
    d'un seul point qui produit des valeurs SHAP invalides.
    """
    logger.info("Calcul des valeurs SHAP pour les facteurs d'échec...")

    try:
        # Sélection des élèves en échec : prédiction par défaut, sinon y_true.
        if use_predictions:
            try:
                preds = model_pipeline.predict(X_sample)
                fail_mask = preds < seuil
            except Exception as e_pred:
                logger.warning(f"Impossible de prédire pour filtrer (échec={e_pred}). "
                               "Fallback sur y_true.")
                if y_true is None:
                    logger.warning("Pas de y_true fourni, abandon.")
                    return None
                fail_mask = (y_true < seuil).values
        else:
            if y_true is None:
                logger.warning("y_true requis quand use_predictions=False.")
                return None
            fail_mask = (y_true < seuil).values

        if not np.any(fail_mask):
            logger.warning("Aucun élève prédit en échec dans l'échantillon.")
            return None

        X_fail = X_sample.loc[fail_mask] if hasattr(X_sample, 'loc') else X_sample[fail_mask]

        preprocessor = model_pipeline.named_steps['pre']
        model = model_pipeline.named_steps['model']

        # Preprocessing — on utilise X_sample complet pour le background et
        # X_fail pour les SHAP values, afin que l'explainer ait une référence
        # représentative de la distribution complète.
        X_bg_transformed = preprocessor.transform(X_sample)
        X_fail_transformed = preprocessor.transform(X_fail)

        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        num_feature_names = preprocessor.transformers_[0][2]
        all_feature_names = list(num_feature_names) + list(cat_feature_names)

        if 'select' in model_pipeline.named_steps:
            selector = model_pipeline.named_steps['select']
            X_bg_transformed = selector.transform(X_bg_transformed)
            X_fail_transformed = selector.transform(X_fail_transformed)
            selected_mask = selector.get_support()
            all_feature_names = [name for name, sel in zip(all_feature_names, selected_mask) if sel]

        feature_names_fr = [TRADUCTIONS.get(name, name) for name in all_feature_names]

        for arr_name in ('X_bg_transformed', 'X_fail_transformed'):
            arr = locals()[arr_name]
            if hasattr(arr, 'toarray'):
                locals()[arr_name] = arr.toarray()
            elif hasattr(arr, 'todense'):
                locals()[arr_name] = np.asarray(arr.todense())
        # Reconstruction explicite après modification de locals().
        if hasattr(X_bg_transformed, 'toarray'):
            X_bg_transformed = X_bg_transformed.toarray()
        if hasattr(X_fail_transformed, 'toarray'):
            X_fail_transformed = X_fail_transformed.toarray()

        try:
            model_type = type(model).__name__
            if model_type in ('XGBRegressor', 'XGBClassifier', 'RandomForestRegressor', 'RandomForestClassifier'):
                explainer = shap.TreeExplainer(model)
            else:
                # Background représentatif : k-means summary si shap.sample dispo,
                # sinon échantillon aléatoire jusqu'à 20 lignes.
                bg_size = min(20, X_bg_transformed.shape[0])
                if hasattr(shap, 'sample'):
                    background = shap.sample(X_bg_transformed, bg_size, random_state=42)
                else:
                    background = X_bg_transformed[:bg_size]
                explainer = shap.KernelExplainer(model.predict, background)
        except Exception as e_explainer:
            logger.warning(f"Explainer SHAP (échec) échoué : {e_explainer}. Fallback KernelExplainer.")
            bg_size = min(20, X_bg_transformed.shape[0])
            background = X_bg_transformed[:bg_size] if bg_size > 0 else X_bg_transformed
            explainer = shap.KernelExplainer(model.predict, background)

        shap_values = explainer(X_fail_transformed)
        # Pour rester compatible avec shap.summary_plot ci-dessous.
        X_transformed = X_fail_transformed

        # Graphique rouge pour l'échec
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names_fr,
            plot_type="bar",
            color="#E74C3C",  # Rouge
            show=False
        )

        plt.title("Facteurs corrélés à l'échec scolaire — corrélation ≠ causalité", fontsize=13, pad=20)
        plt.xlabel("Corrélation SHAP moyenne avec le risque d'échec", fontsize=11)
        plt.ylabel("Facteurs analysés", fontsize=12)
        plt.tight_layout()

        if buf is not None:
            plt.savefig(buf, format='png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        logger.info("Analyse SHAP d'échec terminée avec succès.")
        return shap_values
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse SHAP d'échec : {e}", exc_info=True)
        plt.close('all')
        return None


def generate_shap_analysis(model_pipeline: Any, X_sample: pd.DataFrame, buf: Optional[Any] = None) -> Optional[Any]:
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
                # Pour SVM, MLP et autres : KernelExplainer avec un background
                # représentatif (échantillon, jamais 1 seul point).
                bg_size = min(20, X_transformed.shape[0])
                if bg_size < 5:
                    logger.warning(f"Background SHAP trop petit ({bg_size}). Valeurs peu fiables.")
                if hasattr(shap, 'sample') and X_transformed.shape[0] > bg_size:
                    background = shap.sample(X_transformed, bg_size, random_state=42)
                else:
                    background = X_transformed[:bg_size] if X_transformed.shape[0] > 0 else X_transformed
                explainer = shap.KernelExplainer(model.predict, background)
        except Exception as e_explainer:
            logger.warning(f"Explainer SHAP échoué : {e_explainer}. Fallback sur KernelExplainer.")
            # Fallback corrigé : on prend toujours un background représentatif,
            # JAMAIS un seul point (qui produirait des SHAP values invalides).
            bg_size = min(20, X_transformed.shape[0])
            background = X_transformed[:bg_size] if bg_size > 0 else X_transformed
            explainer = shap.KernelExplainer(model.predict, background)

        shap_values = explainer(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names_fr,
            plot_type="bar",
            show=False
        )

        plt.title("Corrélation des facteurs avec la réussite — corrélation ≠ causalité", fontsize=13, pad=20)
        plt.xlabel("Corrélation SHAP moyenne avec la note de l'élève", fontsize=11)
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


def get_individual_shap_values(model_pipeline: Any, X_sample: pd.DataFrame,
                                student_index: int = 0) -> Optional[dict]:
    """
    Extrait les SHAP values pour un élève spécifique.

    Returns
    -------
    dict avec clés : feature_names, shap_values (1D array), base_value
    """
    try:
        preprocessor = model_pipeline.named_steps['pre']
        model = model_pipeline.named_steps['model']

        X_transformed = preprocessor.transform(X_sample)
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        num_feature_names = preprocessor.transformers_[0][2]
        all_feature_names = list(num_feature_names) + list(cat_feature_names)

        if 'select' in model_pipeline.named_steps:
            selector = model_pipeline.named_steps['select']
            X_transformed = selector.transform(X_transformed)
            selected_mask = selector.get_support()
            all_feature_names = [n for n, s in zip(all_feature_names, selected_mask) if s]

        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        elif hasattr(X_transformed, 'todense'):
            X_transformed = np.asarray(X_transformed.todense())

        model_type = type(model).__name__
        if model_type in ('XGBRegressor', 'XGBClassifier', 'RandomForestRegressor', 'RandomForestClassifier'):
            explainer = shap.TreeExplainer(model)
        else:
            bg_size = min(20, X_transformed.shape[0])
            if hasattr(shap, 'sample') and X_transformed.shape[0] > bg_size:
                background = shap.sample(X_transformed, bg_size, random_state=42)
            else:
                background = X_transformed[:bg_size]
            explainer = shap.KernelExplainer(model.predict, background)

        shap_vals = explainer(X_transformed)

        idx = min(student_index, len(X_sample) - 1)
        return {
            "feature_names": all_feature_names,
            "shap_values": shap_vals.values[idx],
            "base_value": float(shap_vals.base_values[idx]) if hasattr(shap_vals, 'base_values') else 0.0,
        }
    except Exception as e:
        logger.error(f"Erreur SHAP individuel : {e}", exc_info=True)
        return None


def get_top_actionable_factors(shap_result: dict, top_n: int = 10) -> list:
    """
    Retourne les top-N facteurs triés par impact SHAP absolu.

    Returns list of (feature_name, shap_value) tuples.
    """
    if shap_result is None:
        return []
    names = shap_result["feature_names"]
    values = shap_result["shap_values"]
    pairs = list(zip(names, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]
