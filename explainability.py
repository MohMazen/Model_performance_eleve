import shap
import matplotlib.pyplot as plt
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Dictionnaire de traduction des variables
TRADUCTIONS = {
    'heures_devoirs': 'Heures de devoirs',
    'motivation': 'Motivation (1-10)',
    'heures_sommeil': 'Heures de sommeil',
    'stress': 'Niveau de stress',
    'absences': "Nombre d'absences",
    'temps_ecrans': "Temps d'écrans",
    'score_equilibre': "Équilibre Vie/Études",
    'stress_absences': 'Interaction Stress/Absences',
    'motivation_travail': 'Interaction Motivation/Travail',
    'genre_M': 'Genre : Garçon',
    'genre_F': 'Genre : Fille',
    'sport_Oui': 'Pratique du sport',
    'sport_Non': 'Pas de sport'
}

def generate_shap_analysis(model_pipeline, X_sample):
    """
    Génère une analyse d'importance des facteurs en français.
    """
    logger.info("Calcul des valeurs SHAP...")
    
    preprocessor = model_pipeline.named_steps['pre']
    model = model_pipeline.named_steps['model']
    
    X_transformed = preprocessor.transform(X_sample)
    
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    num_feature_names = preprocessor.transformers_[0][2]
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    # Application de la traduction
    feature_names_fr = [TRADUCTIONS.get(name, name) for name in all_feature_names]
    
    try:
        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed)
        
        # Graphique en barres (plus lisible pour les non-experts)
        plt.figure(figsize=(10, 8))
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
        plt.show()
        
        return shap_values
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse SHAP : {e}")
        return None
