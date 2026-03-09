"""
Analyse de l'explicabilité avec SHAP.
"""
import shap
import matplotlib.pyplot as plt
import logging
import numpy as np

logger = logging.getLogger(__name__)

def generate_shap_analysis(model_pipeline, X_sample):
    """
    Génère et affiche le Summary Plot de SHAP pour le modèle donné.
    """
    logger.info("Calcul des valeurs SHAP...")
    
    # 1. Pipeline extraction
    preprocessor = model_pipeline.named_steps['pre']
    model = model_pipeline.named_steps['model']
    
    # 2. Transformation des données
    X_transformed = preprocessor.transform(X_sample)
    
    # 3. Noms des colonnes après encodage
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    num_feature_names = preprocessor.transformers_[0][2]
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    # 4. SHAP Explainer
    try:
        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed)
        
        # Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, feature_names=all_feature_names, show=False)
        plt.title("SHAP Global Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return shap_values
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse SHAP : {e}")
        return None
