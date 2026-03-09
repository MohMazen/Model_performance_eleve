"""
Point d'entrée principal de l'application d'analyse.
 Orchestre le chargement, le preprocessing, l'entraînement et le rapport.
"""
import logging
import sys
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

# Ajout du répertoire parent au path pour les imports relatifs si exécuté directement
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_FILE, LOG_FILE, TARGET_REG, TARGET_CLF
from data_utils import generer_donnees_synthetiques, nettoyer_donnees, charger_donnees
from features import add_advanced_features, prenttoyer_horaires
from models import ModelManager
from explainability import generate_shap_analysis
from reporting import generer_visualisations, generer_rapport_markdown

# Configuration du Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnalyseurScolaire")

def main():
    logger.info("Démarrage du workflow d'analyse complet.")
    
    # 1. Chargement / Génération
    if not os.path.exists(DATA_FILE):
        df = generer_donnees_synthetiques(500)
        df.to_csv(DATA_FILE, sep=';', index=False, encoding='utf-8-sig')
    else:
        df = charger_donnees(DATA_FILE)
        
    # 2. Preprocessing & Feature Engineering
    df = nettoyer_donnees(df)
    df = prenttoyer_horaires(df)
    df = add_advanced_features(df)
    
    # 3. Préparation des données
    # Exclure les colonnes d'ID et les notes intermédiaires
    cols_to_drop = ['nom', 'prenom', 'note_francais', 'note_maths', 'note_lecture', 'heure_coucher', 'heure_lever']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns] + [TARGET_REG, TARGET_CLF])
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]
    
    # 4. Modélisation
    mm = ModelManager()
    mm.prepare_pipeline(X)
    
    # Entraînement
    model_reg = mm.train_regression(X, y_reg)
    model_clf = mm.train_classification(X, y_clf)
    
    # Évaluation (sur échantillon simple pour démo)
    y_pred_reg = model_reg.predict(X)
    y_pred_clf = model_clf.predict(X)
    
    metrics_reg = {
        'r2': r2_score(y_reg, y_pred_reg),
        'mae': mean_absolute_error(y_reg, y_pred_reg)
    }
    metrics_clf = {
        'accuracy': accuracy_score(y_clf, y_pred_clf) * 100,
        'f1': f1_score(y_clf, y_pred_clf)
    }
    
    # 5. Explicabilité
    generate_shap_analysis(model_reg, X.sample(50))
    
    # 6. Sauvegarde et Rapport
    mm.save_models()
    generer_rapport_markdown(df, metrics_reg, metrics_clf)
    
    logger.info("Workflow terminé avec succès.")

if __name__ == "__main__":
    main()
