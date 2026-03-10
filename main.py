"""
Point d'entrée principal de l'application d'analyse.
Orchestre le chargement, le preprocessing, l'entraînement et le rapport.
"""
import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score

from src.config import DATA_FILE, LOG_FILE, TARGET_REG, TARGET_CLF, COLS_TO_DROP
from src.data_utils import generer_donnees_synthetiques, nettoyer_donnees, charger_donnees
from src.features import add_advanced_features, prenttoyer_horaires
from src.models import ModelManager
from src.explainability import generate_shap_analysis
from src.reporting import generer_rapport_markdown

# Configuration du Logging
os.makedirs('outputs', exist_ok=True)
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
    os.makedirs('data', exist_ok=True)
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
    X = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns] + [TARGET_REG, TARGET_CLF])
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]

    # 4. Séparation train/test pour une évaluation non biaisée (Bug 1 fix)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # 5. Modélisation
    mm = ModelManager()
    mm.prepare_pipeline(X_train)

    model_reg = mm.train_regression(X_train, y_reg_train)
    model_clf = mm.train_classification(X_train, y_clf_train)

    # Évaluation sur les données de TEST (jamais vues à l'entraînement)
    y_pred_reg = model_reg.predict(X_test)
    y_pred_clf = model_clf.predict(X_test)

    metrics_reg = {
        'r2': r2_score(y_reg_test, y_pred_reg),
        'mae': mean_absolute_error(y_reg_test, y_pred_reg),
        'rmse': np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    }
    metrics_clf = {
        'accuracy': accuracy_score(y_clf_test, y_pred_clf) * 100,
        'f1': f1_score(y_clf_test, y_pred_clf, zero_division=0),
        'precision': precision_score(y_clf_test, y_pred_clf, zero_division=0),
        'recall': recall_score(y_clf_test, y_pred_clf, zero_division=0)
    }

    logger.info(f"Métriques régression : R²={metrics_reg['r2']:.4f}, MAE={metrics_reg['mae']:.4f}")
    logger.info(f"Métriques classification : Accuracy={metrics_clf['accuracy']:.2f}%, F1={metrics_clf['f1']:.4f}")

    # 6. Explicabilité
    sample_size = min(50, len(X_test))
    generate_shap_analysis(model_reg, X_test.iloc[:sample_size])

    # 7. Sauvegarde et Rapport
    mm.save_models()
    generer_rapport_markdown(df, metrics_reg, metrics_clf)

    logger.info("Workflow terminé avec succès.")


if __name__ == "__main__":
    main()
