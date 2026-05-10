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
from src.features import add_advanced_features, nettoyer_horaires
from src.models import ModelManager
from src.explainability import generate_shap_analysis, generate_shap_failure_analysis
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
    regenerate = False
    if os.path.exists(DATA_FILE):
        df = charger_donnees(DATA_FILE)
        # Vérifier la compatibilité du schéma (les colonnes attendues existent-elles ?)
        required_cols = ['activite_sportive', 'heures_etude_soir', 'heures_jeux_video',
                         'heures_sommeil', 'stress_personnel', 'perseverance', 'heures_reseaux_sociaux',
                         'heures_streaming', 'specialite1ere_1_nom', 'note_specialite1ere_1']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Schéma CSV incompatible (colonnes manquantes : {missing}). Régénération...")
            regenerate = True
    else:
        regenerate = True

    if regenerate:
        df = generer_donnees_synthetiques(500)
        df.to_csv(DATA_FILE, sep=';', index=False, encoding='utf-8-sig')
        logger.info("Nouveau fichier CSV généré avec le schéma à jour.")

    # 2. Preprocessing & Feature Engineering
    df = nettoyer_donnees(df)
    df = nettoyer_horaires(df)
    df = add_advanced_features(df)

    # 3. Préparation des données
    X = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns] + [TARGET_REG, TARGET_CLF])
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]

    # 4. Séparation train/test pour une évaluation non biaisée (Bug 1 fix)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # 5. Modélisation : train_all entraîne les 6 modèles (XGBoost, RF, MLP×2,
    #    SVM×2) et sélectionne automatiquement le meilleur en régression
    #    et en classification via les scores CV.
    mm = ModelManager()
    mm.prepare_pipeline(X_train)
    mm.feature_columns = list(X_train.columns)

    cv_scores = mm.train_all(X_train, y_reg_train, y_clf_train, include_svm=True)
    logger.info(f"Scores CV de tous les modèles : {cv_scores}")

    # Modèles sélectionnés (best_overall_*) pour les usages downstream
    model_reg = mm.best_overall_reg
    model_clf = mm.best_overall_clf

    # 6. Explicabilité — basée sur le meilleur modèle de régression.
    sample_size = min(50, len(X_test))
    generate_shap_analysis(model_reg, X_test.iloc[:sample_size])
    # Filtrage selon les PRÉDICTIONS du modèle (correctif point 5).
    generate_shap_failure_analysis(model_reg, X_test.iloc[:sample_size],
                                    y_true=y_reg_test.iloc[:sample_size],
                                    use_predictions=True)

    # 7. Évaluation des 6 modèles sur les données de TEST.
    def _metrics_reg(y_true, y_pred):
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        }

    def _metrics_clf(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
        }

    metrics_reg = _metrics_reg(y_reg_test, mm.best_model_reg.predict(X_test))
    metrics_clf = _metrics_clf(y_clf_test, mm.best_model_clf.predict(X_test))
    metrics_nn_reg = _metrics_reg(y_reg_test, mm.best_model_nn_reg.predict(X_test))
    metrics_nn_clf = _metrics_clf(y_clf_test, mm.best_model_nn_clf.predict(X_test))
    metrics_svm_reg = _metrics_reg(y_reg_test, mm.best_model_svm_reg.predict(X_test))
    metrics_svm_clf = _metrics_clf(y_clf_test, mm.best_model_svm_clf.predict(X_test))

    logger.info(f"XGBoost régression  : R²={metrics_reg['r2']:.4f}, MAE={metrics_reg['mae']:.4f}")
    logger.info(f"RF classification   : Acc={metrics_clf['accuracy']:.2f}%, F1={metrics_clf['f1']:.4f}")
    logger.info(f"MLP régression      : R²={metrics_nn_reg['r2']:.4f}, MAE={metrics_nn_reg['mae']:.4f}")
    logger.info(f"MLP classification  : Acc={metrics_nn_clf['accuracy']:.2f}%, F1={metrics_nn_clf['f1']:.4f}")
    logger.info(f"SVR régression      : R²={metrics_svm_reg['r2']:.4f}, MAE={metrics_svm_reg['mae']:.4f}")
    logger.info(f"SVC classification  : Acc={metrics_svm_clf['accuracy']:.2f}%, F1={metrics_svm_clf['f1']:.4f}")

    # 8. Audit fairness sur le test set (point 7).
    try:
        from src.fairness import audit_fairness, format_fairness_report
        df_test_for_audit = X_test.copy()
        df_test_for_audit[TARGET_REG] = y_reg_test.values
        df_test_for_audit[TARGET_CLF] = y_clf_test.values
        df_test_for_audit['y_pred_clf'] = mm.best_model_clf.predict(X_test)
        df_test_for_audit['y_pred_reg'] = mm.best_model_reg.predict(X_test)

        for sensitive in ('genre', 'classe'):
            if sensitive in df_test_for_audit.columns:
                audit = audit_fairness(df_test_for_audit, sensitive_col=sensitive,
                                       y_true_clf=TARGET_CLF, y_pred_clf='y_pred_clf')
                logger.info(f"Audit fairness ({sensitive}) :\n"
                            + format_fairness_report(audit))
    except ImportError:
        logger.warning("Module fairness indisponible, audit ignoré.")

    # 9. Sauvegarde et Rapport
    mm.save_models()
    generer_rapport_markdown(df, metrics_reg, metrics_clf,
                             metrics_nn_reg=metrics_nn_reg,
                             metrics_nn_clf=metrics_nn_clf)

    logger.info("Workflow terminé avec succès.")


if __name__ == "__main__":
    main()
