"""
Wrapper pour le nouveau système d'analyse modulaire (v2.0).
Ce script redirige vers Claude4_model/main_refactored.py pour profiter des dernières améliorations.
"""
import sys
import os

# Ajouter le répertoire actuel au path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # On importe main du module refactored
    from main_refactored import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    print("Assurez-vous d'avoir installé les dépendances : pip install shap streamlit joblib plotly")
    print("Ou lancez directement : py -m Claude4_model.main_refactored")