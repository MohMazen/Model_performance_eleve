import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.features import add_advanced_features
from src.models import ModelManager

# Target names from src.config
TARGET_CLF = 'reussite'

def test_dynamic_pipeline():
    # 1. Create a non-school related CSV
    np.random.seed(42)
    data = {
        'feature_alpha': np.random.rand(100),
        'feature_beta': np.random.rand(100),
        'my_custom_target': np.random.rand(100) * 20,
        'id_to_drop': range(100)
    }
    df = pd.DataFrame(data)
    
    print("Testing add_advanced_features on custom data...")
    # Should not crash and should return reasonably
    df_feat = add_advanced_features(df)
    print("Columns after feature engineering:", df_feat.columns.tolist())
    
    # 2. Test ModelManager with this data
    print("Testing ModelManager on custom data...")
    mm = ModelManager()
    
    # Simulate the logic in dashboard.py for uploaded data
    target_reg = 'my_custom_target'
    df_feat[TARGET_CLF] = (df_feat[target_reg] >= 10).astype(int)
    
    cols_drop = ['id_to_drop']
    X = df_feat.drop(columns=cols_drop + [target_reg, TARGET_CLF])
    y_reg = df_feat[target_reg]
    
    print(f"Features in X: {X.columns.tolist()}")
    mm.prepare_pipeline(X)
    
    # Minor training check (fastest model - Random Forest is faster to init in local)
    try:
        model_reg = mm.train_regression(X, y_reg)
        print("Model training successful!")
    except Exception as e:
        print(f"Model training failed: {e}")

if __name__ == "__main__":
    test_dynamic_pipeline()
