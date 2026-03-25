import pandas as pd
import numpy as np
from src.features import get_column_mapping, add_advanced_features, prenttoyer_horaires

def test_dynamic_features():
    # 1. Création d'un DataFrame simulant un CSV uploadé avec des noms différents
    data = {
        'Nom_Eleve': ['Alice', 'Bob'],
        'Temps_Sommeil': [8.0, 6.0],
        'Gaming_Hours': [1.0, 4.0],
        'Study_Time': [2.0, 1.0],
        'Sport_User': ['oui', 'non'],
        'Moyenne_Generale': [15.5, 9.5],
        'Bedtime': ['22h30', '23:45']
    }
    df = pd.DataFrame(data)
    
    print("Columns:", df.columns.tolist())
    
    # 2. Test de la détection automatique
    mapping = get_column_mapping(df.columns)
    print("Detected Mapping:", mapping)
    
    # Vérifications attendues
    assert mapping['sommeil'] == 'Temps_Sommeil'
    assert mapping['jeux_video'] == 'Gaming_Hours'
    assert mapping['etude'] == 'Study_Time'
    assert mapping['sport'] == 'Sport_User'
    assert mapping['note_moyenne'] == 'Moyenne_Generale'
    assert mapping['heure_coucher'] == 'Bedtime'
    
    # 3. Test du feature engineering
    df_h = prenttoyer_horaires(df, mapping=mapping)
    df_feat = add_advanced_features(df_h, mapping=mapping)
    
    print("New columns:", [c for c in df_feat.columns if c not in df.columns])
    
    # Vérifications des features créées
    assert 'score_equilibre' in df_feat.columns
    assert 'reussite' in df_feat.columns
    assert 'Bedtime_num' in df_feat.columns
    
    # Alice devrait réussir (15.5 >= 10)
    assert df_feat.loc[0, 'reussite'] == 1
    # Bob devrait échouer (9.5 < 10)
    assert df_feat.loc[1, 'reussite'] == 0
    
    print("✅ All dynamic tests passed!")

if __name__ == "__main__":
    test_dynamic_features()
