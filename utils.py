import pandas as pd
import numpy as np
def get_data(train_values, test_values, train_labels):
    train_data = pd.read_csv(train_values, dtype={
    'geo_level_1_id':np.uint8, 
    'geo_level_2_id':np.uint16, 
    'geo_level_3_id': np.uint16,
    'count_floors_pre_eq_max': np.uint8,
    'age': np.uint16,
    'area_percentage': np.uint8,
    'height_percentage': np.uint8,
    'has_superstructure_adobe_mud'  :             np.uint8,
    'has_superstructure_mud_mortar_stone':        np.uint8,
    'has_superstructure_stone_flag'       :       np.uint8,
    'has_superstructure_cement_mortar_stone':     np.uint8,
    'has_superstructure_mud_mortar_brick'    :    np.uint8,
    'has_superstructure_cement_mortar_brick'  :   np.uint8,
    'has_superstructure_timber'                :  np.uint8,
    'has_superstructure_bamboo'                 : np.uint8,
    'has_superstructure_rc_non_engineered'      : np.uint8,
    'has_superstructure_rc_engineered'          : np.uint8,
    'has_superstructure_other'                  : np.uint8,
    'has_secondary_use'                          : np.uint8,
    'has_secondary_use_agriculture'              : np.uint8,
    'has_secondary_use_hotel'                    : np.uint8,
    'has_secondary_use_rental'                   : np.uint8,
    'has_secondary_use_institution'              : np.uint8,
    'has_secondary_use_school'                   : np.uint8,
    'has_secondary_use_industry'                 : np.uint8,
    'has_secondary_use_health_post'              : np.uint8,
    'has_secondary_use_gov_office'               : np.uint8,
    'has_secondary_use_use_police'               : np.uint8,
    'has_secondary_use_other'                    : np.uint8,
    
    })
    test_data = pd.read_csv(test_values)
    y = pd.read_csv(train_labels)
    full_training_data = train_data.merge(y,on='building_id')
    train_ids= full_training_data.pop('building_id')
    test_ids = test_data.pop('building_id')
    
    return full_training_data, test_data, train_ids, test_ids
    