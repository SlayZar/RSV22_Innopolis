import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

from prep_func import prep, get_res_from_df, read_data

from config import train_datapath, test_datapath, not_fit_cols, my_params, drops, model_path


def scoring_script(scoring_df_path, train_df_path, output_path):
    df_train, df_test = read_data(train_df_path, scoring_df_path, not_fit_cols)
    full_train_poly = prep(df_train[['id', 'crop', '.geo']])
    full_test_poly = prep(df_test[['id', '.geo']])
    res_test = get_res_from_df(full_test_poly, full_train_poly)
    t_test = res_test.merge(df_test.merge(
    full_test_poly[['id', 'lat', 'lon']], on='id', how='left'), how='right', on ='id')
    sols = {}
    for i in range(7):
        sols[i] = []
    for i in os.listdir(model_path):
        cb = CatBoostClassifier()
        cb.load_model(os.path.join(model_path, i))
        for i in cb.classes_:
            sols[i].append(cb.predict_proba(t_test[cb.feature_names_])[:, i])
            
    for i in cb.classes_:
        t_test[i] = np.min(sols[i], axis=0)
    t_test['crop'] = t_test[cb.classes_].idxmax(axis=1)
    t_test = t_test.merge(df_train[['.geo', 'crop']], on = '.geo', how='left')
    
    t_test['crop'] = 1
    t_test.loc[t_test.crop_y.isnull(), 'crop'] = t_test.loc[t_test.crop_y.isnull(), 'crop_x'].astype(int)
    t_test.loc[~t_test.crop_y.isnull(), 'crop'] = t_test.loc[~t_test.crop_y.isnull(), 'crop_y'].astype(int)
    t_test[['id', 'crop']].to_csv(output_path, index=False)
