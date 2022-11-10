import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from prep_func import prep, get_res_from_df, read_data, get_extra_features
from config import train_datapath, test_datapath, not_fit_cols, my_params, drops, model_path, solution_path


def scoring_function(test_datapath, solution_path):
    df_train, df_test = read_data(train_datapath, test_datapath, not_fit_cols)
    df_test = get_extra_features(df_test)
    full_train_poly = prep(df_train[['id', 'crop', '.geo']])
    full_test_poly = prep(df_test[['id', '.geo']])
    res_test = get_res_from_df(full_test_poly, full_train_poly)
    t_test = res_test.merge(df_test.merge(
        full_test_poly[['id', 'lat', 'lon', 'kms', 'diff']], on='id', how='left'), how='right', on ='id')
    sols = {}
    for i in range(7):
        sols[i] = []
    for i, params in enumerate(my_params):
        cb = CatBoostClassifier(**params)
        cb.load_model(f'{model_path}/model2_{i}')
        for i in cb.classes_:
            sols[i].append(cb.predict_proba(t_test[cb.feature_names_])[:, i])
    for i in cb.classes_:
        t_test[i] = np.max(sols[i], axis=0)
    t_test['crop'] = t_test[cb.classes_].idxmax(axis=1)
    t_test.loc[(t_test[6]>0.5), 'crop'] = 6
    t_test['2_max'] = t_test[cb.classes_].apply(lambda x: sorted(x.to_list())[-2], axis=1)
    t_test.loc[(t_test[4]>0.2) & (t_test['2_max'] != t_test[4]) & (~t_test['crop_x'] != 4), 'crop'] = 4
    t_test[['id', 'crop']].to_csv(solution_path, index=False)
    print("Scoring result save in ", solution_path)

    
if __name__ == "__main__":
    scoring_function(test_datapath, solution_path)
