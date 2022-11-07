train_datapath = 'data/train_dataset_train.csv'
test_datapath = 'data/test_dataset_test.csv'
model_path = 'models/'
solution_path = 'best_sol.csv'

not_fit_cols = ['.geo', 'area', 'crop', 'id']

drops = ['nd_mean_2021-05-09', 'nd_mean_2021-04-22', 'nd_mean_2021-04-28', 'nd_mean_2021-05-03', 'nd_mean_2021-07-26',
        'nd_mean_2021-06-20', 'nd_mean_2021-04-23']

my_params = [{'n_estimators': 1286, 'random_state': 123,
                       'max_depth': 8, 'reg_lambda': 8.221298634988695, 'learning_rate': 0.0977489096578585,
                           'border_count': 53, 'random_strength': 0.0003742783361176472,
                           'bagging_temperature': 0.2574226602345471},
             {'n_estimators': 678, 'random_state': 123,
              'max_depth': 8, 'reg_lambda': 1.268413308513964, 'learning_rate': 0.2841620479722883,
              'border_count': 85, 'random_strength': 1.3776570352696018e-09, 'bagging_temperature': 0.379668040904092}]
