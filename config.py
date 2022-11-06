train_datapath = 'data/train_dataset_train.csv'
test_datapath = 'data/test_dataset_test.csv'
model_path = 'models/'
solutions_path = 'solutions'

not_fit_cols = ['.geo', 'area', 'crop', 'id']

drops = ['nd_mean_2021-05-09', 'nd_mean_2021-04-22', 'nd_mean_2021-04-28', 'nd_mean_2021-05-03', 'nd_mean_2021-07-26',
        'nd_mean_2021-06-20']

my_params = [{'reg_lambda': 1.117671338442825,
 'learning_rate': 0.08048364963824424,
 'border_count': 194,
 'random_strength': 1.0824722845011207e-08,
 'bagging_temperature': 0.2798565614842792},
              {
 'reg_lambda': 2.5021774058932618,
 'learning_rate': 0.21990454345268695,
 'border_count': 250,
 'random_strength': 6.785242421438612e-08,
 'bagging_temperature': 0.17135452099556991}, 
              {
 'reg_lambda': 5.387205622118943,
 'learning_rate': 0.18600803957148327,
 'border_count': 207,
 'random_strength': 8.03831546771057e-07,
 'bagging_temperature': 0.4126147179623844}]
