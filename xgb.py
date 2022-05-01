import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import xgboost as xgb
from utils import ModelDumper

FEAT_SELECTIONS=['m', 'n', 'current_pitch', 'current_roll', 'absoluate_roll', 'time1',
       'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9',
       'time10', 'time11', 'time12', 'time13', 'time14', 'omega', 'set']

params={'n_estimators': 2078, 'max_depth': 5, 'subsample': 0.58, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5, 'learning_rate': 0.01}

train_features= pd.read_pickle("data/normalized_x_train_df.pkl")
train_labels=pd.read_pickle("data/normalized_y_train_df.pkl")
val_features=pd.read_pickle("data/normalized_x_val_df.pkl")
val_labels=pd.read_pickle("data/normalized_y_val_df.pkl")




# train_features=train_features[FEAT_SELECTIONS]
# val_features=val_features[FEAT_SELECTIONS]

# Use the random grid to search for best hyperparameters
# First create the base model to tune
xgbooster = xgb.XGBRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores

# xgb_random = RandomizedSearchCV(estimator = xgbooster, param_distributions = params, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# xgb_random = GridSearchCV(estimator = xgbooster, param_grid = random_grid, cv = 5, verbose=2, n_jobs = -1)

# Fit the random search model
# xgb_random.fit(train_features, train_labels)




# base_model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
# base_model.fit(train_features, train_labels)
#
# base_performance = mean_squared_error(val_labels, base_model.predict(val_features), squared=False)
#
# best_random = xgb_random.best_params_
# print(best_random)

# random_performance = mean_squared_error(val_labels, xgb_random.predict(val_features), squared=False)
#
# print('base model score %f ' % base_performance)
# print('random model score %f ' % random_performance)
# save
my_dumper=ModelDumper(train_features.columns,xgb.XGBRegressor)

my_dumper.dump_model(0.18,params)




