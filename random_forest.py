import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np

# FEAT_SELECTIONS=['m', 'n', 'current_pitch', 'current_roll', 'absoluate_roll', 'time1',
#        'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9',
#        'time10', 'time11', 'time12', 'time13', 'time14', 'omega', 'set']
from utils import ModelDumper

FEAT_SELECTIONS=['m', 'n', 'current_pitch', 'current_roll', 'absoluate_roll', 'time1',
       'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9',
       'time10', 'time11', 'time12', 'time13', 'time14']
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,15,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,8]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
train_features= pd.read_pickle("data/normalized_x_train_df.pkl")
train_labels=pd.read_pickle("data/normalized_y_train_df.pkl")
val_features=pd.read_pickle("data/normalized_x_val_df.pkl")
val_labels=pd.read_pickle("data/normalized_y_val_df.pkl")




train_features=train_features[FEAT_SELECTIONS]
val_features=val_features[FEAT_SELECTIONS]

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 3, verbose=2, n_jobs = -1)

# Fit the random search model
rf_random.fit(train_features, train_labels)




base_model = RandomForestRegressor(n_estimators=10, random_state=42)
base_model.fit(train_features, train_labels)

base_performance = mean_squared_error(val_labels, base_model.predict(val_features), squared=False)

best_random = rf_random.best_params_
print(best_random)

random_performance = mean_squared_error(val_labels, rf_random.predict(val_features), squared=False)

print('base model score %f ' % base_performance)
print('random model score %f ' % random_performance)
# save
my_dumper=ModelDumper(FEAT_SELECTIONS,RandomForestRegressor)

#test
my_dumper.dump_model(base_performance,base_model.get_params())

# my_dumper.dump_model(random_performance,best_random)




