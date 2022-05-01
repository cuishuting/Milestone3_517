import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import xgboost as xgb

from ray import tune
from xgboost import XGBRegressor

RANDOMSTATE = 42
kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOMSTATE)
ultimate_train_feat = pd.read_pickle("data/normalized_x_ultimate_train_df.pkl")
ultimate_train_y = pd.read_pickle("data/normalized_y_ultimate_train_df.pkl")
FEAT_SELECTIONS=['m', 'n', 'current_pitch', 'current_roll', 'absoluate_roll', 'time1',
       'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9',
       'time10', 'time11', 'time12', 'time13', 'time14', 'omega', 'set']
FEAT_EXCLUSION= ['time8_delta','time10_delta','time14_delta','time12_delta',
                      'time6_delta','time2_delta','time4_delta','time5_delta',
                      'time3_delta','time13_delta','acc_rate','climb_delta'
                      ]
xgb_tune_kwargs = {
    "n_estimators": tune.loguniform(100, 10000),
    "max_depth": tune.randint(0, 5),
    "subsample": tune.quniform(0.25, 0.75, 0.01),
    "colsample_bytree": tune.quniform(0.05, 0.5, 0.01),
    "colsample_bylevel": tune.quniform(0.05, 0.5, 0.01),
    "learning_rate": tune.quniform(-3.0, -1.0, 0.5),  # powers of 10
}

xgb_tune_params = [k for k in xgb_tune_kwargs.keys() if k != 'wandb']


def my_xgb(config):
    # fix these configs to match calling convention
    # search wants to pass in floats but xgb wants ints
    config['n_estimators'] = int(config['n_estimators'])  # pass float eg loguniform distribution, use int
    # hyperopt needs left to start at 0 but we want to start at 2
    config['max_depth'] = int(config['max_depth']) + 2
    config['learning_rate'] = 10 ** config['learning_rate']

    xgb = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=1,
        random_state=RANDOMSTATE,
        booster='gbtree',
        scale_pos_weight=1,
        **config,
    )
    scores = -cross_val_score(xgb, ultimate_train_feat[df.columns[~df.columns.isin(['C','D'])]], ultimate_train_y,
                              scoring="neg_root_mean_squared_error",
                              cv=kfolds)
    rmse = np.mean(scores)
    tune.report(rmse=rmse)

    return {"rmse": rmse}

algo = HyperOptSearch(random_state_seed=RANDOMSTATE)
# ASHA
scheduler = AsyncHyperBandScheduler()

analysis = tune.run(my_xgb,
                    num_samples=1024,
                    config=xgb_tune_kwargs,
                    name="hyperopt_xgb",
                    metric="rmse",
                    mode="min",
                    search_alg=algo,
                    scheduler=scheduler,
                    verbose=1,
                   )