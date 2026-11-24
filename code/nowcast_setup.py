"""Settings shared by notebooks 03-05, so the cross-validation, the test and the SHAP step use the same
models, inputs and dates. Change a setting here, not in a notebook.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

SEED = 6400                                     # the course number; every random draw starts here
TEST_START = pd.Period("2021-01", freq="M")     # first test month; nothing from here on is used for any choice

# Inputs, as known on the day before each CPI release (built in notebook 02)
CORE = ["core_last", "core_avg3", "core_avg12"]             # core CPI's own first prints
USED_CARS = ["manheim_0", "manheim_1", "manheim_2"]         # Manheim wholesale used-car prices, % change
JET_FUEL = ["jet_0", "jet_1"]                               # Gulf Coast jet fuel, % change of the monthly average

FEATURE_SETS = {                                            # candidate input sets; CV picks one per model
    "core": CORE,
    "core + used cars": CORE + USED_CARS,
    "core + jet fuel": CORE + JET_FUEL,
    "core + used cars + jet fuel": CORE + USED_CARS + JET_FUEL,
}

FEATURE_LABELS = {
    "core_last": "Core CPI, last month's print",
    "core_avg3": "Core CPI, average of the last 3 prints",
    "core_avg12": "Core CPI, average of the last 12 prints",
    "manheim_0": "Used-car prices, same month",
    "manheim_1": "Used-car prices, one month earlier",
    "manheim_2": "Used-car prices, two months earlier",
    "jet_0": "Jet fuel, same month",
    "jet_1": "Jet fuel, one month earlier",
}
BLOCKS = {"core": "Core CPI history", "manheim": "Used-car prices", "jet": "Jet fuel"}

# Expanding-window cross-validation: train on every month before the block, score the 24-month block
FOLDS = [("2013-01", "2014-12"), ("2015-01", "2016-12"), ("2017-01", "2018-12"), ("2019-01", "2020-12")]

GRIDS = {
    "rf": [dict(max_depth=d, min_samples_leaf=leaf, max_features=f)
           for d in (2, 3, 5, None) for leaf in (5, 10, 20) for f in (1.0, 0.5)],
    "xgb": [dict(n_estimators=n, learning_rate=lr, max_depth=d, min_child_weight=w)
            for n in (100, 300, 600) for lr in (0.01, 0.05) for d in (1, 2, 3) for w in (5, 10)],
    "ols": [dict()],
}

MODEL_LABELS = {
    "ols": "Linear regression",
    "rf": "Random forest",
    "xgb": "Gradient boosting (XGBoost)",
}


def make_model(kind, params):
    """A fresh, unfitted model. Fixed seeds, so the same data always gives the same fit."""
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **params)
    if kind == "xgb":
        return XGBRegressor(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", random_state=SEED, n_jobs=1, **params)
    if kind == "ols":
        return LinearRegression()
    raise ValueError(kind)


def load_features(path):
    """features.csv from notebook 02, indexed by the reference month."""
    feat = pd.read_csv(path, parse_dates=["release_date", "nowcast_date"])
    feat["period"] = pd.PeriodIndex(feat["period"], freq="M")
    return feat.set_index("period")
