import numpy as np

from preprocessing.prepare_model import get_best_model

filename = "res/flight_data_train.parquet"

random_grid = {
    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
    "max_depth": [int(x) for x in np.linspace(5, 30, num=6)],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf": [1, 2, 5, 10],
}

get_best_model(filename, random_grid)
