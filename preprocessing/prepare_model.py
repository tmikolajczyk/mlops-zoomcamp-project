import pickle
from Typing import Tuple
from urllib.parse import urlparse

import mlflow
import numpy as np
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from preprocessing.prepare_data import data_for_modeling

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.set_experiment("flight-price-prediction")


def eval_metrics(actual: np.array, pred: np.array) -> Tuple[float, float, float]:
    """
    Helper function for the main one for metrics
    :param actual: numpy array with actual values
    :param pred: numpy array with predicted values
    :return: RMSE, MAE and R2 scores (floats).
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def get_best_model(filename: str, random_grid: dict) -> None:
    """
    Function sekks for the best model with provided grid search parameters.
    :param filename: string with raw data file path
    :param random_grid: dictionary of parameters for cross validated search
    of best model parameters
    :return: None; model is saved to pickled file
    """
    X, y, multi, scaler = data_for_modeling(filename)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )

    with mlflow.start_run():
        mlflow.sklearn.autolog()
        mlflow.set_tag("model", "RandomForest")
        mlflow.set_tag("developer", "tom")
        rf = RandomizedSearchCV(
            estimator=RandomForestRegressor(),
            param_distributions=random_grid,
            scoring="neg_mean_squared_error",
            n_iter=10,
            cv=5,
            verbose=0,
            random_state=42,
            n_jobs=2,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    params = rf.best_params_

    with mlflow.start_run():
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train.ravel())
        predicted_qualities = rf.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(rf, "model", registered_model_name="RF_model")
        else:
            mlflow.sklearn.log_model(rf, "model")

    # let's choose the latest model and save it for app
    client = MlflowClient()
    rm = client.list_registered_models()[-1]
    registered_model = dict(dict(rm).get("latest_versions")[0])
    registered_model.get("version")

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{registered_model.get('name')}/{registered_model.get('version')}"
    )

    dv = DictVectorizer()

    with open("../prediction_service/model.bin", "wb") as f_out:
        pickle.dump((dv, scaler, model), f_out)
