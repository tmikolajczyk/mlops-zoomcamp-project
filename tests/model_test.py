import pandas as pd

from preprocessing.prepare_data import data_for_modeling


def test_prepare_features():
    filename = "flight_data_train.parquet"
    multi = data_for_modeling(filename)[2]
    airlines = pd.read_parquet(filename)["Airline"].unique()
    airlines.sort()

    assert all(airlines == multi.encoders.get("Airline").classes_)


def check_data_modeling():
    filename = "flight_data_train.parquet"
    df = pd.read_parquet(filename)
    cols = ["Airline", "Source", "Destination", "Total_Stops"]

    assert all([col in df.columns for col in cols])
