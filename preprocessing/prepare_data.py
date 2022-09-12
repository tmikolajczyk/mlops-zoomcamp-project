from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.multi_column_labeler import MultiColumnLabelEncoder


def data_for_modeling(
    filename: str,
) -> Tuple[pd.DataFrame, np.ndarray, MultiColumnLabelEncoder, StandardScaler]:
    df = pd.read_parquet(filename)
    df.dropna(inplace=True)

    # correcting airport labels
    df["Destination"] = df["Destination"].replace("New Delhi", "Delhi")
    df["Source"] = df["Source"].replace("New Delhi", "Delhi")

    # drop unnecessary columns
    df.drop(
        [
            "Date_of_Journey",
            "Route",
            "Dep_Time",
            "Arrival_Time",
            "Duration",
            "Additional_Info",
        ],
        inplace=True,
        axis=1,
    )

    df = df.drop_duplicates()

    # encoding categorical columns into numeric
    X = df.drop("Price", axis=1)
    multi = MultiColumnLabelEncoder(
        columns=["Airline", "Source", "Destination", "Total_Stops"]
    )
    X = multi.fit_transform(X)

    # scaling the target column
    y = df[["Price"]].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(y)
    y = scaler.transform(y)

    # remove any non-numeric values
    for col in X.columns:
        X = X[pd.to_numeric(X[col], errors="coerce").notnull()]

    X = X.values

    return X, y, multi, scaler
