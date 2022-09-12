import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.prepare_data import data_for_modeling

X, y = data_for_modeling("../res/flight_data_train.parquet")[:2]
df = pd.concat(
    [
        pd.DataFrame(X, columns=["Airline", "Source", "Destination", "Total_Stops"]),
        pd.DataFrame(y, columns=["Price"]),
    ],
    axis=1,
)
df1, df2 = train_test_split(df, test_size=0.2)
df1, df3 = train_test_split(df1, test_size=0.3)

df1.to_parquet("../evidently_service/datasets/flights_1.parquet")
df2.to_parquet("../evidently_service/datasets/flights_2.parquet")
df3.to_parquet("../evidently_service/datasets/flights_3.parquet")
