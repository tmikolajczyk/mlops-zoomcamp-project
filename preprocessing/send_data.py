import json
import uuid
from time import sleep

import pyarrow.parquet as pq
import requests

table = pq.read_table("flights_2.parquet")
data = table.to_pylist()


with open("target.csv", 'w') as f_target:
    for row in data:
        row_id = str(uuid.uuid4())
        Price = row['Price']
        del row['Price']
        if Price != 0.0:
            f_target.write(f"{row_id},{Price}\n")
        resp = requests.post("http://127.0.0.1:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(row)).json()
        print(f"prediction: {resp['Price']}")
        sleep(1)
