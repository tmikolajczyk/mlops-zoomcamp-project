import requests
from tqdm import tqdm

files = [
    ("flights_1.parquet", "."),
    ("flights_1.parquet", "./evidently_service/datasets"),
]

print(f"Download files:")
for file, path in files:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
    resp = requests.get(url, stream=True)
    save_path = f"{path}/{file}"
    with open(save_path, "wb") as handle:
        for data in tqdm(
            resp.iter_content(),
            desc=f"{file}",
            postfix=f"save to {save_path}",
        ):
            handle.write(data)
