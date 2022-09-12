import os
import pickle

import numpy as np
import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient

MODEL_FILE = os.getenv("MODEL_FILE", "model.bin")

EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

with open(MODEL_FILE, "rb") as f_in:
    (dv, scaler, model) = pickle.load(f_in)


app = Flask("flight-price-prediction")

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")


@app.route("/predict", methods=["POST"])
def predict():

    record = request.get_json()
    X = dv.fit_transform(record)
    pred = model.predict(X)[0]
    pred = int(scaler.inverse_transform(np.array(pred).reshape(-1, 1))[0][0])
    result = {"Price": float(pred)}
    save_to_db(record, float(pred))
    send_to_evidently_service(record, float(pred))
    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec["Price"] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec["Price"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/taxi", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
