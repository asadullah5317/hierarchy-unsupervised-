import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "House Price Prediction API is running."})


@app.route("/predict", methods=["POST"]) 
def predict():
    data = request.get_json(silent=True) or {}

    # Expected keys (example: California housing dataset)
    # Keep your training features and names the same
    expected_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",  # categorical
    ]

    # Create a single-row DataFrame. If you used a Pipeline during training,
    # it will handle scaling/encoding for you.
    row = {name: data.get(name) for name in expected_features}
    df = pd.DataFrame([row])

    prediction = model.predict(df)[0]
    return jsonify({"prediction": float(prediction)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)