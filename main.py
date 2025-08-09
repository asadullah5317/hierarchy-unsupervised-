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
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or empty JSON"}), 400

    # Build DataFrame in one line (works for single object or list of objects)
    df = pd.DataFrame(payload if isinstance(payload, list) else [payload])

    # Optional: align columns if the model exposes feature names
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=list(model.feature_names_in_), fill_value=0)

    preds = model.predict(df)

    # Return single or batch format automatically
    if isinstance(payload, list):
        return jsonify({"predictions": [float(p) for p in preds]}), 200
    else:
        return jsonify({"prediction": float(preds[0])}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)