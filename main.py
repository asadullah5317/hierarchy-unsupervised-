import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Make sure your trained model is saved as 'model.pkl' in the same folder
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Upload a CSV to /predict using field name 'file'"})


@app.route("/predict", methods=["POST"]) 
def predict():
    # Expect a CSV file upload with field name 'file'
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file with field name 'file'"}), 400

    file = request.files["file"]

    # Read the CSV into a DataFrame
    df = pd.read_csv(file)

    # IMPORTANT: CSV columns (headers) must match the features used during training
    preds = model.predict(df)

    return jsonify({"predictions": [float(p) for p in preds]})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)