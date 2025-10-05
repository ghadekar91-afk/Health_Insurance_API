from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

app = Flask(__name__)

# Global model file path
MODEL_FILE = "model.pkl"
model = None

# ---------- Helper Function ----------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess insurance dataset for training/prediction."""
    if "region" in df.columns:
        df = df.drop("region", axis=1)

    df["sex_enc"] = df["sex"].apply(lambda x: 1 if x.lower() == "female" else 0)
    df["smoker_enc"] = df["smoker"].apply(lambda x: 1 if x.lower() == "yes" else 0)
    return df

def load_saved_model():
    """Load model from pickle file if available."""
    global model
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print("✅ Loaded saved model from model.pkl")
    else:
        print("⚠️ No saved model found. Train first.")

# ---------- API Endpoints ----------
@app.route("/train", methods=["POST"])
def train():
    """Train the insurance prediction model using uploaded CSV file and save it."""
    global model

    if "file" not in request.files:
        return jsonify({"error": "Upload CSV file with key 'file'"}), 400

    file = request.files["file"]
    df = pd.read_csv(file)
    df = preprocess(df)

    X_train = df[["age", "bmi", "children", "sex_enc", "smoker_enc"]]
    y_train = df["charges"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save trained model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return jsonify({"status": "Model trained and saved successfully"})


@app.route("/test", methods=["POST"])
def test():
    """Evaluate the trained model on a test dataset (CSV)."""
    global model
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    if "file" not in request.files:
        return jsonify({"error": "Upload CSV file with key 'file'"}), 400

    file = request.files["file"]
    df = pd.read_csv(file)
    df = preprocess(df)

    X_test = df[["age", "bmi", "children", "sex_enc", "smoker_enc"]]
    y_test = df["charges"]

    y_pred = model.predict(X_test)

    metrics = {
        "r2_score": model.score(X_test, y_test),
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
    }

    return jsonify(metrics)


@app.route("/predict", methods=["POST"])
def predict():
    """Make a prediction using named JSON fields."""
    global model
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    req_data = request.get_json()
    if not req_data:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        age = req_data["age"]
        bmi = req_data["bmi"]
        children = req_data["children"]
        sex = req_data["sex"].lower()   # "male" / "female"
        smoker = req_data["smoker"].lower()  # "yes" / "no"
    except KeyError as e:
        return jsonify({"error": f"Missing field {str(e)}"}), 400

    # Encode categorical values
    sex_enc = 1 if sex == "female" else 0
    smoker_enc = 1 if smoker == "yes" else 0

    features = [[age, bmi, children, sex_enc, smoker_enc]]
    prediction = model.predict(features)

    return jsonify({
        "input": req_data,
        "predicted_charges": float(prediction[0])
    })


# ---------- Run ----------
if __name__ == "__main__":
    load_saved_model()  # Try loading existing model.pkl
    app.run(debug=True)
