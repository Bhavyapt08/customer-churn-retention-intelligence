from pathlib import Path
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.scoring import score_single_customer


MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "tuned_xgboost_churn_model.joblib"
)

FEATURE_PATH = (
    MODEL_DIR
    / "model_features.csv"
)


# Load model and feature names once
model = joblib.load(MODEL_PATH)

model_features = (
    pd.read_csv(FEATURE_PATH)["Feature"]
    .tolist()
)


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message":
                "Customer Churn Prediction API",
            "status":
                "running"
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "feature_count": len(model_features)
        }
    )

@app.route("/features", methods=["GET"])
def features():
    return jsonify(
        {
            "feature_count": len(model_features),
            "required_features": model_features
        }
    )

@app.route("/predict", methods=["POST"])
def predict():

    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify(
                {
                    "error":
                        "No input data provided."
                }
            ), 400

        missing_features = [
            feature
            for feature in model_features
            if feature not in input_data
        ]

        if missing_features:
            return jsonify(
                {
                    "error":
                        "Missing required features.",
                    "missing_features":
                        missing_features
                }
            ), 400

        non_numeric_features = []

        for feature in model_features:
            value = input_data[feature]

            if not isinstance(
                value,
                (int, float)
            ):
                non_numeric_features.append(
                    feature
                )

        if non_numeric_features:
            return jsonify(
                {
                    "error":
                        "All model features must be numeric.",
                    "invalid_features":
                        non_numeric_features
                }
            ), 400

        customer_features = {
            feature: input_data[feature]
            for feature in model_features
        }

        result = score_single_customer(
            model=model,
            customer_features=customer_features,
            model_features=model_features
        )

        response = {
            "customer_id":
                input_data.get(
                    "Customer ID",
                    None
                ),

            "churn_probability":
                round(
                    result[
                        "churn_probability"
                    ],
                    4
                ),

            "risk_segment":
                result[
                    "risk_segment"
                ],

            "retention_priority":
                result[
                    "retention_priority"
                ],

            "recommended_action":
                result[
                    "recommended_action"
                ]
        }

        return jsonify(response)

    except Exception as error:

        return jsonify(
            {
                "error":
                    str(error)
            }
        ), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )