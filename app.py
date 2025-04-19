from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging
import traceback

import pandas as pd
import joblib  # Changed import to directly import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

def scale(payload):
    """Scales Payload"""
    LOG.info("Scaling Payload: %s", payload)
    scaler = StandardScaler().fit(payload)
    scaled_adhoc_predict = scaler.transform(payload)
    return scaled_adhoc_predict

@app.route("/")
def home():
    html = "<h3>Sklearn Prediction Home: From Azure Pipelines (Continuous Delivery)</h3>"
    return html.format(format)

@app.route("/predict", methods=["POST"])
def predict():
    """Performs an sklearn prediction"""
    try:
        clf = joblib.load("boston_housing_prediction.joblib")
    except FileNotFoundError as e:
        LOG.error("Model file not found: %s", str(e))
        return jsonify({"error": "Model file not found"}), 404  # Return 404 if model file is not found
    except Exception as e:
        LOG.error("Error loading model: %s", str(e))
        LOG.error("Exception traceback: %s", traceback.format_exc())
        return jsonify({"error": "Model not loaded"}), 500  # Return error response if model fails to load

    try:
        json_payload = request.json
        if json_payload is None:
            raise ValueError("Invalid JSON payload")  # Raise an error if the payload is None

        LOG.info("JSON payload: %s", json_payload)
        inference_payload = pd.DataFrame(json_payload)
        LOG.info("Inference payload DataFrame: %s", inference_payload)
        scaled_payload = scale(inference_payload)
        
        prediction = list(clf.predict(scaled_payload))
        return jsonify({"prediction": prediction})
    
    except ValueError as e:
        LOG.error("Value error: %s", str(e))
        return jsonify({"error": str(e)}), 400  # Return 400 for bad requests
    except Exception as e:
        LOG.error("Error during prediction: %s", str(e))
        LOG.error("Exception traceback: %s", traceback.format_exc())
        return jsonify({"error": "Prediction failed"}), 500  # Return error response for prediction failure

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
