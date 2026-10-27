from flask import Flask, jsonify, request
from model import model_train, model_predict, SUPPORTED_COUNTRIES
import os

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    req = request.get_json(silent=True) or {}
    test_mode = req.get('mode', 'prod') == 'test'
    model_train(test=test_mode)
    return jsonify({"status": "success", "message": "Models trained successfully."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json(silent=True)
    if not req:
        return jsonify({"error": "Invalid JSON payload provided."}), 400

    country = req.get('country', 'all')
    year = req.get('year')
    month = req.get('month')
    day = req.get('day')
    test_mode = req.get('mode', 'prod') == 'test'

    if not all([year, month, day]):
        return jsonify({"error": "Missing required parameters: year, month, day"}), 400

    if country not in SUPPORTED_COUNTRIES:
        return jsonify({"error": f"Unsupported country: {country}"}), 400

    result = model_predict(country, year, month, day, test=test_mode)
    return jsonify(result), 200

@app.route('/logs', methods=['GET'])
def list_logs():
    """Endpoint for performance monitoring log retrieval."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    files = os.listdir(log_dir) if os.path.exists(log_dir) else []
    return jsonify({"available_logs": files}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
