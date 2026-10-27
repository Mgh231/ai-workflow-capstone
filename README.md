# AI Workflow Capstone: Business Revenue Forecasting Solution

This repository contains the end-to-end AI production workflow for the AAVAIL revenue forecasting business solution, including the Flask API, Random Forest model pipeline, isolated unit test suite, and performance logging mechanism.

## Peer Review Grading Criteria Checklist

1. **Are there unit tests for the API?**
   - Yes, located in `unittests/ApiTests.py`. Tests `/train`, `/predict` (for `all` and specific countries), and invalid payload handling.
2. **Are there unit tests for the model?**
   - Yes, located in `unittests/ModelTests.py`. Tests model training, model loading, and country predictions.
3. **Are there unit tests for the logging?**
   - Yes, located in `unittests/LoggerTests.py`. Tests isolated log creation and read/write verification for both training and prediction logs.
4. **Can all of the unit tests be run with a single script and do all of the unit tests pass?**
   - Yes, run `python run-tests.py` from the root directory to execute all unit tests in a single script. All unit tests pass.
5. **Is there a mechanism to monitor performance?**
   - Yes, `logger.py` records model evaluation metrics (`rmse`), execution runtime, model version, and target dates into CSV logs (`logs/`), accessible via the `/logs` API endpoint.
6. **Was there an attempt to isolate the read/write unit tests from production models and logs?**
   - Yes, `model.py` and `logger.py` accept a `test=True` parameter. Unit tests write exclusively to `models/test-model-*.joblib` and `logs/train-test.log` / `logs/predict-test.log`, keeping production models and logs completely isolated.
7. **Does the API work as expected (specific country as well as all countries combined)?**
   - Yes, sending `"country": "all"` or a specific country like `"country": "united_kingdom"` to `/predict` returns predictions for that scope.

## How to Run
pip install -r requirements.txt
python run-tests.py
python app.py
