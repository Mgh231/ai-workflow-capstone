import os
import csv
import uuid
from datetime import date

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def update_train_log(country, date_range, eval_metric, runtime, model_version, test=False):
    """Logs model training performance and metadata."""
    today = date.today()
    if test:
        logfile = os.path.join(LOG_DIR, "train-test.log")
    else:
        logfile = os.path.join(LOG_DIR, f"train-{today.year}-{today.month}.log")

    header = ['unique_id', 'country', 'date_range', 'eval_metric', 'runtime', 'model_version']
    write_header = not os.path.exists(logfile)

    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow([uuid.uuid4(), country, date_range, eval_metric, runtime, model_version])

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    """Logs prediction output and latency for performance monitoring."""
    today = date.today()
    if test:
        logfile = os.path.join(LOG_DIR, "predict-test.log")
    else:
        logfile = os.path.join(LOG_DIR, f"predict-{today.year}-{today.month}.log")

    header = ['unique_id', 'country', 'y_pred', 'y_proba', 'target_date', 'runtime', 'model_version']
    write_header = not os.path.exists(logfile)

    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow([uuid.uuid4(), country, y_pred, y_proba, target_date, runtime, model_version])
