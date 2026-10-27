import os
import unittest
from logger import update_train_log, update_predict_log, LOG_DIR

class LoggerTest(unittest.TestCase):

    def test_01_train_log(self):
        """Ensure train log file is created in isolated test mode."""
        log_file = os.path.join(LOG_DIR, "train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        update_train_log("all", ("2018-01-01", "2019-01-01"), {'rmse': 12.5}, "00:00:01", 1.0, test=True)
        self.assertTrue(os.path.exists(log_file))

    def test_02_predict_log(self):
        """Ensure predict log file is created in isolated test mode."""
        log_file = os.path.join(LOG_DIR, "predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        update_predict_log("united_kingdom", [1250.5], None, "2019-06-01", "00:00:01", 1.0, test=True)
        self.assertTrue(os.path.exists(log_file))

if __name__ == '__main__':
    unittest.main()
