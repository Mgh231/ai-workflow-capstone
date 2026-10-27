import os
import unittest
from model import model_train, model_load, model_predict, _get_model_path

class ModelTest(unittest.TestCase):

    def test_01_train(self):
        """Test isolated model training."""
        model_train(test=True)
        self.assertTrue(os.path.exists(_get_model_path('all', test=True)))
        self.assertTrue(os.path.exists(_get_model_path('united_kingdom', test=True)))

    def test_02_load(self):
        """Test loading the trained model."""
        model = model_load(country='all', test=True)
        self.assertTrue(hasattr(model, 'predict'))

    def test_03_predict_all_and_specific_country(self):
        """Test predictions for both 'all' and a specific country."""
        res_all = model_predict('all', '2019', '06', '01', test=True)
        self.assertIsNotNone(res_all.get('y_pred'))

        res_uk = model_predict('united_kingdom', '2019', '06', '01', test=True)
        self.assertIsNotNone(res_uk.get('y_pred'))

if __name__ == '__main__':
    unittest.main()
