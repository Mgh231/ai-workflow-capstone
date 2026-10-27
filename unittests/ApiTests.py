import unittest
from app import app

class ApiTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_01_train_endpoint(self):
        response = self.client.post('/train', json={'mode': 'test'})
        self.assertEqual(response.status_code, 200)

    def test_02_predict_all_countries(self):
        payload = {'country': 'all', 'year': '2019', 'month': '06', 'day': '05', 'mode': 'test'}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn('y_pred', response.get_json())

    def test_03_predict_specific_country(self):
        payload = {'country': 'united_kingdom', 'year': '2019', 'month': '06', 'day': '05', 'mode': 'test'}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['country'], 'united_kingdom')

    def test_04_invalid_input_handling(self):
        payload = {'country': 'invalid_country', 'year': '2019', 'month': '06', 'day': '05', 'mode': 'test'}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
