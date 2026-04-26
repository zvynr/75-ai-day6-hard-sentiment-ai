import unittest
import json
from app import app

class TestSentimentAPI(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_home_endpoint(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertEqual(data['engineer'], 'Himanshu')

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

    def test_predict_success(self):
        response = self.client.post(
            '/predict',
            data=json.dumps({'text': 'This is amazing!'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('sentiment', data)
        self.assertIn('confidence', data)

    def test_predict_no_text(self):
        response = self.client.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_empty_text(self):
        response = self.client.post(
            '/predict',
            data=json.dumps({'text': ''}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
