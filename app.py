from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging
from config import config
from transformers import pipeline
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
# restrict access to spesific domain
# CORS(app, resources={r"/*": {"origins": ["https://yourdomain.com"]}})


env = os.environ.get('FLASK_ENV', 'production')
app.config.from_object(config[env])

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="memory://"
)

logger.info(f"Starting application in {env} environment")
logger.info("Loading sentiment analysis model...")

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

logger.info("Model loaded successfully!")

@app.route('/')
def home():
    logger.info("Home endpoint accessed")
    return jsonify({
        'message': app.config['API_TITLE'],
        'engineer': app.config['ENGINEER'],
        'status': 'running',
        'environment': app.config['ENVIRONMENT'],
        'model': 'distilbert-base-uncased-finetuned-sst-2-english'
    })

@app.route('/health')
def health():
    try:
        test_text = "test"
        result = sentiment_analyzer(test_text)

        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_responsive': True
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': sentiment_analyzer is not None,
            'model_responsive': False,
            'error': str(e)
        }), 503

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        logger.warning("Prediction request missing text field")
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    if len(text) == 0:
        logger.warning("Prediction request with empty text")
        return jsonify({'error': 'Empty text'}), 400

    if len(text) > 512:
        logger.warning(f"Prediction request with text too long: {len(text)} chars")
        return jsonify({'error': 'Text too long (max 512 characters)'}), 400

    logger.info(f"Processing prediction for text of length {len(text)}")

    result = sentiment_analyzer(text)[0]

    logger.info(f"Prediction: {result['label']} (confidence: {result['score']:.2f})")

    return jsonify({
        'text': text,
        'sentiment': result['label'],
        'confidence': result['score']
    })
