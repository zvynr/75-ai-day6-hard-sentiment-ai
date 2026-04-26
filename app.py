from flask import Flask, request, jsonify
import os
import logging
from config import config
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

env = os.environ.get('FLASK_NEW', 'production')
app.config.from_object(config[env])

logger.info(f"Starting application in {env} environment")
logger.info("Loading sentiment analysis model...")
# model loads once when app starts, all request use the same model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
logger.info("Model loaded sucessfully!")
@app.route('/')
def home():
    logger.info("Home endpoint accessed")
    return jsonify({
        'message' : app.config['API_TITLE'],
        'engineer' : app.config['ENGINEER'],
        'status' : 'running',
        'environment' :app.config['ENVIRONMENT'],
        "model": "distilbert-base-uncased-finetuned-sst-2-english"
    })
@app.route("/health")
def health():
    return jsonify({
        'status' : "healthy",
        "model_loaded" : sentiment_analyzer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "text" not in data: 
        logger.warning("Prediction request missing text field")
        return jsonify({
            "error" : "No text provided"
        }),400
    text = data.get('text', '')
    if len(text) == 0:
        logger.warning("Prediction request with empty text")
        return jsonify({
            "error" : "Empty text"
        }), 400
    if len(text) > 512: 
        logger.warning(f"Prediction request with text too long: {len(text)} chars")
        return jsonify({
            "error" : "Text too long (max 512 characters)"
        }),400
    result = sentiment_analyzer(text)[0]
    logger.info(f"Prediction: {result['label']} (confidence: {result['score']:.2f})")
    return jsonify({
        'text' : text,
        'sentiment' : result["label"],
        'confidence' : result["score"]
    })

# this blocked doesnt needed when using gunicorn, cause it import app diectly so this never been in use 
# if __name__ == '__main__' : 
#     app.run(debug=app.config['DEBUG'],
#             port=app.config['PORT'], 
#             host='0.0.0.0') #this makes the server accesible from other machine