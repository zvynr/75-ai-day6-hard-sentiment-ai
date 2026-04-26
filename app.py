from flask import Flask, request, jsonify
import os
from config import config
from transformers import pipeline

app = Flask(__name__)

env = os.environ.get('FLASK_NEW', 'production')
app.config.from_object(config[env])

print("Loading sentiment analysis model...")
# model loads once when app starts, all request use the same model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded sucessfully!")
@app.route('/')
def home():
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
        return jsonify({
            "error" : "No text provided"
        }),400
    text = data.get('text', '')
    if len(text) == 0:
        return jsonify({
            "error" : "Empty text"
        }), 400
    if len(text) > 512: 
        return jsonify({
            "error" : "Text too long (max 512 characters)"
        }),400
    result = sentiment_analyzer(text)[0]

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