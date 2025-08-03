"""
api.py

Provides a REST API for the chatbot using the quantized ONNX QNNX model.
Endpoints:
  POST /predict
    Request JSON: { "message": "...user text..." }
    Response JSON: { "response": "...bot answer..." }

Run with:
  uvicorn api:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available
for pkg in ("punkt", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

# Load metadata
with open('data/chatbot_data.json', 'r') as f:
    metadata = json.load(f)
vocabulary = metadata['vocabulary']
intents = metadata['intents']
responses_map = metadata['intents_responses']

# Initialize ONNX Runtime session
session = ort.InferenceSession('models/chatbot_qnnx.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Helper functions
lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text: str):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(tok.lower()) for tok in tokens]

def bag_of_words(tokens):
    return np.array([1 if w in tokens else 0 for w in vocabulary], dtype=np.float32)

# FastAPI setup
app = FastAPI(title="Chatbot API")

class Message(BaseModel):
    message: str

class Prediction(BaseModel):
    response: str

@app.post("/predict", response_model=Prediction)
def predict(msg: Message):
    words = tokenize_and_lemmatize(msg.message)
    bow = bag_of_words(words)
    # ONNX Runtime expects batch dimension
    input_tensor = bow.reshape(1, -1)
    try:
        preds = session.run([output_name], {input_name: input_tensor})[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Get predicted intent
    intent_idx = int(np.argmax(preds, axis=1)[0])
    intent_tag = intents[intent_idx]
    # Choose a random response
    choices = responses_map.get(intent_tag, [])
    if not choices:
        reply = "Sorry, I don't understand."
    else:
        reply = np.random.choice(choices)
    return Prediction(response=reply)
