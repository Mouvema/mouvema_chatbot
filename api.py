"""
api.py

Provides a REST API for the chatbot using the quantized ONNX QNNX model.
This version uses only Python's standard library for the web server to minimize dependencies.

Endpoint:
  POST /predict
    Request JSON: { "message": "...user text..." }
    Response JSON: { "response": "...bot answer..." }

Run with:
  python api.py
"""
import http.server
import socketserver
import json
import os
import numpy as np
import onnxruntime as ort

# --- Model & Data Loading ---
# Get the absolute path of the directory containing this script
_DIR = os.path.dirname(os.path.abspath(__file__))

# Load metadata
with open(os.path.join(_DIR, 'data', 'chatbot_data.json'), 'r') as f:
    metadata = json.load(f)
vocabulary = metadata['vocabulary']
intents = metadata['intents']
responses_map = metadata['intents_responses']

# Initialize ONNX Runtime session
session = ort.InferenceSession(os.path.join(_DIR, 'models', 'chatbot_qnnx.onnx'))
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Helper Functions ---
def tokenize(text: str):
    """Simple whitespace tokenizer and lowercase."""
    return text.lower().split()

def bag_of_words(tokens):
    """Generates a bag-of-words array for a sentence."""
    return np.array([1 if w in tokens else 0 for w in vocabulary], dtype=np.float32)

def get_prediction(message: str):
    """Runs the full prediction pipeline."""
    words = tokenize(message)
    bow = bag_of_words(words)
    # ONNX Runtime expects a batch dimension
    input_tensor = bow.reshape(1, -1)
    
    preds = session.run([output_name], {input_name: input_tensor})[0]
    
    intent_idx = int(np.argmax(preds, axis=1)[0])
    intent_tag = intents[intent_idx]
    
    choices = responses_map.get(intent_tag, [])
    if not choices:
        return "Sorry, I don't understand."
    
    return np.random.choice(choices)

# --- HTTP Server ---
class ChatbotRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handles incoming HTTP requests."""
    def do_POST(self):
        """Handles POST requests to /predict."""
        if self.path == '/predict':
            try:
                # Read and parse the request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                body = json.loads(post_data)
                
                if 'message' not in body:
                    raise ValueError("Request JSON must contain a 'message' key.")

                # Get the bot's response
                response_text = get_prediction(body['message'])
                response_payload = json.dumps({'response': response_text})
                
                # Send the response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(response_payload.encode('utf-8'))

            except (json.JSONDecodeError, ValueError) as e:
                self.send_error(400, message=f"Bad Request: {e}")
            except Exception as e:
                self.send_error(500, message=f"Internal Server Error: {e}")
        else:
            self.send_error(404, "Not Found")

def run_server(port=8000):
    """Starts the HTTP server."""
    with socketserver.TCPServer(("", port), ChatbotRequestHandler) as httpd:
        print(f"Serving chatbot API on http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
