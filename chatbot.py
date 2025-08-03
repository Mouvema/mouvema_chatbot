import json
import random
import nltk
import torch
import torch.nn as nn

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ChatbotModel(nn.Module):
    """
    The same model architecture used during training.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class Chatbot:
    """
    Chatbot class for inference. Load trained model and metadata.
    """
    def __init__(self, model_path, data_path, function_mappings=None):
        # Load metadata (vocabulary, intents, responses)
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.vocabulary = data['vocabulary']
        self.intents = data['intents']
        self.intents_responses = data['intents_responses']
        self.function_mappings = function_mappings
        self.lemmatizer = nltk.WordNetLemmatizer()

        # Load trained model
        self.model = ChatbotModel(data['input_size'], data['output_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def _tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize input text."""
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(tok.lower()) for tok in tokens]

    def _bag_of_words(self, words):
        """Convert token list to bag-of-words vector."""
        return [1 if w in words else 0 for w in self.vocabulary]

    def get_response(self, message):
        """
        Return a response for the given message.
        """
        # Preprocess
        words = self._tokenize_and_lemmatize(message)
        bag = self._bag_of_words(words)
        input_tensor = torch.tensor([bag], dtype=torch.float32)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
        intent_idx = torch.argmax(outputs, dim=1).item()
        intent_tag = self.intents[intent_idx]

        # Execute mapped function if any
        if self.function_mappings and intent_tag in self.function_mappings:
            self.function_mappings[intent_tag]()

        # Return random response
        responses = self.intents_responses.get(intent_tag, [])
        return random.choice(responses) if responses else None
