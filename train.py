import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
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
    Neural network model for the chatbot. Simple feed-forward network with two hidden layers.
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

class ChatbotTrainer:
    """
    Handles loading data, training the model, and saving artifacts.
    """
    def __init__(self, intents_path):
        self.intents_path = intents_path
        self.lemmatizer = WordNetLemmatizer()
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.X = None
        self.y = None
        self.model = None

    def _tokenize_and_lemmatize(self, text):
        """Tokenize text and lemmatize tokens."""
        words = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(w.lower()) for w in words]

    def _bag_of_words(self, words):
        """Convert token list to bag-of-words vector."""
        return [1 if w in words else 0 for w in self.vocabulary]

    def parse_intents(self):
        """Load and process intents JSON."""
        with open(self.intents_path, 'r') as f:
            data = json.load(f)
        for intent in data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent['responses']
            for pattern in intent['patterns']:
                tokens = self._tokenize_and_lemmatize(pattern)
                self.documents.append((tokens, tag))
                self.vocabulary.extend(tokens)
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        """Prepare training data (X, y)."""
        bags = []
        labels = []
        for tokens, tag in self.documents:
            bags.append(self._bag_of_words(tokens))
            labels.append(self.intents.index(tag))
        self.X = np.array(bags)
        self.y = np.array(labels)

    def train_model(self, batch_size, lr, epochs):
        """Train the neural network model."""
        dataset = TensorDataset(
            torch.tensor(self.X, dtype=torch.float32),
            torch.tensor(self.y, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(1, epochs+1):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch}/{epochs} - Loss: {total_loss/len(loader):.4f}')

    def save_model(self, model_path, data_path):
        """Save model state and metadata."""
        torch.save(self.model.state_dict(), model_path)
        metadata = {
            'input_size': self.X.shape[1],
            'output_size': len(self.intents),
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'intents_responses': self.intents_responses
        }
        with open(data_path, 'w') as f:
            json.dump(metadata, f, indent=4)

if __name__ == '__main__':
    # Train and save the chatbot model
    trainer = ChatbotTrainer('data/intents.json')
    trainer.parse_intents()
    trainer.prepare_data()
    trainer.train_model(batch_size=8, lr=0.001, epochs=100)
    trainer.save_model('models/chatbot_model.pth', 'data/chatbot_data.json')
    print('Training complete. Model and data saved.')
