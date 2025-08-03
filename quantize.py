"""
quantize.py

This script dynamically quantizes the trained chatbot model for smaller size
and faster CPU inference. It uses PyTorch's dynamic quantization on all Linear layers.
"""
import json
import torch
import torch.quantization
from chatbot import ChatbotModel

# Load model metadata to reconstruct the network architecture
with open('data/chatbot_data.json', 'r') as f:
    metadata = json.load(f)

input_size = metadata['input_size']
output_size = metadata['output_size']

# Initialize the model and load trained weights
model = ChatbotModel(input_size, output_size)
state_dict = torch.load('models/chatbot_model.pth')
model.load_state_dict(state_dict)
model.eval()

# Apply dynamic quantization to all nn.Linear layers
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Convert to TorchScript for deployment
scripted_model = torch.jit.script(quantized_model)

# Save the quantized model
quantized_path = 'models/chatbot_model_quantized.pt'
scripted_model.save(quantized_path)

print(f'Quantized model saved to {quantized_path}')
