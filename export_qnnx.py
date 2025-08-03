"""
export_qnnx.py

This script exports the trained PyTorch chatbot model to ONNX,
then applies static quantization to produce a QNNX-compatible ONNX model.
Requires `onnxruntime-tools` to be installed (pip install onnxruntime-tools).
"""
import json
import numpy as np
import torch
from chatbot import ChatbotModel
from onnxruntime_tools.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader

# Load model metadata to reconstruct the model architecture
with open('data/chatbot_data.json', 'r') as f:
    metadata = json.load(f)
input_size = metadata['input_size']
output_size = metadata['output_size']

# Initialize and load the trained model
model = ChatbotModel(input_size, output_size)
state_dict = torch.load('models/chatbot_model.pth')
model.load_state_dict(state_dict)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, input_size)
torch.onnx.export(
    model,
    dummy_input,
    'models/chatbot.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=13
)
print('Exported PyTorch model to models/chatbot.onnx')

"""
Static quantization: insert QuantizeLinear/DequantizeLinear nodes into the ONNX graph.
"""
class RandomCalibrationReader(CalibrationDataReader):
    """Generates random samples for calibration."""
    def __init__(self, input_name, size, num_samples=10):
        self.input_name = input_name
        self.size = size
        self.num_samples = num_samples
        self.count = 0
    def get_next(self):
        if self.count < self.num_samples:
            data = np.random.rand(1, self.size).astype(np.float32)
            self.count += 1
            return {self.input_name: data}
        return None

print('Running static quantization with ONNX Runtime...')
calib_reader = RandomCalibrationReader(input_name='input', size=input_size)
quantize_static(
    model_input='models/chatbot.onnx',
    model_output='models/chatbot_qnnx.onnx',
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8
)
print('Quantized ONNX model saved to models/chatbot_qnnx.onnx')
