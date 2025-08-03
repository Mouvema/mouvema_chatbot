Project Structure Overview
==========================

This document explains the organization of the Mouvema Chatbot project, describing each directory, Python script, model artifact, and JSON file so that a new contributor can quickly understand and extend the project.

1. Root Directory
-----------------
- PROJECT_OVERVIEW.txt      : This overview document.
- app.py                    : Launches the chatbot in a Tkinter GUI window for inference.
- chatbot.py                : Defines the `ChatbotModel` (FFN) and `Chatbot` inference class.
- train.py                  : Parses `data/intents.json`, trains the model, and saves model & metadata.
- quantize.py               : Dynamically quantizes the PyTorch model and scripts it to TorchScript.
- export_qnnx.py            : Exports the full-precision model to ONNX, then applies static quantization (QNNX) via ONNX Runtime Tools.
- main.py                   : Deprecated legacy training/inference demo; replaced by `train.py` and `app.py`.
- venv/                     : Python virtual environment (contains dependencies).

2. data/ Directory
------------------
Holds all JSON metadata and training data:

- intents.json                   : Defines chatbot intents, example patterns, and responses.
- nlu_training_data.json         : (Optional) Additional user utterance examples for training.
- nlu_training_data_complex.json : (Optional) Complex utterance examples for advanced training.
- dimensions.json                : (Deprecated) Saved input/output sizes (legacy from `main.py`).
- chatbot_data.json              : Metadata for inference (vocabulary, sizes, intent tags, responses).

3. models/ Directory
--------------------
Contains trained and exported model artifacts:

- chatbot_model.pth          : Full-precision PyTorch state dict for `ChatbotModel`.
- chatbot_model_quantized.pt : TorchScript–serialized, dynamically quantized model (8-bit linear layers).
- chatbot.onnx               : Floating-point ONNX export of the model (opset 13).
- chatbot_qnnx.onnx          : Statically quantized ONNX graph in QOperator (QNNX) format.(best for deployment)

4. Script Details
-----------------

- app.py:
  • Initializes `Chatbot(model_path, data_path)` and starts a Tkinter window.
  • Displays conversation history and handles user input.

- chatbot.py:
  • Defines `ChatbotModel` (two hidden layers + dropout).
  • Implements `Chatbot` class for inference: loads metadata (`data/chatbot_data.json`), loads the model state, tokenizes input, makes predictions, and returns random responses.

- train.py:
  • Loads `data/intents.json` and builds vocabulary/document pairs.
  • Converts text to bag-of-words vectors and trains the FFN model.
  • Saves the model (`models/chatbot_model.pth`) and metadata (`data/chatbot_data.json`).

- quantize.py:
  • Reads metadata and full-precision weights.
  • Applies PyTorch dynamic quantization to all `nn.Linear` layers.
  • Scripts the quantized model via TorchScript and outputs `models/chatbot_model_quantized.pt`.

- export_qnnx.py:
  • Reads metadata and full-precision weights.
  • Exports to ONNX (FP32) with dynamic batch dimension.
  • Defines `RandomCalibrationReader` to generate sample inputs.
  • Calls ONNX Runtime Tools’ `quantize_static` to insert quantization operators and output `models/chatbot_qnnx.onnx`.

- main.py (Deprecated):
  • Legacy combined trainer/inference example.
  • Use `train.py` and `app.py` instead.

5. How to Run
-------------

1. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. Install requirements (if not already installed):
   ```powershell
   pip install -r requirements.txt
   ```
3. Train the model:
   ```powershell
   python train.py
   ```
4. Quantize PyTorch model:
   ```powershell
   python quantize.py
   ```
5. Generate ONNX + QNNX model:
   ```powershell
   python export_qnnx.py
   ```
6. Launch chatbot GUI:
   ```powershell
   python app.py
   ```

Now you’re ready to extend or integrate this chatbot in production! Feel free to explore code, add new intents, or swap frontend interfaces.
