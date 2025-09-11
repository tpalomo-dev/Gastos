# import os
# import onnxruntime as ort
# from tokenizers import Tokenizer
# import numpy as np
# import json

# # Build paths relative to this script
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_DIR = os.path.join(BASE_DIR, "lightweight_model")
# MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
# TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
# ID2LABEL_PATH = os.path.join(MODEL_DIR, "id2label.json")

# # Load ONNX model
# ort_sess = ort.InferenceSession(MODEL_PATH)

# # Load tokenizer
# tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# # Load id2label mapping
# with open(ID2LABEL_PATH, "r", encoding="utf-8") as f:
#     id2label = json.load(f)

# # Prediction function
# def predict_category(text: str):
#     encoded = tokenizer.encode(text)
#     input_ids = np.array([encoded.ids], dtype=np.int64)
#     attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

#     logits = ort_sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
#     predicted_class_id = logits.argmax(axis=1)[0]

#     return id2label[str(predicted_class_id)]

import tflite_runtime.interpreter as tflite
