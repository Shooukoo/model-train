import torch
from ultralytics import YOLO

# Load the model
model_path = '/home/shooxd/Documentos/model-train/runs/classify/runs_zarzamora/modelo_v8/weights/best.pt'
try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
    print("Classes:")
    print(model.names)
except Exception as e:
    print(f"Error loading model: {e}")
