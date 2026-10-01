from ultralytics import YOLO
import os
from pathlib import Path

# Find latest checkpoint in runs/detect/train*/weights/last.pt
checkpoints = sorted(Path("runs/detect").glob("train*/weights/last.pt"), key=os.path.getmtime, reverse=True)
model_path = str(checkpoints[0]) if checkpoints else "yolo26n.pt"

print(f"Loading model from: {model_path}")

# Load model (latest checkpoint or pretrained)
model = YOLO(model_path)

# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, device="mps")