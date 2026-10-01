from ultralytics import YOLO
import os
import torch
from pathlib import Path

# Auto-detect device: CUDA (g5.2xlarge) > MPS (M4 Mac) > CPU
if torch.cuda.is_available():
    device = "0"  # Use first NVIDIA GPU
    batch_size = 32  # G5.2xlarge has 24GB VRAM
    workers = 8  # G5.2xlarge has 8 vCPUs
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"  # Use Apple Silicon GPU
    batch_size = 16  # M4 shared memory
    workers = 4
else:
    device = "cpu"
    batch_size = 8
    workers = 2

print(f"Using device: {device} | batch: {batch_size} | workers: {workers}")

# Find latest checkpoint in runs/detect/train*/weights/last.pt
checkpoints = sorted(Path("runs/detect").glob("train*/weights/last.pt"), key=os.path.getmtime, reverse=True)
model_path = str(checkpoints[0]) if checkpoints else "yolo26n.pt"

print(f"Loading model from: {model_path}")

# Load model (latest checkpoint or pretrained)
model = YOLO(model_path)

# Train the model - optimized for each hardware
results = model.train(
    data="VisDrone.yaml",
    epochs=100,
    imgsz=640,
    device=device,
    batch=batch_size,
    workers=workers,
    amp=True  # Mixed precision for faster training
)