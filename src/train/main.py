from ultralytics import YOLO

# Load a model - resume from checkpoint or pretrained
model = YOLO("runs/detect/train-2/weights/last.pt")  # resume from last VisDrone training

# Train the model - will continue from checkpoint
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, device="mps")