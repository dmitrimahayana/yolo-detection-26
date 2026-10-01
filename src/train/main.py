from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model - dataset will auto-download on first run
# resume=True will continue from last checkpoint in runs/detect/train*/weights/last.pt
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, device="mps", resume=True)