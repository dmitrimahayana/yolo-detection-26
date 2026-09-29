from ultralytics import YOLO
import os
from pathlib import Path

# Paths
model_path = "runs/detect/volleyball/weights/best.pt"  # or models/volleyball_final.pt
test_images_dir = "src/dataset/Volleyball_v2.v3-1024x768.yolo26/test/images"
output_dir = "runs/predict/volleyball"

# Load trained model
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Predict on test images
print(f"\nRunning predictions on: {test_images_dir}")
print(f"Output directory: {output_dir}")

results = model.predict(
    source=test_images_dir,
    imgsz=1024,
    conf=0.25,  # Confidence threshold
    iou=0.45,   # NMS IoU threshold
    save=True,  # Save predictions
    save_txt=True,  # Save labels
    save_conf=True,  # Save confidence in labels
    save_crop=True,  # Save cropped predictions
    project="runs/predict",
    name="volleyball",
    exist_ok=True,
    visualize=False,
    device="mps",
    verbose=True
)

# Summary statistics
print("\n" + "="*50)
print("PREDICTION SUMMARY")
print("="*50)
total_images = len(results)
total_detections = sum(len(r.boxes) for r in results)
print(f"Total images processed: {total_images}")
print(f"Total detections: {total_detections}")
print(f"Average detections per image: {total_detections/total_images:.2f}")

# Sample predictions
print("\nFirst 5 predictions:")
for i, result in enumerate(results[:5]):
    img_path = Path(result.path).name
    num_boxes = len(result.boxes)
    if num_boxes > 0:
        confidences = result.boxes.conf.cpu().numpy()
        avg_conf = confidences.mean()
        max_conf = confidences.max()
        print(f"{i+1}. {img_path}: {num_boxes} detections, avg conf={avg_conf:.3f}, max conf={max_conf:.3f}")
    else:
        print(f"{i+1}. {img_path}: No detections")

print("\n" + "="*50)
print(f"Predictions saved to: {output_dir}")
print("  - labels/: Detection labels (.txt)")
print("  - crops/: Cropped detections")
print("  - Images with bounding boxes")
print("="*50)

# Predict on single image (example)
print("\n" + "="*50)
print("SINGLE IMAGE PREDICTION EXAMPLE")
print("="*50)
single_image = Path(test_images_dir) / os.listdir(test_images_dir)[0]
print(f"Testing on: {single_image.name}")

single_result = model.predict(
    source=str(single_image),
    imgsz=1024,
    conf=0.25,
    save=True,
    project="runs/predict",
    name="volleyball_single",
    exist_ok=True
)

for r in single_result:
    print(f"Detections: {len(r.boxes)}")
    if len(r.boxes) > 0:
        for i, box in enumerate(r.boxes):
            conf = box.conf.item()
            cls = int(box.cls.item())
            cls_name = r.names[cls]
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"  {i+1}. {cls_name} (conf={conf:.3f}) at [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
