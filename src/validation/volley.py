from ultralytics import YOLO
import os

# Paths
dataset_path = "src/dataset/Volleyball_v2.v3-1024x768.yolo26/data.yaml"
model_path = "runs/detect/volleyball/weights/best.pt"  # or models/volleyball_final.pt

# Load trained model
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Validate on validation set
print("\n" + "="*50)
print("VALIDATION SET METRICS")
print("="*50)
val_results = model.val(
    data=dataset_path,
    split='val',
    imgsz=1024,
    batch=16,
    save_json=True,
    save_hybrid=True,
    plots=True,
    verbose=True
)

# Print detailed metrics
print(f"\nPrecision: {val_results.box.p.mean():.4f}")
print(f"Recall: {val_results.box.r.mean():.4f}")
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"Fitness: {val_results.fitness:.4f}")

# Per-class metrics
print("\nPer-class metrics:")
for i, class_name in enumerate(val_results.names.values()):
    print(f"{class_name}:")
    print(f"  Precision: {val_results.box.p[i]:.4f}")
    print(f"  Recall: {val_results.box.r[i]:.4f}")
    print(f"  mAP50: {val_results.box.ap50[i]:.4f}")
    print(f"  mAP50-95: {val_results.box.ap[i]:.4f}")

# Test on test set
print("\n" + "="*50)
print("TEST SET METRICS")
print("="*50)
test_results = model.val(
    data=dataset_path,
    split='test',
    imgsz=1024,
    batch=16,
    save_json=True,
    save_hybrid=True,
    plots=True,
    verbose=True
)

print(f"\nPrecision: {test_results.box.p.mean():.4f}")
print(f"Recall: {test_results.box.r.mean():.4f}")
print(f"mAP50: {test_results.box.map50:.4f}")
print(f"mAP50-95: {test_results.box.map:.4f}")
print(f"Fitness: {test_results.fitness:.4f}")

print("\n" + "="*50)
print(f"Results saved to: runs/detect/val")
print("="*50)
