from ultralytics import YOLO
from pathlib import Path

# Dataset path
dataset_path = "src/dataset/Volleyball_v2.v3-1024x768.yolo26/data.yaml"

# Check for existing checkpoint
checkpoint_path = Path("runs/detect/volleyball/weights/last.pt")

if checkpoint_path.exists():
    print("=" * 50)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print(f"Loading: {checkpoint_path}")
    print("=" * 50)

    # CORRECT WAY: Load checkpoint directly, then call train(resume=True)
    model = YOLO(str(checkpoint_path))  # Load the checkpoint model
    results = model.train(resume=True)   # Resume training - continues from saved epoch

else:
    print("=" * 50)
    print("STARTING FRESH TRAINING")
    print("Loading pretrained: yolo26n.pt")
    print("=" * 50)
    model = YOLO("yolo26n.pt")  # yolo26n, yolo26s, yolo26m, yolo26l, yolo26x

    # Train new model
    results = model.train(
        data=dataset_path,
        epochs=100,
        imgsz=1024,  # Match dataset: 1024x768
        batch=16,
        device="mps",  # Mac M-series GPU
        name="volleyball",  # Creates runs/detect/volleyball
        patience=50,  # Early stopping
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs
        plots=True,
        val=True,  # Validate during training
    )

# Validate on validation set
val_results = model.val()
print(f"\nValidation mAP50: {val_results.box.map50:.4f}")
print(f"Validation mAP50-95: {val_results.box.map:.4f}")

# Test on test set
test_results = model.val(data=dataset_path, split='test')
print(f"\nTest mAP50: {test_results.box.map50:.4f}")
print(f"Test mAP50-95: {test_results.box.map:.4f}")

# Save final model
model.save('/Users/dmitri/Documents/data-engineer/yolo-detection-26/models/volleyball_final.pt')
print(f"\nModel saved to: models/volleyball_final.pt")
