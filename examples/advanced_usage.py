"""
Advanced usage examples for BILT library.
"""

from ..bilt import BILT
from pathlib import Path

# Example 1: Training with callback
print("="*60)
print("Example 1: Training with custom callback")
print("="*60)

def training_callback(info):
    """Custom callback for training progress."""
    epoch = info['epoch']
    total = info['total_epochs']
    train_loss = info['train_loss']
    val_loss = info['val_loss']
    lr = info['lr']
    
    progress = (epoch / total) * 100
    print(f"[{progress:5.1f}%] Epoch {epoch}/{total} | "
          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")

model = BILT()
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=100,
    batch_size=8,
    img_size=640,
    learning_rate=0.001,
    save_dir="runs/train",
    name="custom_experiment"
)

# Example 2: Inference with custom thresholds
print("\n" + "="*60)
print("Example 2: Custom inference parameters")
print("="*60)

model = BILT("weights/best.pth")

# Low confidence for finding more objects
results_low = model.predict("image.jpg", conf=0.1, iou=0.3)
print(f"Low confidence: {len(results_low)} detections")

# High confidence for precision
results_high = model.predict("image.jpg", conf=0.8, iou=0.5)
print(f"High confidence: {len(results_high)} detections")

# Different image sizes
results_small = model.predict("image.jpg", img_size=320)
results_large = model.predict("image.jpg", img_size=1280)

# Example 3: Save and load models
print("\n" + "="*60)
print("Example 3: Save and load models")
print("="*60)

# Train model
model = BILT()
model.train(dataset="datasets/my_dataset", epochs=10)

# Save to custom location
model.save("models/my_custom_model.pth")

# Load from saved location
loaded_model = BILT("models/my_custom_model.pth")
results = loaded_model.predict("test.jpg")

# Example 4: Processing entire dataset
print("\n" + "="*60)
print("Example 4: Process entire dataset")
print("="*60)

model = BILT("weights/best.pth")
dataset_path = Path("datasets/test_images")

all_results = []
for img_path in dataset_path.glob("*.jpg"):
    results = model.predict(str(img_path), conf=0.3)
    all_results.append({
        'image': img_path.name,
        'detections': results,
        'count': len(results)
    })
    print(f"{img_path.name}: {len(results)} objects")

# Summary
total_detections = sum(r['count'] for r in all_results)
print(f"\nTotal: {len(all_results)} images, {total_detections} detections")

# Example 5: GPU/CPU selection
print("\n" + "="*60)
print("Example 5: Device selection")
print("="*60)

# Use CPU explicitly
model_cpu = BILT("weights/best.pth", device="cpu")

# Use GPU if available
try:
    model_gpu = BILT("weights/best.pth", device="cuda")
    print("Using GPU")
except:
    print("GPU not available, using CPU")

# Auto-detect (default)
model_auto = BILT("weights/best.pth")
print(f"Auto-detected device: {model_auto.device}")

# Example 6: Access model properties
print("\n" + "="*60)
print("Example 6: Model properties")
print("="*60)

model = BILT("weights/best.pth")
print(f"Model: {model}")
print(f"Classes: {model.names}")
print(f"Number of classes: {model.num_classes}")
print(f"Device: {model.device}")