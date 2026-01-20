"""
Basic usage examples for BILT library.
"""

from bilt import BILT

# Example 1: Load model and predict
print("="*60)
print("Example 1: Load model and predict")
print("="*60)

model = BILT("weights/best.pth")
results = model.predict("images/test.jpg", conf=0.25)

print(f"\nDetected {len(results)} objects:")
for i, det in enumerate(results, 1):
    print(f"{i}. {det['class_name']}: {det['score']:.2f} @ {det['bbox']}")

# Example 2: Predict on multiple images
print("\n" + "="*60)
print("Example 2: Batch prediction")
print("="*60)

results = model.predict("images/", conf=0.3, return_images=True)
print(f"\nProcessed {len(results)} images")
results.save("runs/detect/exp")

# Example 3: Train new model
print("\n" + "="*60)
print("Example 3: Train new model")
print("="*60)

model = BILT()
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=50,
    batch_size=4,
    img_size=640
)

print(f"\nTraining complete!")
print(f"Final train loss: {metrics['final_train_loss']:.4f}")
print(f"Final val loss: {metrics['final_val_loss']:.4f}")
print(f"Model saved to: {metrics['model_path']}")

# Example 4: Evaluate model
print("\n" + "="*60)
print("Example 4: Evaluate model")
print("="*60)

metrics = model.evaluate("datasets/my_dataset/val", conf=0.25)
print(f"\nEvaluation results:")
print(f"Total images: {metrics['total_images']}")
print(f"Total predictions: {metrics['total_predictions']}")
print(f"Avg predictions per image: {metrics['avg_predictions_per_image']:.2f}")

# Example 5: Different input types
print("\n" + "="*60)
print("Example 5: Different input types")
print("="*60)

from PIL import Image
import numpy as np

# From path
results1 = model.predict("image.jpg")

# From PIL Image
img = Image.open("image.jpg")
results2 = model.predict(img)

# From numpy array
img_array = np.array(img)
results3 = model.predict(img_array)

# From list
results4 = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])

print("All input types work!")