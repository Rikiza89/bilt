# BILT Documentation

Welcome to the BILT documentation.
BILT is a lightweight, CPU-friendly object detection library built on PyTorch
with a fully original FPN-based detection architecture.

---

## Contents

| Guide | Description |
|-------|-------------|
| [Installation](installation.md) | System requirements, pip install, platform notes |
| [Quick Start](quickstart.md) | Get from zero to a running model in 5 minutes |
| [Model Variants](variants.md) | spark / flash / core / pro / max — when to use each |
| [Dataset Preparation](dataset.md) | Label format, directory layout, annotation tools |
| [Training Guide](training.md) | Hyperparameters, callbacks, tips, Trainer API |
| [Inference Guide](inference.md) | All input types, thresholds, batch processing |
| [API Reference](api.md) | Complete reference for every public class and function |
| [Architecture](architecture.md) | Backbone, FPN, detection head, anchor system, losses |

---

## At a glance

```python
from bilt import BILT

# Train
model = BILT("core")
model.train(dataset="data/", epochs=100)

# Predict
detections = model.predict("image.jpg", conf=0.25)

# Annotated output
results = model.predict("images/", return_images=True)
results.save("runs/detect/exp")
```

Model sizes:

| Variant | Backbone | Input | Use case |
|---------|----------|-------|----------|
| `spark` | MobileNetV2 | 320 px | Edge / real-time |
| `flash` | MobileNetV3-Small | 416 px | Fast inference |
| `core` | MobileNetV3-Large | 512 px | General (default) |
| `pro` | ResNet-50 | 640 px | High accuracy |
| `max` | ResNet-101 | 640 px | Maximum accuracy |

---

## License

BILT is copyright © 2024 Rikiza89 and released under the
[GNU Affero General Public License v3.0](../LICENSE).

The detection architecture (FPN neck, detection head, anchor system, focal
loss) is written from scratch. No code from Ultralytics or any other
AGPL-encumbered project is incorporated.
