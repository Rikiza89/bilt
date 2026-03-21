# BILT — Because I Like Twice

**BILT** is a lightweight object detection library built entirely on PyTorch.
It provides a clean, high-level API inspired by modern detection frameworks while remaining
fully independent — using an original FPN-based detection architecture rather than any
third-party detection model.

BILT automatically uses a CUDA GPU when one is available. No configuration needed — just
install a CUDA-enabled PyTorch build and BILT picks it up for both training and inference.

> **License:** GNU Affero General Public License v3.0
> **Copyright:** © 2026 Rikiza89

---

## Features

- **Five model sizes** — *spark / flash / core / pro / max* — each with a different backbone so you can choose the right speed/accuracy trade-off
- **Original architecture** — custom FPN neck + anchor-based detection head + focal loss; no dependency on proprietary detection code
- **Pretrained backbones** — MobileNetV2, MobileNetV3-S/L, ResNet-50/101 (MIT/BSD-licensed, via torchvision)
- **Simple API** — `BILT("core")`, `.train()`, `.predict()`, `.evaluate()`, `.save()`, `.load()`
- **GPU-first** — automatically uses CUDA when available; falls back to CPU seamlessly
- **Pin-memory & non-blocking transfers** — DataLoader pins memory and tensor moves
  use `non_blocking=True` on CUDA, overlapping CPU→GPU transfer with the forward pass
- **Edge-friendly** — works on laptops, Raspberry Pi, and any CPU-only device

---

## Model Variants

| Variant | Backbone | Input | FPN ch | Best for |
|---------|----------|-------|--------|----------|
| `spark` | MobileNetV2 | 320 px | 64 | Embedded / real-time |
| `flash` | MobileNetV3-Small | 416 px | 96 | Edge / fast inference |
| `core` | MobileNetV3-Large | 512 px | 128 | General use (default) |
| `pro` | ResNet-50 | 640 px | 256 | High accuracy |
| `max` | ResNet-101 | 640 px | 256 | Maximum accuracy |

Short aliases are supported: `n / s / m / l / x` and `nano / small / medium / large / xlarge`.

```python
from bilt import BILT, list_variants
list_variants()   # prints the table above
```

---

## Installation

```bash
pip install torch torchvision pillow numpy pyyaml
pip install -e .          # development install from source
```

### Requirements

- Python 3.8+
- torch ≥ 1.10
- torchvision ≥ 0.11
- Pillow ≥ 8.0
- numpy ≥ 1.19
- pyyaml ≥ 5.4

---

## Quick Start

### Inference on a saved model

```python
from bilt import BILT

model = BILT("weights/best.pth")
detections = model.predict("image.jpg", conf=0.25)

for det in detections:
    print(f"{det['class_name']}: {det['score']:.2f}  bbox={det['bbox']}")
```

### Training from scratch

BILT auto-detects CUDA — no `device=` argument needed on a GPU machine.

```python
from bilt import BILT

model = BILT("core")          # MobileNetV3-Large, 512 px

metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=100,
    batch_size=8,             # increase to 16–32 on a GPU
    learning_rate=5e-4,
)
print(metrics)
```

### Choosing a different size

```python
model = BILT("spark")   # nano — fastest
model = BILT("flash")   # small
model = BILT("core")    # medium (default)
model = BILT("pro")     # large
model = BILT("max")     # xlarge — most accurate

# Short aliases also work
model = BILT("n")       # same as spark
model = BILT("m")       # same as core
model = BILT("x")       # same as max
```

### Batch prediction with annotated images

```python
results = model.predict("images/", conf=0.3, return_images=True)
results.save("runs/detect/exp")   # saves annotated images
results.show()                    # displays with matplotlib
```

### Evaluation

```python
metrics = model.evaluate("datasets/my_dataset")
print(f"Avg detections/image: {metrics['avg_predictions_per_image']:.2f}")
```

---

## GPU Acceleration

### Automatic device selection

BILT checks `torch.cuda.is_available()` at startup in `Trainer`, `Inferencer`,
and `Evaluator`. If a CUDA GPU is present it is used automatically — you do not
need to pass `device="cuda"` anywhere.

```python
model = BILT("core")
model.train(dataset="data/", epochs=100)   # uses GPU if available
```

To verify which device is active:

```python
model = BILT("weights/best.pth")
print(model.device)    # cuda  or  cpu
```

### Forcing a specific device

```python
# Force CPU even on a GPU machine
model = BILT("core", device="cpu")
model.train(dataset="data/", device="cpu")

# Force a specific GPU
model = BILT("core", device="cuda:1")

# Apple Silicon
model = BILT("core", device="mps")
```

### What's optimised on CUDA

| Optimisation | Benefit |
|---|---|
| `pin_memory=True` on DataLoader | Faster CPU→GPU page-locked memory copy |
| `non_blocking=True` on `.to(device)` | GPU forward pass overlaps with next batch transfer |
| `persistent_workers=True` when `workers > 0` | Worker processes stay alive between epochs |

### CUDA install

Install the CUDA-enabled PyTorch wheel that matches your driver:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is visible to PyTorch before running BILT:

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))
```

---

## Dataset Format

BILT uses the standard normalised label format compatible with most annotation
tools (LabelImg, Roboflow, CVAT, etc.):

```
<class_id>  <x_center>  <y_center>  <width>  <height>
```

All five values are normalised to `[0, 1]` relative to the image size.

### Expected directory layout

```
dataset/
├── train/
│   ├── images/   *.jpg / *.png / …
│   └── labels/   *.txt
├── val/
│   ├── images/
│   └── labels/
└── data.yaml     (optional but recommended)
```

### data.yaml

```yaml
nc: 3
names: [cat, dog, person]
```

---

## Architecture

```
Input image (H×W)
      │
  ┌───▼────────────────────────┐
  │  BILTBackbone              │  pretrained (MobileNet / ResNet)
  │  C3 (1/8)  C4 (1/16)  C5 (1/32)
  └───┬────────────────────────┘
      │
  ┌───▼────────────────────────┐
  │  FPNNeck  (original)       │
  │  P3    P4    P5    P6      │  all with fpn_channels
  └───┬────────────────────────┘
      │
  ┌───▼────────────────────────┐
  │  BILTHead  (original)      │  shared across all FPN levels
  │  cls_preds  +  box_preds   │
  └───┬────────────────────────┘
      │
  ┌───▼────────────────────────┐    ┌──────────────────────────────┐
  │  Training                  │    │  Inference                   │
  │  Anchor matching           │    │  Box decode + per-class NMS  │
  │  Focal loss + Smooth-L1    │    │  Score filter                │
  └────────────────────────────┘    └──────────────────────────────┘
```

**Training losses**

| Loss | Purpose |
|------|---------|
| Focal loss (α=0.25, γ=2.0) | Classification — handles class imbalance |
| Smooth-L1 | Bounding-box regression |

---

## API Reference

### `BILT(weights=None, device=None)`

| Argument | Description |
|----------|-------------|
| `weights` | Variant name (`"spark"` … `"max"`), a `.pth` checkpoint path, or `None` |
| `device`  | `"cpu"`, `"cuda"`, or `None` (auto) |

### `.predict(source, conf, iou, img_size, return_images)`

| Argument | Default | Description |
|----------|---------|-------------|
| `source` | — | File path, directory, PIL Image, numpy array, or list |
| `conf` | `0.25` | Minimum confidence score |
| `iou` | `0.45` | NMS IoU threshold |
| `img_size` | variant default | Override inference resolution |
| `return_images` | `False` | Return `Results` with annotated images |

### `.train(dataset, epochs, batch_size, img_size, learning_rate, ...)`

| Argument | Default | Description |
|----------|---------|-------------|
| `dataset` | — | Path to dataset root |
| `epochs` | `50` | Training epochs |
| `batch_size` | `4` | Images per batch |
| `img_size` | variant default | Training resolution |
| `learning_rate` | `5e-4` | AdamW learning rate |
| `variant` | `"core"` | Override model variant for this run |
| `save_dir` | `"runs/train"` | Output directory |
| `name` | `"exp"` | Run sub-directory name |

### `.evaluate(dataset, batch_size, conf)`

Returns a dict with `total_images`, `total_predictions`, `total_ground_truth`,
`avg_predictions_per_image`, `avg_ground_truth_per_image`.

### `.save(path)` / `.load(weights)`

Checkpoints include the variant name, class names, input size and full model
state so a single file is sufficient to restore a complete model.

### `BILT.variants()`

Prints a summary of all five variants (static method, callable on the class).

---

## Legal

BILT is an original work by **Rikiza89**, released under the
**GNU Affero General Public License v3.0**.

- The detection architecture (FPN neck, detection head, anchor matching,
  focal loss, smooth-L1) is written from scratch and is not derived from any
  other project.
- Pretrained backbone weights are loaded from **torchvision** (BSD/MIT licensed).
- No code from any other AGPL-encumbered project is incorporated.

See [LICENSE](LICENSE) for the full license text.

---

## Roadmap

- [ ] mAP evaluation (COCO-style)
- [ ] Data augmentation pipeline (mosaic, flips, colour jitter)
- [ ] ONNX and TensorRT export
- [ ] Mixed-precision (fp16) training
- [ ] Multi-GPU training
- [ ] Pre-trained BILT checkpoints (spark / flash / core trained on COCO)
- [ ] Web demo (Gradio)
