# Training Guide

---

## Basic training

```python
from bilt import BILT

model = BILT("core")          # choose a variant
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=100,
    batch_size=4,
)
```

BILT automatically:
- Loads train and validation splits from `dataset/train/` and `dataset/val/`.
- Reads class names from `data.yaml` if present.
- Initialises the detector with randomly initialised weights — training from scratch on your data.
- **Selects the best available device** — CUDA GPU if present, otherwise CPU.
- Enables `pin_memory` and `non_blocking` transfers when on CUDA.
- Freezes the backbone for the first 5 epochs (warmup), then unfreezes it.
- Saves the best checkpoint (lowest validation loss) to `runs/train/exp/weights/best.pth`.

---

## train() parameters

```python
model.train(
    dataset        = "datasets/my_dataset",
    epochs         = 100,
    batch_size     = 4,
    img_size       = None,          # None = use variant default
    learning_rate  = 5e-4,
    device         = None,          # None = auto (cuda if available, else cpu)
    save_dir       = "runs/train",  # parent directory for run outputs
    name           = "exp",         # sub-directory name (auto-incremented)
    variant        = None,          # override the variant for this run
    workers        = 0,             # dataloader worker processes
)
```

### Return value

`train()` returns a dict:

```python
{
    "variant":          "core",
    "num_epochs":       100,
    "final_train_loss": 0.3412,
    "final_val_loss":   0.3891,
    "best_val_loss":    0.3764,
    "training_time":    4823.1,     # seconds
    "model_path":       "runs/train/exp/weights/best.pth",
}
```

---

## Choosing hyperparameters

### Batch size

| Hardware | Recommended batch size |
|----------|------------------------|
| CPU (8 GB RAM) | 2–4 |
| GPU 4 GB VRAM | 4–8 (`spark`/`flash`) |
| GPU 8 GB VRAM | 8–16 (`core`) or 4–8 (`pro`/`max`) |
| GPU 16+ GB | 16–32 |

Batch size must be ≥ 2 (required for BatchNorm in the backbone). BILT
automatically clamps it to 2 if a lower value is passed.

### Learning rate

The default `5e-4` works well for most cases with AdamW. If training is
unstable (loss spikes), try `2e-4`. If convergence is too slow, try `1e-3`.

```python
model.train(dataset="data/", learning_rate=2e-4)
```

### Epochs

| Dataset size | Recommended epochs |
|-------------|-------------------|
| < 500 images | 50–80 |
| 500–2000 images | 80–150 |
| 2000–10000 images | 100–200 |
| > 10000 images | 50–100 (large datasets converge faster) |

### Image size

```python
# Use the variant's default (recommended)
model.train(dataset="data/")

# Override — smaller = faster, larger = better small-object detection
model.train(dataset="data/", img_size=320)
model.train(dataset="data/", img_size=640)
```

Image size must be divisible by 32. Common choices: 320, 416, 512, 640.

---

## Device selection

BILT checks `torch.cuda.is_available()` inside `Trainer` and uses the GPU
automatically. You only need to pass `device=` when you want to override this.

```python
# Auto-detect — uses CUDA if available, otherwise CPU (recommended)
model = BILT("core")
model.train(dataset="data/")

# Force a specific GPU
model.train(dataset="data/", device="cuda:1")

# Force CPU even on a GPU machine
model.train(dataset="data/", device="cpu")

# Apple Silicon
model.train(dataset="data/", device="mps")
```

You can check which device was selected by looking at the training log:

```
INFO  Trainer using device: cuda
```

Or after training:

```python
print(model.device)   # cuda  or  cpu
```

### GPU performance tips

| Tip | Detail |
|-----|--------|
| Larger batch size | GPU throughput scales well — try 16 or 32 on a 8+ GB card |
| DataLoader workers | Use `workers=2` or `workers=4` on multi-core Linux/macOS |
| Larger variant | `pro` and `max` benefit most from GPU; `spark`/`flash` are already fast on CPU |

---

## Training output structure

```
runs/
└── train/
    ├── exp/                     ← first run
    │   └── weights/
    │       └── best.pth         ← best checkpoint (lowest val loss)
    ├── exp1/                    ← second run (auto-incremented)
    │   └── weights/
    │       └── best.pth
    └── my_experiment/           ← custom name
        └── weights/
            └── best.pth
```

### Custom run names

```python
model.train(
    dataset="data/",
    save_dir="experiments",
    name="resnet50_v2",
)
# saves to experiments/resnet50_v2/weights/best.pth
```

---

## Training callbacks

Use callbacks to monitor training progress, send notifications, or implement
early stopping:

```python
def my_callback(info):
    epoch       = info["epoch"]
    total       = info["total_epochs"]
    train_loss  = info["train_loss"]
    val_loss    = info["val_loss"]
    lr          = info["lr"]

    pct = epoch / total * 100
    print(f"[{pct:5.1f}%] E{epoch}/{total}  "
          f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

    # Example: early stopping
    if val_loss < 0.05:
        raise StopIteration("Target reached!")

from bilt.trainer import Trainer
trainer = Trainer(
    dataset_path="data/",
    num_classes=3,
    class_names=["cat", "dog", "person"],
    variant="core",
)
trainer.train(save_path="weights/best.pth", callback=my_callback)
```

---

## Resuming training

BILT does not have built-in resume support, but you can load a checkpoint
and continue training by implementing a custom loop with `Trainer`:

```python
from bilt import BILT

# Load the existing model
model = BILT("runs/train/exp/weights/best.pth")

# Re-train with more epochs (starts from the loaded weights)
metrics = model.train(
    dataset="data/",
    epochs=50,
    name="exp_resumed",
)
```

Note: the optimiser state is not saved in the checkpoint, so the learning
rate schedule restarts from epoch 0. For precise resume behaviour, use the
`Trainer` class directly and manage state yourself.

---

## Multi-variant comparison

```python
from bilt import BILT

results = {}
for variant in ["spark", "core", "pro"]:
    m = BILT(variant)
    r = m.train(
        dataset="data/",
        epochs=50,
        save_dir="runs/compare",
        name=variant,
    )
    results[variant] = r["best_val_loss"]
    print(f"{variant}: {r['best_val_loss']:.4f}")

best = min(results, key=results.get)
print(f"\nBest variant: {best}")
```

---

## Training on a small dataset (fine-tuning tips)

When you have fewer than 200 images:

1. **Use `spark` or `flash`** — fewer parameters, less overfitting.
2. **Freeze the backbone longer** — you can do this by modifying the warmup
   in `Trainer` (default: 5 epochs, increase to 15–20).
3. **Use a lower learning rate** — try `1e-4` or `5e-5`.
4. **More epochs** — small datasets may need 200+ epochs.

```python
model = BILT("flash")
model.train(
    dataset="data/",
    epochs=200,
    learning_rate=1e-4,
    batch_size=2,
)
```

---

## Training loss components

The training log prints three loss values each batch:

```
Epoch 3/100  batch 12/48  loss=1.2341  cls=0.8102  box=0.4239
```

| Component | Description |
|-----------|-------------|
| `loss` | Total loss (cls + box) |
| `cls` | Focal classification loss |
| `box` | Smooth-L1 regression loss |

Both losses are normalised by the number of positive anchors so the scale
remains roughly constant regardless of batch size.

Typical healthy values:
- `cls` should decrease from ~1.5–2.0 to < 0.5 over training.
- `box` should decrease from ~0.5–1.0 to < 0.3.
- If `box` is near zero but `cls` is stuck high, the model is not finding
  any positive anchor matches — check your label files.

---

## Using the Trainer directly

For full control, bypass `BILT.train()` and use `Trainer` directly:

```python
from bilt.trainer import Trainer
from pathlib import Path

trainer = Trainer(
    dataset_path=Path("data/"),
    num_classes=3,
    class_names=["cat", "dog", "person"],
    batch_size=8,
    learning_rate=3e-4,
    num_epochs=150,
    input_size=640,
    device="cuda",
    variant="pro",
)

# Train one epoch at a time
for epoch in range(trainer.num_epochs):
    train_loss = trainer.train_one_epoch()
    val_loss   = trainer.validate()
    print(f"E{epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}")

trainer.detection_model.save(Path("weights/final.pth"), trainer.class_names)
```
