# CLAUDE.md — BILT Library Quick Reference

Fast reference for Claude Code. Read this before touching any file.

---

## Repository Layout

```
bilt/
├── model.py        ← Public API: BILT class, predict(), train(), save(), load()
├── trainer.py      ← Training loop: Trainer class, validate(), EMA, LR warmup
├── core.py         ← BILTDetector (nn.Module), DetectionModel.load(), FPN strides
├── backbone.py     ← BILTBackbone: MobileNetV2/V3, ResNet-50/101 feature extractors
├── neck.py         ← FPNNeck: P3–P6 feature pyramid with lateral connections
├── head.py         ← BILTHead: classification + box prediction convolutions
├── loss.py         ← BILTLoss: Focal loss + Smooth-L1 or CIoU
├── anchors.py      ← Anchor generation and GT matching
├── inferencer.py   ← Inferencer: detect() single image, detect_batch() batched GPU
├── evaluator.py    ← Evaluator: evaluate() against a dataset
├── dataset.py      ← BILTDataset: image loading, mosaic, label parsing
├── variants.py     ← Variant configs: spark/flash/core/pro/max (anchor sizes, FPN ch)
├── augment.py      ← Augmentation pipeline
└── utils.py        ← Misc helpers
```

---

## Architecture Flow

```
Input → BILTBackbone → [C3@stride8, C4@stride16, C5@stride32]
      → FPNNeck      → [P3, P4, P5, P6]   (P6 = stride-2 conv on C5)
      → BILTHead     → cls_preds + box_preds per FPN level
      → Training:    Focal loss + Smooth-L1 (or CIoU)
      → Inference:   box decode + per-class NMS
```

**FPN strides** (defined in `core.py`): `_FPN_STRIDES = [8, 16, 32, 64]`

**Backbone output channels:**
- MobileNet: `[32–40, 48–112, 576–1280]` (variant-dependent)
- ResNet-50:  `[512, 1024, 2048]`
- ResNet-101: `[512, 1024, 2048]`

**Variant configs** (`variants.py`):

| Variant | Backbone | FPN ch | Head convs | Anchors | Input |
|---------|----------|--------|------------|---------|-------|
| spark | MobileNetV2 | 64 | 2 | [32,64,128,256] | 320 |
| flash | MobileNetV3-S | 96 | 3 | [32,64,128,256] | 416 |
| core  | MobileNetV3-L | 128 | 3 | [32,64,128,256] | 512 |
| pro   | ResNet-50 | 256 | 4 | [64,128,256,512] | 640 |
| max   | ResNet-101 | 256 | 4 | [64,128,256,512] | 640 |

---

## Key Classes and Entry Points

### `model.py` — `BILT`
- `BILT(weights, device)` — constructor; `weights` can be variant name or `.pth` path
- `.train(dataset, epochs, batch_size, ...)` — creates `Trainer`, runs training
- `.predict(source, conf, iou, ...)` — single/batch inference
  - Fast path: list of images without `return_images` → `inferencer.detect_batch()`
  - Slow path: `return_images=True` → single `inferencer.detect()` per image
- `._to_pil(img)` — static helper: PIL / np.ndarray / path → PIL Image RGB
- `.save(path)` / `.load(weights)` — float16 checkpoint save/load

### `trainer.py` — `Trainer`
Constructor signature (all params must be passed explicitly — no global config):
```python
Trainer(model, train_dataset, val_dataset, device,
        epochs, batch_size, learning_rate, workers,
        save_dir, warmup_epochs, backbone_lr_mult,
        weight_decay, cos_lr_min, grad_clip,
        focal_alpha, focal_gamma, box_loss_weight,
        augment, flip_prob, color_jitter,
        lr_warmup_epochs,   # ← LambdaLR warmup schedule
        use_ciou,           # ← CIoU vs Smooth-L1
        use_ema,            # ← ModelEMA shadow
        ema_decay,          # ← EMA decay cap (auto-tuned down for small datasets)
        cache_images,       # ← pre-load images into RAM
        mosaic,             # ← 4-image mosaic tiles
        mosaic_prob)        # ← mosaic probability per batch
```

**Critical: `validate()` BN behaviour**
`validate()` calls `self.detection_model.train()` (required for loss branch),
then immediately freezes ALL `nn.BatchNorm2d` layers with `.eval()`.
After the loop it calls `self.detection_model.train()` again to restore.
**Never remove this pattern** — without it, ResNet validation (53+ BN layers)
corrupts `running_mean`/`running_var` with validation batch stats, producing
near-zero confidence scores at inference time.

### `core.py` — `BILTDetector` / `DetectionModel`
- `BILTDetector.forward(images, targets=None)`:
  - `self.training=True` → returns `loss_dict` (total, cls, box)
  - `self.training=False` → returns decoded predictions
- `DetectionModel.load(path, device)` — **`@staticmethod` returning a TUPLE**:
  ```python
  detector, class_names = DetectionModel.load("model.pth")
  # NOT: dm = DetectionModel.load(...); dm.model  ← AttributeError
  ```
  This is a common gotcha — the return type is `(BILTDetector, List[str])`.

### `inferencer.py` — `Inferencer`
- `detect(image: PIL.Image)` → list of dicts with `class_name`, `score`, `bbox`
- `detect_batch(images: List[PIL.Image])` → list of lists (one per image)
  - Stacks images into a single GPU tensor for batched forward pass
  - `BILT.predict()` uses this when `return_images=False` and multiple images passed

### `loss.py` — `BILTLoss`
- CIoU path: decodes both pred and target boxes before geometric loss
- Requires `anchors` kwarg when calling forward with CIoU enabled
- Smooth-L1 path: compares encoded deltas directly (no decode needed)

---

## Critical Bugs Fixed (2026-03)

### Bug 1 — `BILT.train()` dropped 7 advanced parameters (`model.py`)
`BILT.train()` accepted `**kwargs` but 7 parameters were never in the signature:
`lr_warmup_epochs`, `use_ciou`, `use_ema`, `ema_decay`, `cache_images`, `mosaic`,
`mosaic_prob`. They were silently swallowed into `**kwargs` without being forwarded
to `Trainer`. Fixed: all 7 added to signature with defaults and forwarded.

### Bug 2 — BatchNorm stat corruption during validation (`trainer.py`)
`validate()` ran with `model.train()` mode active but inside `torch.no_grad()`.
`torch.no_grad()` does NOT prevent BN from updating `running_mean`/`running_var`.
Every validation call overwrote training-adapted stats with validation-batch stats.
The saved "best" checkpoint contained mixed/corrupted BN stats.
At inference (eval mode uses running stats) → degraded features → near-zero scores.

**ResNet was far more affected**: 53+ BN layers vs 15–17 for MobileNet.
MobileNet survived because fewer layers = milder corruption staying above threshold.
ResNet gave zero detections after NMS.

Fix: freeze BN layers with `.eval()` during validate(), restore after.

---

## Important Patterns

### Adding a new training parameter
1. Add to `Trainer.__init__()` signature with default
2. Store as `self.xxx`
3. Use in `_train_epoch()` or `validate()` as needed
4. Add to `BILT.train()` signature with same default
5. Forward via `Trainer(... xxx=xxx ...)`
6. Update README.md `.train()` API table

### Checkpoint format
Saved as float16 state dict. Includes: `model_state_dict`, `class_names`,
`variant`, `input_size`, `num_classes`. Loaded and upcasted to float32.

### EMA auto-tuning
`Trainer.__init__()` auto-computes EMA decay from dataset size and batch size:
```python
steps_per_epoch = max(1, len(dataset) // batch_size)
auto_decay = max(0.90, min(0.9999, 1.0 - 1.0 / (2 * steps_per_epoch)))
```
If `auto_decay` differs from user `ema_decay` by >0.001, logs a warning.
The auto-tuned value is always used — `ema_decay` acts as documentation/cap.

### LR warmup schedule
Uses `torch.optim.lr_scheduler.LambdaLR`.
Ramps from 10% to 100% of base LR over `lr_warmup_epochs`.
After warmup, cosine annealing takes over.

---

## Where to Look for What

| Question | File |
|----------|------|
| Public API surface | `model.py` |
| Training loop, loss aggregation | `trainer.py` |
| Model forward pass, loss vs inference branch | `core.py` |
| Backbone feature extraction | `backbone.py` |
| FPN lateral connections | `neck.py` |
| Detection head convolutions | `head.py` |
| Focal loss, CIoU, Smooth-L1 | `loss.py` |
| Anchor generation, GT matching | `anchors.py` |
| Single and batch inference | `inferencer.py` |
| Variant hyperparameters | `variants.py` |
| Dataset loading, mosaic | `dataset.py` |


### ⚠️ FOR BUG FIXES AND IMPLEMENTATION ⚠️

**CRITICAL: BEFORE writing any BUG FIX or implementation:**

1. **Read** repository using the Read tool
2. **Study** the exact structure
3. **Use repository files as the LITERAL STARTING POINT** - not just inspiration
4. **Keep all FIXED sections exactly as shown** when no bug fix or implementation is needed

**Avoid:**
- ❌ Creating existing files from scratch
- ❌ Inventing custom sections without user approval

**Follow these practices:**
- ✅ Copy the file's exact structure
