# Architecture

This document describes BILT's internal detection architecture for users who
want to understand, extend, or audit the code.

---

## Overview

```
Input image  (B, 3, H, W)  — normalised
       │
 ┌─────▼──────────────────────────────────────────────┐
 │  BILTBackbone                                       │
 │  Feature extractor (MobileNet / ResNet, random init)│
 │                                                     │
 │  C3 (stride 8,  ~H/8  × W/8 )                      │
 │  C4 (stride 16, ~H/16 × W/16)                      │
 │  C5 (stride 32, ~H/32 × W/32)                      │
 └─────┬──────────────────────────────────────────────┘
       │  [C3, C4, C5]
 ┌─────▼──────────────────────────────────────────────┐
 │  FPNNeck  (Feature Pyramid Network)                 │
 │                                                     │
 │  Top-down pathway with lateral connections:         │
 │    P5 = Conv(C5)                                    │
 │    P4 = Conv(C4) + Upsample(P5)                     │
 │    P3 = Conv(C3) + Upsample(P4)                     │
 │    P6 = Stride-2 conv on C5  (large objects)        │
 │                                                     │
 │  All levels: fpn_channels wide (64–256 per variant) │
 └─────┬──────────────────────────────────────────────┘
       │  [P3, P4, P5, P6]
 ┌─────▼──────────────────────────────────────────────┐
 │  BILTHead  (shared across all FPN levels)           │
 │                                                     │
 │  Classification tower  →  (B, A·C, H, W)            │
 │  Regression tower      →  (B, A·4, H, W)            │
 │                                                     │
 │  A = anchors per location  (3 by default)           │
 │  C = num_classes                                    │
 └─────┬──────────────────────────────────────────────┘
       │
 ┌─────▼──────────────────────────────────────────────┐    training
 │  AnchorGenerator                                    │ ──────────▶ BILTLoss
 │  AnchorMatcher                                      │    (cls + box targets)
 │  encode_boxes / decode_boxes                        │
 └─────────────────────────────────────────────────────┘
                                                           inference
                                                       ──────────────▶ [boxes,
                                                           NMS           scores,
                                                                         labels]
```

---

## Backbone

`bilt/backbone.py`

Wraps five torchvision backbone architectures to expose three multi-scale
feature maps. All weights are randomly initialised and trained from scratch
on the user's dataset. Each backbone is split at fixed layer boundaries to
produce C3, C4, C5 at stride 8, 16, 32 respectively.

| Variant | Backbone | C3 ch | C4 ch | C5 ch |
|---------|----------|-------|-------|-------|
| spark | MobileNetV2 | 32 | 96 | 1280 |
| flash | MobileNetV3-Small | 24 | 48 | 576 |
| core | MobileNetV3-Large | 40 | 112 | 960 |
| pro | ResNet-50 | 512 | 1024 | 2048 |
| max | ResNet-101 | 512 | 1024 | 2048 |

**Head warmup:** The backbone is frozen for the first 5 training epochs
so the randomly-initialised FPN and detection head can stabilise before
full end-to-end training begins.

```python
backbone.freeze()    # freeze all backbone parameters
backbone.unfreeze()  # unfreeze all backbone parameters
```

---

## FPN Neck

`bilt/neck.py`

Implements a standard top-down Feature Pyramid Network:

```
C5 ──────────── lat5 ─────────────────────── P5
                    └──(upsample)──▶ lat4 ── P4
C4 ─────────────────────────────── lat4 ─┘
                                       └──(upsample)──▶ lat3 ── P3
C3 ──────────────────────────────────────────────── lat3 ─┘
C5 ──(stride-2 conv)──────────────────────────────────────────── P6
```

Lateral projections use **1×1 convolutions** to reduce backbone channels to
`fpn_channels`. Output convolutions use **3×3 Conv-BN-ReLU** to smooth
artefacts from the nearest-neighbour upsampling.

The extra **P6 level** is a stride-2 convolution applied directly to C5. It
produces a 1/64 scale feature map, giving the model the ability to anchor
to very large objects.

---

## Detection Head

`bilt/head.py`

A single head with two branches is shared across **all four FPN levels**
(P3–P6). Sharing weights regularises training and ensures consistent
predictions across scales.

Each branch is a tower of `num_convs` Conv-GroupNorm-ReLU blocks followed
by a 1×1 output projection:

```
Input feature (fpn_channels)
    ↓
[Conv 3×3 → GroupNorm → ReLU] × num_convs
    ↓
Conv 1×1 → cls predictions  (A × num_classes per location)
         → box predictions  (A × 4             per location)
```

**GroupNorm** is used instead of BatchNorm so the head works correctly at
batch size 1 (inference) without special-casing. Groups = min(32, C/4).

**Prior bias initialisation:** The final classification layer's bias is
initialised so that `sigmoid(bias) ≈ 0.01`. This prevents the model from
flooding itself with false positives at the start of training (the focal
loss trick from RetinaNet).

---

## Anchor System

`bilt/anchors.py`

### AnchorGenerator

Generates a grid of anchor boxes for each FPN level. Each spatial location
gets `num_anchors` anchors (one per aspect ratio).

**Default configuration:**

| FPN level | Stride | Base size | Aspect ratios |
|-----------|--------|-----------|---------------|
| P3 | 8 | 32 px | 0.5, 1.0, 2.0 |
| P4 | 16 | 64 px | 0.5, 1.0, 2.0 |
| P5 | 32 | 128 px | 0.5, 1.0, 2.0 |
| P6 | 64 | 256 px | 0.5, 1.0, 2.0 |

Anchor centres are at the centre of each stride cell:
`x = (col + 0.5) × stride`,  `y = (row + 0.5) × stride`.

For a 512 × 512 input:

| Level | Grid size | Anchors |
|-------|-----------|---------|
| P3 | 64 × 64 | 64 × 64 × 3 = 12 288 |
| P4 | 32 × 32 | 32 × 32 × 3 = 3 072 |
| P5 | 16 × 16 | 16 × 16 × 3 = 768 |
| P6 | 8 × 8 | 8 × 8 × 3 = 192 |
| **Total** | | **16 320** |

### AnchorMatcher

Assigns ground-truth boxes to anchors based on pairwise IoU:

- IoU ≥ 0.5 → **positive** (assigned the GT class)
- IoU < 0.4 → **negative** (background, class 0)
- 0.4 ≤ IoU < 0.5 → **ignore** (excluded from loss)

Every GT box is additionally force-matched to its highest-IoU anchor,
guaranteeing that every object contributes to the gradient.

### Delta encoding

Positive anchors regress to GT boxes via a delta parameterisation:

```
dx = (gx - ax) / aw
dy = (gy - ay) / ah
dw = log(gw / aw)
dh = log(gh / ah)
```

where `(ax, ay, aw, ah)` are the anchor centre and size. Inverse of this
encoding is applied at inference time to recover absolute boxes.

---

## Loss Functions

`bilt/loss.py`

### Sigmoid focal loss (classification)

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

- **α = 0.25** down-weights the contribution of well-classified negatives.
- **γ = 2.0** makes the model focus on hard examples by reducing the loss
  contribution of easy ones.
- Applied to all non-ignored anchors.
- One-vs-all binary classification per class (no softmax).

### Smooth-L1 loss (regression)

```
L(x) = 0.5 x² / β        if |x| < β
       |x| - 0.5 β        otherwise
```

β = 0.1 (default). Applied only to **positive** anchors.

### Normalisation

Both losses are divided by the number of positive anchors in the batch
(clamped to ≥ 1), keeping the loss scale stable as batch size and object
density vary.

---

## Inference decoding

At inference time:

1. Apply sigmoid to classification logits → class probabilities.
2. Decode box deltas → absolute `[x1, y1, x2, y2]` boxes (inverse of encoding).
3. Clip boxes to image bounds.
4. Filter: keep only boxes with max-class score > `score_threshold` (0.05 internal).
5. Per-class NMS with `nms_iou_threshold` (user-controlled via the `iou=` argument).
6. Sort by score, keep top `max_detections` (300 by default).
7. Scale boxes from model-input space to original image space.

---

## Image preprocessing

All images are:
1. Converted to RGB.
2. Resized to the variant's `input_size × input_size` (nearest-neighbour for training, bilinear for inference via `T.Resize`).
3. Converted to float tensor `[0, 1]`.
4. Normalised: `(x - mean) / std`:
   - mean = `[0.485, 0.456, 0.406]`
   - std  = `[0.229, 0.224, 0.225]`

---

## File map

```
bilt/
├── variants.py     Variant name → config dict
├── backbone.py     BILTBackbone (wraps MobileNet/ResNet)
├── neck.py         FPNNeck (top-down FPN + P6)
├── head.py         BILTHead (cls tower + reg tower)
├── anchors.py      AnchorGenerator, AnchorMatcher, encode/decode
├── loss.py         BILTLoss (focal + smooth-L1)
├── core.py         BILTDetector (assembles all of the above)
│                   DetectionModel (save/load wrapper)
├── model.py        BILT (high-level API)
├── trainer.py      Trainer (training loop)
├── inferencer.py   Inferencer (preprocessing + postprocessing)
├── dataset.py      ObjectDetectionDataset, DataLoader factory
├── evaluator.py    Evaluator (basic detection statistics)
└── utils.py        Logging, label parsing, NMS, drawing helpers
```
