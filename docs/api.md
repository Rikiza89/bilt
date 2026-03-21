# API Reference

Complete reference for all public classes and functions.

---

## `bilt.BILT`

Main interface for object detection.

```python
from bilt import BILT

model = BILT(weights=None, device=None)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | str \| Path \| None | `None` | Variant name (`"spark"`, `"flash"`, `"core"`, `"pro"`, `"max"`, or aliases), a `.pth` checkpoint path, or `None` |
| `device` | str \| None | `None` | `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, or `None` for auto-detect |

**Examples:**

```python
BILT()                          # uninitialized, default device
BILT("core")                    # medium variant, no weights yet
BILT("m")                       # same as above (alias)
BILT("runs/train/exp/best.pth") # load saved model
BILT("best.pth", device="cuda") # load on GPU
```

---

### `.train()`

```python
metrics = model.train(
    dataset,
    epochs=50,
    batch_size=4,
    img_size=None,
    learning_rate=5e-4,
    device=None,
    save_dir="runs/train",
    name="exp",
    variant=None,
    workers=0,
)
```

Train a new detector from scratch on the user's dataset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str \| Path | **required** | Root directory containing `train/` and `val/` |
| `epochs` | int | `50` | Total training epochs |
| `batch_size` | int | `4` | Images per batch (minimum 2) |
| `img_size` | int \| None | `None` | Square input resolution; `None` uses the variant's default |
| `learning_rate` | float | `5e-4` | Initial AdamW learning rate |
| `device` | str \| None | `None` | Override device for this run |
| `save_dir` | str \| Path | `"runs/train"` | Parent directory for outputs |
| `name` | str | `"exp"` | Sub-directory name (auto-incremented if it already exists) |
| `variant` | str \| None | `None` | Override the variant name for this run |
| `workers` | int | `0` | DataLoader worker processes |

**Returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `variant` | str | Variant name used |
| `num_epochs` | int | Total epochs trained |
| `final_train_loss` | float | Training loss at the last epoch |
| `final_val_loss` | float | Validation loss at the last epoch |
| `best_val_loss` | float | Lowest validation loss achieved |
| `training_time` | float | Total training time in seconds |
| `model_path` | str | Path to the saved best checkpoint |

---

### `.predict()`

```python
result = model.predict(
    source,
    conf=0.25,
    iou=0.45,
    img_size=None,
    return_images=False,
)
```

Run object detection on one or more images.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str \| Path \| PIL.Image \| np.ndarray \| list | **required** | Input image(s) |
| `conf` | float | `0.25` | Minimum confidence threshold in [0, 1] |
| `iou` | float | `0.45` | NMS IoU threshold in [0, 1] |
| `img_size` | int \| None | `None` | Override inference resolution |
| `return_images` | bool | `False` | Return a `Results` object with annotated images |

**Returns:**

- Single image + `return_images=False` → `List[Dict]`
- Multiple images or `return_images=True` → `Results`

**Detection dict:**

```python
{
    "class_name": str,   # human-readable label
    "class_id":   int,   # 1-indexed class index
    "score":      float, # confidence in [0, 1]
    "bbox":       list,  # [x1, y1, x2, y2] absolute pixels
}
```

---

### `.evaluate()`

```python
metrics = model.evaluate(
    dataset,
    batch_size=4,
    conf=0.25,
)
```

Evaluate the model on a validation split.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str \| Path | **required** | Dataset root or `val/` subdirectory |
| `batch_size` | int | `4` | Images per evaluation batch |
| `conf` | float | `0.25` | Confidence threshold for counting detections |

**Returns:** dict with keys:

| Key | Description |
|-----|-------------|
| `total_images` | Total images evaluated |
| `total_predictions` | Total detections above threshold |
| `total_ground_truth` | Total ground-truth boxes |
| `avg_predictions_per_image` | Mean detections per image |
| `avg_ground_truth_per_image` | Mean GT boxes per image |
| `confidence_threshold` | The threshold used |

---

### `.save()`

```python
model.save(path)
```

Save the model checkpoint (weights + metadata) to `path`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str \| Path | Output file path (`.pth` extension recommended) |

The checkpoint stores: model weights, class names, variant name, and
input size. Everything needed to restore the model is included.

---

### `.load()`

```python
model = model.load(weights)
```

Load a checkpoint. Returns `self` for method chaining.

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | str \| Path | Path to a `.pth` checkpoint |

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model.variant` | str | Active variant name (`"core"`, etc.) |
| `model.num_classes` | int \| None | Number of detection classes |
| `model.names` | List[str] | Class name list |
| `model.device` | torch.device | Active device |

---

### Static methods

```python
BILT.variants()   # print variant summary table (no instance needed)
```

---

## `bilt.Results`

Container for batch inference results with optional annotated images.
Returned by `model.predict(..., return_images=True)`.

```python
results = model.predict("images/", return_images=True)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `results.detections` | List[List[Dict]] | All detection lists |
| `results.images` | List[PIL.Image] | Annotated images |
| `results.class_names` | List[str] | Class name list |

### Methods

```python
len(results)             # number of images
results[i]               # detection list for image i
results.save(save_dir)   # save annotated images to save_dir/
results.show()           # display with matplotlib
```

---

## `bilt.list_variants()`

```python
from bilt import list_variants
list_variants()
```

Prints a formatted table of all available model variants.

---

## `bilt.get_variant_config()`

```python
from bilt import get_variant_config

cfg = get_variant_config("pro")
```

Returns a copy of the configuration dict for the given variant name or alias.

| Key | Type | Description |
|-----|------|-------------|
| `backbone` | str | Backbone model name |
| `input_size` | int | Default input resolution |
| `fpn_channels` | int | FPN output channel width |
| `head_num_convs` | int | Convolutional layers in detection head |
| `anchor_sizes` | list | Anchor base sizes per FPN level |
| `anchor_aspect_ratios` | tuple | Aspect ratios per anchor |
| `description` | str | Human-readable variant description |

---

## `bilt.VARIANT_CONFIGS`

```python
from bilt import VARIANT_CONFIGS

print(VARIANT_CONFIGS.keys())
# dict_keys(['spark', 'flash', 'core', 'pro', 'max'])
```

Read-only dict of all variant configurations.

---

## `bilt.set_logging_level()`

```python
from bilt import set_logging_level

set_logging_level("WARNING")   # suppress INFO logs
set_logging_level("DEBUG")     # verbose output
set_logging_level("ERROR")     # errors only
```

Valid levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

---

## Low-level classes

These are used internally but are available for advanced use cases.

### `bilt.core.BILTDetector`

```python
from bilt.core import BILTDetector

detector = BILTDetector(
    variant="core",
    num_classes=3,
)
```

Raw `nn.Module` implementing the full detection pipeline. In training mode,
`forward(images, targets)` returns a loss dict. In eval mode, `forward(images)`
returns a list of detection dicts.

### `bilt.core.DetectionModel`

Convenience wrapper around `BILTDetector` with `save()`, `load()`, `to()`,
`train()`, and `eval()` helpers. Used internally by `Trainer`.

### `bilt.trainer.Trainer`

```python
from bilt.trainer import Trainer

trainer = Trainer(
    dataset_path="data/",
    num_classes=3,
    class_names=["cat", "dog", "person"],
    batch_size=8,
    learning_rate=3e-4,
    num_epochs=100,
    input_size=640,
    device="cuda",
    variant="pro",
)
metrics = trainer.train(save_path="weights/best.pth", callback=fn)
```

See [Training Guide](training.md) for full usage.

### `bilt.inferencer.Inferencer`

```python
from bilt.inferencer import Inferencer

inf = Inferencer(
    model=detector,          # BILTDetector in eval mode
    class_names=["cat"],
    confidence_threshold=0.25,
    nms_threshold=0.45,
    input_size=512,
    device="cpu",
)
dets = inf.detect(pil_image)
dets = inf.detect_batch([img1, img2])
dets = inf.detect_from_path("photo.jpg")
```

### `bilt.dataset.ObjectDetectionDataset`

```python
from bilt.dataset import ObjectDetectionDataset, get_transforms

ds = ObjectDetectionDataset(
    images_dir="data/train/images",
    labels_dir="data/train/labels",
    transforms=get_transforms(512, training=True),
    input_size=512,
)
img_tensor, target = ds[0]
# target = {"boxes": Tensor(N,4), "labels": Tensor(N,)}
```

### `bilt.anchors` module

```python
from bilt.anchors import (
    AnchorGenerator,  # generates anchor boxes for FPN levels
    AnchorMatcher,    # assigns GT boxes to anchors
    encode_boxes,     # GT → delta encoding
    decode_boxes,     # delta → absolute boxes
    box_iou,          # pairwise IoU matrix
)
```

### `bilt.loss.BILTLoss`

```python
from bilt.loss import BILTLoss, sigmoid_focal_loss, smooth_l1_loss

criterion = BILTLoss(
    num_classes=3,
    alpha=0.25,       # focal loss alpha
    gamma=2.0,        # focal loss gamma
    box_weight=1.0,   # relative weight of regression loss
)
losses = criterion(cls_preds, box_preds, cls_targets, box_targets, pos_mask)
# losses = {"total": scalar, "cls": scalar, "box": scalar}
```
