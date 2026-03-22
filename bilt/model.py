# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the GNU Affero General Public License v3.0

"""
BILT – main high-level API.

Usage
-----
    from bilt import BILT

    # Create or load a model by variant name
    model = BILT("core")           # MobileNetV3-Large, 512 px
    model = BILT("spark")          # MobileNetV2, 320 px (fastest)
    model = BILT("pro")            # ResNet-50, 640 px

    # Load a previously saved model
    model = BILT("weights/best.pth")

    # Train
    metrics = model.train(dataset="datasets/my_data", epochs=50)

    # Infer
    results = model.predict("image.jpg", conf=0.25)

    # Evaluate
    metrics = model.evaluate("datasets/my_data")

    # Save
    model.save("runs/exp/weights/best.pth")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .core import DetectionModel
from .evaluator import Evaluator
from .inferencer import Inferencer
from .trainer import Trainer
from .utils import get_logger
from .variants import DEFAULT_VARIANT, is_variant_name, list_variants

logger = get_logger(__name__)


class BILT:
    """
    BILT (Because I Like Twice) — object detection library.

    Parameters
    ----------
    weights : str or Path, optional
        Either:
        - A variant name: ``"spark"``, ``"flash"``, ``"core"``, ``"pro"``,
          ``"max"`` (and short aliases ``n/s/m/l/x``).
        - A path to a ``.pth`` checkpoint produced by :meth:`save`.
        - ``None`` – defaults to the ``"core"`` variant.
    device : str, optional
        ``"cpu"``, ``"cuda"``, or ``None`` for auto-detect.
    """

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = self._resolve_device(device)
        self.model: Optional[torch.nn.Module] = None
        self.class_names: Optional[List[str]] = None
        self.num_classes: Optional[int] = None
        self.inferencer: Optional[Inferencer] = None
        self._variant: str = DEFAULT_VARIANT

        if weights is None:
            # Uninitialized – will be fully set up in train()
            logger.info(
                f"BILT created without weights. "
                f"Call train() to build a model, or pass a variant name."
            )
        elif isinstance(weights, (str, Path)) and is_variant_name(str(weights)):
            # Variant name or alias supplied  e.g.  BILT("spark") or BILT("n")
            from .variants import VARIANT_ALIASES
            key = str(weights).lower().strip()
            self._variant = VARIANT_ALIASES.get(key, key)
            logger.info(
                f"BILT variant '{self._variant}' selected. "
                f"Call train() to train from scratch, or save/load weights."
            )
        else:
            # Treat as a checkpoint path
            self.load(weights)

    # ---------------------------------------------------------------------- #
    # Inference                                                               #
    # ---------------------------------------------------------------------- #

    def predict(
        self,
        source: Union[str, Path, "Image.Image", np.ndarray, List],
        conf: float = 0.15,
        iou: float = 0.45,
        img_size: Optional[int] = None,
        return_images: bool = False,
    ) -> Union[List[Dict], "Results"]:
        """
        Run object detection on one or more images.

        Parameters
        ----------
        source       : file path, directory path, PIL Image, numpy array,
                       or a list of any of the above.
        conf         : minimum confidence score to keep a detection.
        iou          : NMS IoU threshold.
        img_size     : override inference resolution (defaults to variant size).
        return_images: return a :class:`Results` object that includes annotated
                       images instead of raw detection dicts.

        Returns
        -------
        Single image  → list of detection dicts
        Multiple / return_images=True  → :class:`Results`
        """
        if self.inferencer is None:
            raise RuntimeError(
                "No model loaded. Call load() or train() first."
            )

        self.inferencer.confidence_threshold = conf
        self.inferencer.nms_threshold = iou
        if img_size is not None:
            self.inferencer.input_size = img_size

        images = self._prepare_source(source)

        all_detections: List[List[Dict]] = []
        original_images: List[Optional[Image.Image]] = []

        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
                original_images.append(pil_img if return_images else None)
                detections = self.inferencer.detect(pil_img)
            elif isinstance(img, Image.Image):
                original_images.append(img if return_images else None)
                detections = self.inferencer.detect(img)
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
                original_images.append(pil_img if return_images else None)
                detections = self.inferencer.detect(pil_img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            all_detections.append(detections)

        if len(all_detections) == 1 and not return_images:
            return all_detections[0]

        if return_images:
            return Results(all_detections, original_images, self.class_names)

        return all_detections

    # ---------------------------------------------------------------------- #
    # Training                                                                #
    # ---------------------------------------------------------------------- #

    def train(
        self,
        dataset: Union[str, Path],
        epochs: int = 50,
        batch_size: int = 4,
        img_size: Optional[int] = None,
        learning_rate: float = 2e-3,
        device: Optional[str] = None,
        save_dir: Union[str, Path] = "runs/train",
        name: str = "exp",
        variant: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train an object detection model on a BILT-format dataset.

        Parameters
        ----------
        dataset       : Root directory containing train/ and val/ splits.
        epochs        : Number of training epochs.
        batch_size    : Images per batch (minimum 2 for BatchNorm).
        img_size      : Input resolution; defaults to the variant's setting.
        learning_rate : Initial learning rate for AdamW.
        device        : Override device (e.g. ``"cuda"``).
        save_dir      : Parent directory for training run outputs.
        name          : Run sub-directory name (auto-incremented if exists).
        variant       : Override the variant to train (e.g. ``"pro"``).
        **kwargs      : Extra options passed to :class:`~bilt.trainer.Trainer`
                        (e.g. ``workers=2``).

        Returns
        -------
        dict with training metrics.
        """
        dataset = Path(dataset)
        save_dir = Path(save_dir)

        if device:
            self.device = torch.device(device)

        # Determine variant
        active_variant = variant or self._variant or DEFAULT_VARIANT

        # Batch size guard
        if batch_size < 2:
            logger.warning("batch_size < 2 is not recommended. Setting to 2.")
            batch_size = 2

        # ------------------------------------------------------------------ #
        # Read dataset class info                                             #
        # ------------------------------------------------------------------ #
        from .dataset import ObjectDetectionDataset, get_transforms

        train_ds = ObjectDetectionDataset(
            images_dir=dataset / "train" / "images",
            labels_dir=dataset / "train" / "labels",
            transforms=get_transforms(img_size or 512, training=True),
            input_size=img_size or 512,
        )

        yaml_path = dataset / "data.yaml"
        if not yaml_path.exists():
            for alt in [dataset / "data.yml", dataset / "dataset.yaml"]:
                if alt.exists():
                    yaml_path = alt
                    break

        class_names = train_ds.get_class_names(
            yaml_path if yaml_path.exists() else None
        )
        num_classes = train_ds.num_classes
        logger.info(f"Dataset: {num_classes} classes → {class_names}")

        # ------------------------------------------------------------------ #
        # Create run directory                                                #
        # ------------------------------------------------------------------ #
        run_dir = save_dir / name
        counter = 1
        while run_dir.exists():
            run_dir = save_dir / f"{name}{counter}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "weights" / "best.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Train                                                               #
        # ------------------------------------------------------------------ #
        trainer = Trainer(
            dataset_path=dataset,
            num_classes=num_classes,
            class_names=class_names,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
            num_workers=kwargs.get("workers", 0),
            input_size=img_size,
            device=self.device,
            variant=active_variant,
        )

        results = trainer.train(model_path)

        # Load the best checkpoint
        self.load(model_path)
        logger.info(f"Training complete. Best model at {model_path}")

        return results

    # ---------------------------------------------------------------------- #
    # Evaluation                                                              #
    # ---------------------------------------------------------------------- #

    def evaluate(
        self,
        dataset: Union[str, Path],
        batch_size: int = 4,
        conf: float = 0.25,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the loaded model on a validation split.

        Parameters
        ----------
        dataset    : Path to dataset root or val/ subdirectory.
        batch_size : Images to evaluate per batch.
        conf       : Confidence threshold for counting detections.

        Returns
        -------
        dict with evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() or train() first.")

        dataset = Path(dataset)

        if (dataset / "val" / "images").exists():
            images_dir = dataset / "val" / "images"
            labels_dir = dataset / "val" / "labels"
        elif (dataset / "images").exists():
            images_dir = dataset / "images"
            labels_dir = dataset / "labels"
        else:
            raise ValueError(f"Could not locate images directory in {dataset}")

        evaluator = Evaluator(
            model=self.model,
            class_names=self.class_names,
            device=self.device,
        )
        return evaluator.evaluate_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            batch_size=batch_size,
            confidence_threshold=conf,
        )

    # ---------------------------------------------------------------------- #
    # Save / Load                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model weights and metadata to *path*.

        Parameters
        ----------
        path : File path ending in ``.pth``.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        half_sd = {
            k: v.half() if v.is_floating_point() else v
            for k, v in self.model.state_dict().items()
        }
        checkpoint = {
            "model_state_dict": half_sd,
            "storage_dtype":    "float16",
            "num_classes":      self.num_classes,
            "class_names":      self.class_names,
            "variant":          getattr(self.model, "variant", self._variant),
            "input_size":       getattr(self.model, "input_size", 512),
            "architecture":     "bilt_fpn",
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, weights: Union[str, Path]) -> "BILT":
        """
        Load a checkpoint produced by :meth:`save`.

        Parameters
        ----------
        weights : Path to ``.pth`` file.

        Returns
        -------
        self  (for chaining).
        """
        self.model, self.class_names = DetectionModel.load(weights, str(self.device))
        self.num_classes = len(self.class_names)
        self.model.to(self.device)
        self.model.eval()
        self._variant = getattr(self.model, "variant", DEFAULT_VARIANT)

        self.inferencer = Inferencer(
            model=self.model,
            class_names=self.class_names,
            device=self.device,
            input_size=getattr(self.model, "input_size", 512),
        )
        logger.info(
            f"Loaded BILT-{self._variant} "
            f"({self.num_classes} classes) on {self.device}"
        )
        return self

    # ---------------------------------------------------------------------- #
    # Properties / dunder                                                     #
    # ---------------------------------------------------------------------- #

    @property
    def names(self) -> List[str]:
        """Class names list."""
        return self.class_names or []

    @property
    def variant(self) -> str:
        """Active variant name."""
        return self._variant

    @staticmethod
    def variants() -> None:
        """Print a summary of all available model variants."""
        list_variants()

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_source(self, source) -> List:
        if isinstance(source, list):
            return source
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                imgs = []
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    imgs.extend(p.glob(f"*{ext}"))
                    imgs.extend(p.glob(f"*{ext.upper()}"))
                return sorted(imgs)
            return [p]
        return [source]

    def __repr__(self) -> str:
        if self.model is not None:
            return (
                f"BILT(variant={self._variant}, "
                f"classes={self.num_classes}, device={self.device})"
            )
        return f"BILT(variant={self._variant}, unloaded, device={self.device})"


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

class Results:
    """
    Holds batch inference results together with optional annotated images.

    Returned by :meth:`BILT.predict` when ``return_images=True``.
    """

    def __init__(
        self,
        detections: List[List[Dict]],
        images: List[Optional[Image.Image]],
        class_names: Optional[List[str]],
    ):
        self.detections = detections
        self.images = images
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, idx: int) -> List[Dict]:
        return self.detections[idx]

    def save(self, save_dir: Union[str, Path] = "runs/detect") -> None:
        """Save annotated images to *save_dir*."""
        from .utils import draw_detections

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, (dets, img) in enumerate(zip(self.detections, self.images)):
            if img is None:
                continue
            annotated = draw_detections(img, dets)
            out = save_dir / f"result_{i}.jpg"
            annotated.save(out)
            logger.info(f"Saved {out}")

    def show(self) -> None:
        """Display annotated images (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            from .utils import draw_detections

            for dets, img in zip(self.detections, self.images):
                if img is None:
                    continue
                annotated = draw_detections(img, dets)
                plt.figure(figsize=(12, 8))
                plt.imshow(annotated)
                plt.axis("off")
                plt.show()
        except ImportError:
            logger.error("matplotlib required for show(). Use save() instead.")
