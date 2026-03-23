# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the GNU Affero General Public License v3.0

"""
BILT dataset loader.

Expects the standard label format used by most annotation tools:

    <class_id>  <x_center>  <y_center>  <width>  <height>

All five values are normalised to [0, 1] relative to the image dimensions.
One label file per image, with the same stem and a ``.txt`` extension,
located in the ``labels/`` directory that mirrors the ``images/`` directory.

Dataset directory layout
------------------------
    <root>/
        train/
            images/   *.jpg / *.png / …
            labels/   *.txt
        val/
            images/
            labels/
        data.yaml     (optional – used for human-readable class names)

data.yaml format
----------------
    names: [cat, dog, person]
    nc: 3
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .backbone import NORM_MEAN, NORM_STD
from .utils import get_logger, load_yaml_classes, parse_bilt_label

logger = get_logger(__name__)


class ObjectDetectionDataset(Dataset):
    """
    PyTorch Dataset for object detection.

    Loads images and their corresponding annotation files, remaps class IDs
    to a consecutive zero-indexed range, resizes images to *input_size* and
    applies standard normalisation.

    Parameters
    ----------
    images_dir : Path   Directory containing image files.
    labels_dir : Path   Directory containing label ``.txt`` files.
    transforms : callable, optional  Applied to each PIL image.
    input_size : int    Target square resolution (only used if no transforms).
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transforms=None,
        input_size: int = 512,
        training: bool = False,
        augment: Optional[bool] = None,
        flip_prob: float = 0.5,
        color_jitter: Optional[Tuple[float, float, float, float]] = (0.4, 0.4, 0.4, 0.1),
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.input_size = input_size
        self.training = training
        # augment defaults to True for training split, False for val
        self.augment = training if augment is None else augment
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter

        # Discover image files
        self.image_files: List[Path] = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.image_files.extend(self.images_dir.glob(f"*{ext}"))
            self.image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        self.image_files = sorted(self.image_files)

        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        logger.info(f"Dataset: {len(self.image_files)} images in {self.images_dir}")

        # Collect unique class IDs present in the label files
        self.class_ids: List[int] = []
        _ids: set = set()
        for img_path in self.image_files:
            lbl = self.labels_dir / f"{img_path.stem}.txt"
            if lbl.exists():
                try:
                    with open(lbl) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                _ids.add(int(parts[0]))
                except Exception as exc:
                    logger.warning(f"Could not read {lbl}: {exc}")

        self.class_ids = sorted(_ids)
        self.num_classes = len(self.class_ids)

        # Map original class IDs to consecutive indices (0-based internally,
        # but the model receives 1-based labels to distinguish from background)
        self.class_id_to_idx: Dict[int, int] = {
            cid: idx for idx, cid in enumerate(self.class_ids)
        }
        self.idx_to_class_id: Dict[int, int] = {
            idx: cid for cid, idx in self.class_id_to_idx.items()
        }

        logger.info(
            f"Found {self.num_classes} classes with IDs: {self.class_ids}"
        )

    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_files[idx]
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
        except Exception as exc:
            logger.error(f"Cannot load {img_path}: {exc}")
            img = Image.new("RGB", (self.input_size, self.input_size))
            orig_w, orig_h = self.input_size, self.input_size

        # Parse annotations
        anns = parse_bilt_label(lbl_path, orig_w, orig_h)

        if anns:
            boxes = torch.tensor(
                [a["bbox"] for a in anns], dtype=torch.float32
            )
            # Labels are 1-indexed (0 = background in the anchor matcher)
            labels = torch.tensor(
                [self.class_id_to_idx[a["class_id"]] + 1 for a in anns],
                dtype=torch.int64,
            )
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        # Bbox-safe augmentation (controlled by self.augment)
        if self.augment:
            # Random horizontal flip — mirror boxes along the x-axis
            if random.random() < self.flip_prob:
                img = TF.hflip(img)
                if boxes.numel() > 0:
                    flipped = boxes.clone()
                    flipped[:, 0] = orig_w - boxes[:, 2]
                    flipped[:, 2] = orig_w - boxes[:, 0]
                    boxes = flipped
            # Color jitter — brightness, contrast, saturation, hue variation
            if self.color_jitter is not None:
                b, c, s, h = self.color_jitter
                img = T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)(img)

        # Apply transforms (resize + to-tensor + normalise)
        if self.transforms:
            img = self.transforms(img)

        # Scale bounding boxes to match the transformed image size
        if isinstance(img, torch.Tensor):
            _, th, tw = img.shape
        else:
            tw, th = img.size

        if boxes.numel() > 0:
            sx = tw / orig_w
            sy = th / orig_h
            boxes[:, 0] *= sx
            boxes[:, 1] *= sy
            boxes[:, 2] *= sx
            boxes[:, 3] *= sy
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, tw)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, th)

            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes  = boxes[valid]
            labels = labels[valid]

        return img, {"boxes": boxes, "labels": labels}

    # ---------------------------------------------------------------------- #

    def get_class_names(
        self, yaml_path: Optional[Path] = None
    ) -> List[str]:
        """
        Return a list of human-readable class names.

        Loads from *yaml_path* if available; otherwise generates
        ``class_<id>`` placeholders.
        """
        if yaml_path and yaml_path.exists():
            names = load_yaml_classes(yaml_path)
            if names:
                logger.info(f"Class names from YAML: {names}")
                return [
                    names[cid] if cid < len(names) else f"class_{cid}"
                    for cid in self.class_ids
                ]
        logger.warning("No YAML found – using auto-generated class names.")
        return [f"class_{cid}" for cid in self.class_ids]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(input_size: int = 512, training: bool = True) -> T.Compose:
    """
    Build the preprocessing pipeline for the given split.

    Both training and validation images are resized and normalised with
    standard statistics before being fed into the model.

    Parameters
    ----------
    input_size : int   Target square resolution.
    training   : bool  (reserved for future data augmentation).
    """
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Lightweight class-info reader (no image loading)
# ---------------------------------------------------------------------------

def read_dataset_info(
    labels_dir: Path,
    yaml_path: Optional[Path] = None,
) -> Tuple[int, List[str]]:
    """
    Return ``(num_classes, class_names)`` by scanning label files only —
    no images are loaded.  Used by ``BILT.train()`` to avoid creating a
    full dataset object just to discover class metadata.

    Parameters
    ----------
    labels_dir : Path  Directory containing ``.txt`` label files.
    yaml_path  : Path  Optional ``data.yaml`` for human-readable names.

    Returns
    -------
    (num_classes, class_names)
    """
    labels_dir = Path(labels_dir)
    class_ids: set = set()
    for lbl in labels_dir.glob("*.txt"):
        try:
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_ids.add(int(parts[0]))
        except Exception:
            pass

    sorted_ids = sorted(class_ids)
    num_classes = len(sorted_ids)

    if yaml_path and yaml_path.exists():
        names = load_yaml_classes(yaml_path)
        if names:
            class_names = [
                names[cid] if cid < len(names) else f"class_{cid}"
                for cid in sorted_ids
            ]
            return num_classes, class_names

    return num_classes, [f"class_{cid}" for cid in sorted_ids]


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Stack images into a batch tensor; keep targets as a list of dicts."""
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    images_dir: Path,
    labels_dir: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    input_size: int = 512,
    training: bool = True,
    pin_memory: bool = False,
    augment: Optional[bool] = None,
    flip_prob: float = 0.5,
    color_jitter: Optional[Tuple[float, float, float, float]] = (0.4, 0.4, 0.4, 0.1),
) -> Tuple[DataLoader, int]:
    """
    Create a DataLoader for the given images and labels directories.

    Returns
    -------
    (dataloader, num_classes)
    """
    dataset = ObjectDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transforms=get_transforms(input_size, training),
        input_size=input_size,
        training=training,
        augment=augment,
        flip_prob=flip_prob,
        color_jitter=color_jitter,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return loader, dataset.num_classes
