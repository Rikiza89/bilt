# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the GNU Affero General Public License v3.0

"""
BILT training engine.

Handles the full training loop:
  - Head warm-up with frozen backbone (epochs 0–4), allowing the detection
    head to stabilise before the backbone starts learning
  - Backbone unfreeze (epoch 5+) — full end-to-end training from scratch
  - AdamW optimiser + cosine LR annealing
  - Gradient clipping
  - Best-checkpoint saving (lowest validation loss)
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .core import DetectionModel, get_lr_scheduler, get_optimizer, get_optimizer_differential
from .dataset import create_dataloader
from .loss import BILTLoss
from .utils import get_logger
from .variants import DEFAULT_VARIANT

logger = get_logger(__name__)


class Trainer:
    """
    Training engine for BILT detectors.

    Parameters
    ----------
    dataset_path  : Path   Root dataset directory (must contain train/ and val/).
    num_classes   : int    Number of object categories.
    class_names   : list   Human-readable names for the categories.
    batch_size    : int    Images per batch (min 2).
    learning_rate : float  Initial AdamW learning rate.
    num_epochs    : int    Total training epochs.
    num_workers   : int    DataLoader worker processes (0 = main process).
    input_size    : int    Image resolution for training; None = variant default.
    device        : torch.device
    variant       : str    BILT model variant (spark / flash / core / pro / max).
    """

    def __init__(
        self,
        dataset_path: Path,
        num_classes: int,
        class_names: list,
        batch_size: int = 4,
        learning_rate: float = 2e-3,
        num_epochs: int = 50,
        num_workers: int = 0,
        input_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        variant: str = DEFAULT_VARIANT,
        # Training loop
        warmup_epochs: int = 3,
        backbone_lr_mult: float = 0.1,
        weight_decay: float = 1e-4,
        cos_lr_min: float = 1e-6,
        grad_clip: float = 5.0,
        # Loss
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_loss_weight: float = 1.0,
        # Augmentation
        augment: bool = True,
        flip_prob: float = 0.5,
        color_jitter: Optional[Tuple[float, float, float, float]] = (0.4, 0.4, 0.4, 0.1),
    ):
        self.dataset_path = Path(dataset_path)
        self.num_classes = num_classes
        self.class_names = class_names
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.variant = variant
        self.warmup_epochs = warmup_epochs
        self.backbone_lr_mult = backbone_lr_mult
        self.weight_decay = weight_decay
        self.cos_lr_min = cos_lr_min
        self.grad_clip = grad_clip

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Trainer using device: {self.device}")

        # Resolve input size from variant default if not provided
        if input_size is None:
            from .variants import get_variant_config
            input_size = get_variant_config(variant)["input_size"]
        self.input_size = input_size

        # ------------------------------------------------------------------ #
        # Data loaders                                                        #
        # ------------------------------------------------------------------ #
        _pin = self.device.type == "cuda"
        logger.info("Building training dataloader …")
        self.train_loader, _ = create_dataloader(
            images_dir=self.dataset_path / "train" / "images",
            labels_dir=self.dataset_path / "train" / "labels",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            input_size=self.input_size,
            training=True,
            pin_memory=_pin,
            augment=augment,
            flip_prob=flip_prob,
            color_jitter=color_jitter,
        )

        logger.info("Building validation dataloader …")
        self.val_loader, _ = create_dataloader(
            images_dir=self.dataset_path / "val" / "images",
            labels_dir=self.dataset_path / "val" / "labels",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            input_size=self.input_size,
            training=False,
            pin_memory=_pin,
        )

        # ------------------------------------------------------------------ #
        # Model                                                               #
        # ------------------------------------------------------------------ #
        logger.info(f"Initialising BILT-{variant} for {num_classes} classes …")
        self.detection_model = DetectionModel(
            variant=variant,
            num_classes=num_classes,
            class_names=class_names,
        )
        self.detection_model.to(self.device)

        # Override the loss criterion with user-supplied hyperparameters
        self.detection_model.model.criterion = BILTLoss(
            num_classes,
            alpha=focal_alpha,
            gamma=focal_gamma,
            box_weight=box_loss_weight,
        )

        # ------------------------------------------------------------------ #
        # Optimiser and LR scheduler                                         #
        # ------------------------------------------------------------------ #
        # If warmup is disabled, use differential LR from the start so the
        # pretrained backbone always trains at a lower rate than the head.
        if warmup_epochs > 0:
            self.optimizer = get_optimizer(
                self.detection_model.model, learning_rate, weight_decay
            )
        else:
            self.optimizer = get_optimizer_differential(
                self.detection_model.model, learning_rate, backbone_lr_mult, weight_decay
            )
        self.scheduler = get_lr_scheduler(self.optimizer, num_epochs, cos_lr_min)

        # Training state
        self.current_epoch = 0
        self.training_losses: list = []
        self.validation_losses: list = []

        logger.info("Trainer ready.")

    # ---------------------------------------------------------------------- #

    def train_one_epoch(self) -> float:
        """Run one forward-backward pass over the training set."""
        self.detection_model.train()
        epoch_loss = 0.0
        num_batches = 0

        _nb = self.device.type == "cuda"
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=_nb)
            targets = [
                {k: v.to(self.device, non_blocking=_nb) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = self.detection_model(images, targets)
            loss = loss_dict["total"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.detection_model.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                cls_l = loss_dict.get("cls", torch.tensor(0.0)).item()
                box_l = loss_dict.get("box", torch.tensor(0.0)).item()
                logger.info(
                    f"Epoch {self.current_epoch + 1}/{self.num_epochs}  "
                    f"batch {batch_idx}/{len(self.train_loader)}  "
                    f"loss={loss.item():.4f}  "
                    f"cls={cls_l:.4f}  box={box_l:.4f}"
                )

        return epoch_loss / max(num_batches, 1)

    # ---------------------------------------------------------------------- #

    def validate(self) -> float:
        """Compute validation loss without updating model parameters."""
        self.detection_model.train()   # loss computation requires train mode
        epoch_loss = 0.0
        num_batches = 0

        _nb = self.device.type == "cuda"
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=_nb)
                targets = [
                    {k: v.to(self.device, non_blocking=_nb) for k, v in t.items()}
                    for t in targets
                ]
                loss_dict = self.detection_model(images, targets)
                epoch_loss += loss_dict["total"].item()
                num_batches += 1

        return epoch_loss / max(num_batches, 1)

    # ---------------------------------------------------------------------- #

    def train(
        self,
        save_path: Path,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Parameters
        ----------
        save_path : Path  Where to write the best checkpoint.
        callback  : optional callable invoked at the end of each epoch with
                    a metrics dict.

        Returns
        -------
        dict with final training metrics.
        """
        logger.info(
            f"Training BILT-{self.variant}  "
            f"epochs={self.num_epochs}  "
            f"input={self.input_size}px  "
            f"device={self.device}"
        )
        t0 = time.time()
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Backbone warmup: freeze backbone for the first warmup_epochs so
            # the randomly-initialised head can stabilise before touching
            # pretrained features.  warmup_epochs=0 skips this entirely.
            if epoch == 0 and self.warmup_epochs > 0:
                logger.info(
                    f"Warming up: backbone frozen for first {self.warmup_epochs} epochs."
                )
                self.detection_model.model.backbone.freeze()
            if self.warmup_epochs > 0 and epoch == self.warmup_epochs:
                logger.info(
                    "Unfreezing backbone with differential LR "
                    f"(backbone={self.learning_rate * self.backbone_lr_mult:.1e}, "
                    f"head={self.learning_rate:.1e})."
                )
                self.detection_model.model.backbone.unfreeze()
                self.optimizer = get_optimizer_differential(
                    self.detection_model.model,
                    self.learning_rate,
                    self.backbone_lr_mult,
                    self.weight_decay,
                )
                # Reinitialise the scheduler to track the new optimiser for the
                # remaining epochs.
                remaining = self.num_epochs - epoch
                self.scheduler = get_lr_scheduler(self.optimizer, remaining, self.cos_lr_min)

            train_loss = self.train_one_epoch()
            self.training_losses.append(train_loss)

            val_loss = self.validate()
            self.validation_losses.append(val_loss)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={current_lr:.2e}"
            )

            if callback:
                callback({
                    "epoch":        epoch + 1,
                    "total_epochs": self.num_epochs,
                    "train_loss":   train_loss,
                    "val_loss":     val_loss,
                    "lr":           current_lr,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                class_id_mapping = {
                    "class_id_to_idx": getattr(
                        self.train_loader.dataset, "class_id_to_idx", None
                    ),
                    "idx_to_class_id": getattr(
                        self.train_loader.dataset, "idx_to_class_id", None
                    ),
                }
                self.detection_model.save(
                    save_path, self.class_names, class_id_mapping
                )
                logger.info(f"  ↳ New best model saved (val_loss={val_loss:.4f})")

        elapsed = time.time() - t0
        logger.info(f"Training finished in {elapsed:.1f}s")

        return {
            "variant":           self.variant,
            "num_epochs":        self.num_epochs,
            "final_train_loss":  self.training_losses[-1],
            "final_val_loss":    self.validation_losses[-1],
            "best_val_loss":     best_val_loss,
            "training_time":     elapsed,
            "model_path":        str(save_path),
        }
