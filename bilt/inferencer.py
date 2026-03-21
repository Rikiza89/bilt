# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the GNU Affero General Public License v3.0

"""
BILT inference engine.

Handles image preprocessing (resize + normalisation), model invocation,
postprocessing (confidence filter + per-class NMS) and coordinate scaling
back to the original image space.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image

from .backbone import NORM_MEAN, NORM_STD
from .utils import get_logger

logger = get_logger(__name__)


class Inferencer:
    """
    Image-to-detections inference wrapper.

    Parameters
    ----------
    model                : BILTDetector  (in eval mode, on *device*)
    class_names          : list of str
    confidence_threshold : float  minimum score to keep a detection
    nms_threshold        : float  IoU threshold for NMS
    input_size           : int    resize images to (input_size, input_size)
    device               : torch.device
    """

    def __init__(
        self,
        model,
        class_names: List[str],
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: int = 512,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        self._build_transforms()

        logger.info(
            f"Inferencer: {len(class_names)} classes | "
            f"conf={confidence_threshold} | nms_iou={nms_threshold} | "
            f"size={input_size}"
        )

    # ---------------------------------------------------------------------- #

    def _build_transforms(self) -> None:
        """Rebuild the preprocessing pipeline for the current input_size."""
        self._transforms = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

    # ---------------------------------------------------------------------- #

    def preprocess_image(self, image: Image.Image):
        """
        Convert a PIL image to a normalised tensor and record the original size.

        Returns
        -------
        tensor       : (1, 3, input_size, input_size)
        original_size: (width, height)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = image.size  # (W, H)
        tensor = self._transforms(image).unsqueeze(0).to(self.device)
        return tensor, original_size

    # ---------------------------------------------------------------------- #

    def postprocess_predictions(
        self,
        raw: Dict[str, torch.Tensor],
        original_size: tuple,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw model output to a list of detection dicts.

        Parameters
        ----------
        raw           : dict with 'boxes', 'scores', 'labels' in model space
        original_size : (W, H) of the original (pre-resize) image

        Returns
        -------
        list of dicts:
            bbox       : [x1, y1, x2, y2]  absolute pixel coords in original
            score      : float
            class_id   : int  (1-indexed)
            class_name : str
        """
        boxes  = raw["boxes"].cpu()
        scores = raw["scores"].cpu()
        labels = raw["labels"].cpu()

        # Confidence filter
        keep = scores > self.confidence_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if boxes.numel() == 0:
            return []

        # NMS (already applied inside the model, but we apply again with the
        # user-supplied threshold so the user can tune it at inference time)
        from .utils import apply_nms
        nms_keep = apply_nms(boxes, scores, self.nms_threshold)
        boxes  = boxes[nms_keep]
        scores = scores[nms_keep]
        labels = labels[nms_keep]

        # Scale boxes from model-input space → original image space
        orig_w, orig_h = original_size
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        detections: List[Dict[str, Any]] = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()

            x1 = max(0, min(int(x1 * scale_x), orig_w))
            y1 = max(0, min(int(y1 * scale_y), orig_h))
            x2 = max(0, min(int(x2 * scale_x), orig_w))
            y2 = max(0, min(int(y2 * scale_y), orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(label.item())
            cls_name = (
                self.class_names[cls_id - 1]
                if 0 < cls_id <= len(self.class_names)
                else f"class_{cls_id}"
            )

            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "score":      float(score.item()),
                "class_id":   cls_id,
                "class_name": cls_name,
            })

        return detections

    # ---------------------------------------------------------------------- #

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run detection on a single PIL image.

        Returns
        -------
        list of detection dicts.
        """
        if self.input_size != self._transforms.transforms[0].size[0]:
            self._build_transforms()

        tensor, original_size = self.preprocess_image(image)

        with torch.no_grad():
            outputs = self.model(tensor)

        raw = outputs[0]
        detections = self.postprocess_predictions(raw, original_size)

        logger.debug(
            f"Detected {len(detections)} objects in image {original_size}"
        )
        return detections

    # ---------------------------------------------------------------------- #

    def detect_batch(
        self, images: List[Image.Image]
    ) -> List[List[Dict[str, Any]]]:
        """Run detection on a list of PIL images."""
        return [self.detect(img) for img in images]

    def detect_from_path(self, image_path: Path) -> List[Dict[str, Any]]:
        """Run detection on an image file."""
        try:
            return self.detect(Image.open(image_path))
        except Exception as exc:
            logger.error(f"Failed to process {image_path}: {exc}")
            return []
