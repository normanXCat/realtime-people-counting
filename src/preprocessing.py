"""Prétraitement de la frame avant détection et ré-identification (lot 7).

Le CLAHE est appliqué **en place** sur la frame que lit Ultralytics, avant son
propre prétraitement (letterbox) : ``result.orig_img`` est alors la frame
normalisée, et l'extracteur d'apparence (OSNet) découpe exactement l'image
qu'a vue YOLO. Dimensions et origine des coordonnées sont inchangées.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from config import ClaheConfig


class ClaheNormalizer:
    """CLAHE sur le canal L de l'espace LAB ; teinte (a, b) inchangée."""

    def __init__(self, config: ClaheConfig) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=float(config.clip_limit),
            tileGridSize=tuple(int(v) for v in config.tile_grid_size),
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Retourne une nouvelle frame BGR normalisée sur la luminance seule."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_inplace(self, frame: np.ndarray) -> None:
        """Remplace le contenu de ``frame`` sans changer l'objet ni sa forme."""
        frame[...] = self.apply(frame)


def make_predict_callback(
    normalizer: ClaheNormalizer, record: Callable[[float], None] | None = None
) -> Callable[[object], None]:
    """Rappel Ultralytics ``on_predict_batch_start`` : normalise le lot en place.

    ``predictor.batch`` vaut ``(chemins, images, infos)`` ; les images sont
    celles que le prédicteur renverra dans ``result.orig_img``.
    """
    import time

    def _callback(predictor: object) -> None:
        batch = getattr(predictor, "batch", None)
        if not batch or len(batch) < 2:
            return
        start = time.perf_counter()
        for image in batch[1]:
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
                normalizer.apply_inplace(image)
        if record is not None:
            record((time.perf_counter() - start) * 1000.0)

    return _callback
