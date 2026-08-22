"""Association d'identités hybride ByteTrack / DeepSORT.

ByteTrack reste le tracker principal et fournit les IDs cinématiques à chaque
frame. Ce module ne remplace donc pas le tracker Ultralytics : il stabilise ses
IDs. En situation ambiguë (IoU élevé entre personnes, ID dupliqué ou nouvelle
piste après une absence), il active une association de type DeepSORT :
embeddings d'apparence + proximité de prédiction/mouvement, résolue par
l'algorithme hongrois.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from torchvision import models, transforms


@dataclass
class IdentityState:
    embedding: np.ndarray
    bbox: np.ndarray
    last_seen: int
    velocity: np.ndarray


class HybridAssociation:
    """Stabilise les IDs ByteTrack avec une association Re-ID ponctuelle."""

    def __init__(
        self,
        occlusion_iou: float = 0.35,
        max_age: int = 150,
        appearance_threshold: float = 0.52,
        motion_weight: float = 0.35,
        appearance_weight: float = 0.65,
    ) -> None:
        self.occlusion_iou = occlusion_iou
        self.max_age = max_age
        self.appearance_threshold = appearance_threshold
        self.motion_weight = motion_weight
        self.appearance_weight = appearance_weight
        self.states: Dict[int, IdentityState] = {}
        self.raw_to_stable: Dict[int, int] = {}
        self.next_id = 1
        self._model: Optional[nn.Module] = None
        self._transform = None
        self._device: Optional[torch.device] = None

    def resolve(
        self,
        raw_ids: Sequence[int],
        boxes: np.ndarray,
        frame: np.ndarray,
        frame_index: int,
    ) -> Tuple[List[int], bool]:
        """Retourne les IDs stables et indique si le mode DeepSORT a été activé."""
        raw_ids = [int(x) for x in raw_ids]
        boxes = np.asarray(boxes, dtype=np.float32)
        if len(raw_ids) != len(boxes):
            raise ValueError("raw_ids et boxes doivent avoir la même longueur")
        if not raw_ids:
            self._purge(frame_index)
            return [], False

        has_reappearance = any(raw_id not in self.raw_to_stable for raw_id in raw_ids) and any(frame_index - state.last_seen <= self.max_age for state in self.states.values())
        imminent = self._has_imminent_occlusion(boxes) or len(set(raw_ids)) < len(raw_ids) or has_reappearance
        embeddings = [self._embedding(frame, box) for box in boxes] if imminent else [None] * len(boxes)
        output: List[Optional[int]] = [None] * len(raw_ids)
        used: set[int] = set()

        # Régime normal : ByteTrack garde la main, sans coût Re-ID par frame.
        if not imminent:
            for i, raw_id in enumerate(raw_ids):
                stable_id = self.raw_to_stable.get(raw_id, raw_id)
                if stable_id in used:
                    imminent = True
                    break
                output[i] = stable_id
                used.add(stable_id)

        # Mode de secours DeepSORT : association apparence + mouvement.
        if imminent:
            output = self._associate(boxes, embeddings, frame_index, used)
            for raw_id, stable_id in zip(raw_ids, output):
                if stable_id is not None:
                    self.raw_to_stable[raw_id] = stable_id

        for i, stable_id in enumerate(output):
            if stable_id is None:
                stable_id = self._new_id(used)
                output[i] = stable_id
            state = self.states.get(stable_id)
            should_refresh = imminent or state is None or (frame_index - state.last_seen >= 15)
            emb = embeddings[i] if embeddings[i] is not None else (self._embedding(frame, boxes[i]) if should_refresh else state.embedding)
            if emb is not None:
                previous = state
                velocity = boxes[i][:2] - previous.bbox[:2] if previous else np.zeros(2, dtype=np.float32)
                self.states[stable_id] = IdentityState(emb, boxes[i].copy(), frame_index, velocity)
            elif state is not None:
                # En régime ByteTrack, actualiser la prédiction sans lancer Re-ID.
                state.velocity = boxes[i][:2] - state.bbox[:2]
                state.bbox = boxes[i].copy()
                state.last_seen = frame_index
            self.raw_to_stable[raw_ids[i]] = stable_id
            used.add(stable_id)

        self._purge(frame_index)
        return [int(x) for x in output], imminent

    def _associate(self, boxes, embeddings, frame_index: int, used: set[int]) -> List[Optional[int]]:
        candidates = [sid for sid, state in self.states.items() if frame_index - state.last_seen <= self.max_age and sid not in used]
        if not candidates:
            return [None] * len(boxes)
        cost = np.full((len(boxes), len(candidates)), 1e3, dtype=np.float32)
        for i, (box, emb) in enumerate(zip(boxes, embeddings)):
            if emb is None:
                continue
            for j, sid in enumerate(candidates):
                state = self.states[sid]
                similarity = float(np.dot(emb, state.embedding))
                predicted = state.bbox.copy()
                predicted[:2] += state.velocity
                predicted[2:] += state.velocity
                iou = self._iou(box, predicted)
                if similarity >= self.appearance_threshold:
                    cost[i, j] = self.appearance_weight * (1.0 - similarity) + self.motion_weight * (1.0 - iou)
        rows, cols = linear_sum_assignment(cost)
        result: List[Optional[int]] = [None] * len(boxes)
        for row, col in zip(rows, cols):
            if cost[row, col] < 1e3:
                result[row] = candidates[col]
        return result

    def _has_imminent_occlusion(self, boxes: np.ndarray) -> bool:
        return any(self._iou(a, b) >= self.occlusion_iou for i, a in enumerate(boxes) for b in boxes[i + 1 :])

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    def _new_id(self, used: set[int]) -> int:
        while self.next_id in used or self.next_id in self.states:
            self.next_id += 1
        value = self.next_id; self.next_id += 1
        return value

    def _purge(self, frame_index: int) -> None:
        expired = [sid for sid, state in self.states.items() if frame_index - state.last_seen > self.max_age]
        for sid in expired:
            self.states.pop(sid, None)
        self.raw_to_stable = {raw: sid for raw, sid in self.raw_to_stable.items() if sid in self.states}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.classifier = nn.Identity()
        self._model = backbone.eval().to(self._device)
        self._transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((128, 64)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _embedding(self, frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        self._ensure_model()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        vector = self._model(self._transform(crop).unsqueeze(0).to(self._device)).squeeze().cpu().numpy()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-6 else None
