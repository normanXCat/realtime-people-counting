"""Post-Occlusion Recovery — étape 1 (présence visible uniquement).

Une personne comptée, occultée depuis moins de la grâce, peut réapparaître
détectée par YOLO sous ``new_track_thresh`` : BoT-SORT ne lui crée alors pas de
piste. Ce module rattache une telle détection **brute** (prise avant le
tracker, sans seconde inférence) à l'identité occultée, sous conditions
strictes, et la suit ensuite par recouvrement.

Effet strictement limité à la visibilité : l'identité reste ``OCCULTEE`` pour
la FSM (aucune ancre, aucun côté, aucune zone n'est calculé depuis la boîte
récupérée), donc jamais de IN, OUT, NEW ni de variation d'occupation. Le
rattachement aux futures pistes BoT-SORT relève de l'étape 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from config import PostOcclusionRecoveryConfig
from occupancy_types import PersonTrack, TrackState


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


@dataclass
class _Pending:
    confirmations: int
    last_frame: int
    bbox: np.ndarray


class PostOcclusionRecovery:
    """Rattachement de détections brutes à des identités occultées (étape 1)."""

    def __init__(
        self,
        config: PostOcclusionRecoveryConfig,
        describe: Callable[[np.ndarray, Sequence[float]], np.ndarray | None],
        reference_feature: Callable[[int], np.ndarray | None],
        emit: Callable[..., None],
    ) -> None:
        self.config = config
        self._describe = describe
        self._reference = reference_feature
        self._emit = emit
        self._pending: dict[int, _Pending] = {}
        #: Identités récupérées suivies par recouvrement : person_id -> boîte.
        self.active: dict[int, np.ndarray] = {}
        #: (événement, raison) déjà journalisés par identité, pour l'épisode
        #: d'occultation en cours : pas un événement par image, même quand la
        #: raison alterne d'une image à l'autre.
        self._reported: dict[int, set[str]] = {}
        self.recovered_total = 0
        self.descriptors_computed = 0

    # ------------------------------------------------------------------
    def step(
        self,
        tracks: Mapping[int, PersonTrack],
        raw_detections: np.ndarray | None,
        tracked_boxes: Iterable[Sequence[float]],
        seen: set[int],
        frame: np.ndarray | None,
        timestamp_s: float,
        frame_index: int,
        grace_seconds: float,
    ) -> dict[int, tuple[np.ndarray, float]]:
        """Retourne les identités visibles par récupération sur cette frame.

        ``{person_id: (boîte, confiance)}``. N'écrit rien dans les pistes : c'est
        à l'appelant d'en tirer la visibilité, jamais une décision de comptage.
        """
        cfg = self.config
        tracked = [np.asarray(b, dtype=float) for b in tracked_boxes]
        free: list[tuple[np.ndarray, float]] = []
        if raw_detections is not None and len(raw_detections):
            for row in np.asarray(raw_detections, dtype=float):
                box, conf = row[:4], float(row[4])
                if conf < cfg.min_confidence:
                    continue
                if any(box_iou(box, t) >= cfg.max_iou_with_tracked for t in tracked):
                    continue
                free.append((box, conf))

        # 1. Suivi par recouvrement des identités déjà récupérées.
        result: dict[int, tuple[np.ndarray, float]] = {}
        used: set[int] = set()
        for person_id in list(self.active):
            track = tracks.get(person_id)
            if track is None or person_id in seen or track.state is not TrackState.OCCULTEE:
                del self.active[person_id]
                continue
            best, best_iou = None, cfg.follow_min_iou
            for k, (box, _conf) in enumerate(free):
                if k in used:
                    continue
                iou = box_iou(box, self.active[person_id])
                if iou >= best_iou:
                    best, best_iou = k, iou
            if best is None:
                del self.active[person_id]  # la détection brute a cessé
                continue
            used.add(best)
            self.active[person_id] = free[best][0]
            result[person_id] = free[best]

        # 2. Candidats : occultées, comptées, perdues depuis moins de la grâce.
        eligible = [
            t for pid, t in tracks.items()
            if pid not in self.active and pid not in seen
            and t.state is TrackState.OCCULTEE and t.inside_occupancy
            and t.bbox is not None and t.bbox_height > 0
            and (timestamp_s - t.last_seen_s) < grace_seconds
        ]
        eligible_ids = {t.person_id for t in eligible}
        for pid in [p for p in self._pending if p not in eligible_ids]:
            del self._pending[pid]
        for pid in [p for p in self._reported if p not in eligible_ids]:
            del self._reported[pid]  # fin de l'épisode : un futur épisode sera journalisé
        if not eligible:
            return result

        # Similarités, descripteur calculé seulement pour une détection candidate.
        pairs: list[tuple[float, int, int]] = []  # (similarité, index détection, person_id)
        per_detection: dict[int, list[tuple[float, int]]] = {}
        for k, (box, _conf) in enumerate(free):
            if k in used:
                continue
            near = [
                t for t in eligible
                if np.hypot(*(np.subtract(_center(box), _center(t.bbox))))
                <= cfg.radius_height_ratio * t.bbox_height
            ]
            near = [t for t in near if self._reference(t.person_id) is not None]
            if not near or frame is None:
                continue
            feature = self._describe(frame, box)
            self.descriptors_computed += 1
            if feature is None:
                continue
            sims = sorted(
                ((_cosine(feature, self._reference(t.person_id)), t.person_id) for t in near),
                reverse=True,
            )
            per_detection[k] = sims
            for sim, pid in sims:
                pairs.append((sim, k, pid))

        # 3. Décision par détection, puis appariement un pour un (glouton).
        accepted: list[tuple[float, int, int]] = []
        for k, sims in per_detection.items():
            best_sim, best_pid = sims[0]
            second = sims[1][0] if len(sims) > 1 else None
            if best_sim < cfg.similarity_threshold:
                self._report(best_pid, "POST_OCCLUSION_REJECTED", timestamp_s, frame_index,
                             reason="below_similarity_threshold", similarity=best_sim,
                             bbox=free[k][0], confidence=free[k][1])
                continue
            if second is not None and best_sim - second < cfg.min_margin:
                self._report(best_pid, "POST_OCCLUSION_AMBIGUOUS", timestamp_s, frame_index,
                             reason="margin_below_minimum", similarity=best_sim,
                             second_similarity=second, bbox=free[k][0], confidence=free[k][1],
                             candidates=[p for _s, p in sims[:2]])
                continue
            accepted.append((best_sim, k, best_pid))
        taken_det: set[int] = set()
        taken_pid: set[int] = set()
        matched: dict[int, tuple[int, float]] = {}
        for sim, k, pid in sorted(accepted, reverse=True):
            if k in taken_det or pid in taken_pid:
                continue
            # L'identité doit aussi être le meilleur choix de son côté : si une
            # autre détection libre lui ressemble davantage, pas de rattachement.
            rival = max((s for s, kk, p in pairs if p == pid and kk != k), default=None)
            if rival is not None and sim - rival < cfg.min_margin:
                self._report(pid, "POST_OCCLUSION_AMBIGUOUS", timestamp_s, frame_index,
                             reason="competing_detections", similarity=sim,
                             second_similarity=rival, bbox=free[k][0], confidence=free[k][1])
                continue
            taken_det.add(k)
            taken_pid.add(pid)
            matched[pid] = (k, sim)

        # 4. Confirmation sur plusieurs images (images manquées tolérées).
        for pid in list(self._pending):
            if pid not in matched and frame_index - self._pending[pid].last_frame > cfg.max_missed_frames + 1:
                del self._pending[pid]
        for pid, (k, sim) in matched.items():
            box, conf = free[k]
            pending = self._pending.get(pid)
            if pending is None:
                pending = self._pending[pid] = _Pending(0, frame_index, box)
            pending.confirmations += 1
            pending.last_frame = frame_index
            pending.bbox = box
            if pending.confirmations < cfg.confirm_frames:
                continue
            del self._pending[pid]
            track = tracks[pid]
            self.active[pid] = box
            self.recovered_total += 1
            self._reported.pop(pid, None)
            self._emit(
                "POST_OCCLUSION_RECOVERED", timestamp_s, frame_index,
                person_id=int(pid),
                technical_track_id=int(track.technical_track_id),
                similarity=round(float(sim), 4),
                confidence=round(float(conf), 4),
                bbox=[round(float(v), 1) for v in box],
                lost_seconds=round(float(timestamp_s - track.last_seen_s), 3),
                confirmations=int(cfg.confirm_frames),
            )
            result[pid] = (box, conf)
        return result

    def _report(self, person_id: int, event: str, timestamp_s: float, frame_index: int,
                **fields: Any) -> None:
        key = f"{event}:{fields.get('reason')}"
        reported = self._reported.setdefault(person_id, set())
        if key in reported:
            return
        reported.add(key)
        payload = {
            name: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for name, v in fields.items()
        }
        if "bbox" in payload:
            payload["bbox"] = [round(float(v), 1) for v in payload["bbox"]]
        self._emit(event, timestamp_s, frame_index, person_id=int(person_id), **payload)


def _cosine(a: np.ndarray, b: np.ndarray | None) -> float:
    if b is None:
        return 0.0
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0
