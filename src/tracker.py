"""Comptage robuste par franchissement géométrique d'une ligne virtuelle."""

import math
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple


Point = Tuple[float, float]


def check_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Retourne True si les segments [p1,p2] et [p3,p4] se coupent.

    L'orientation CCW permet de détecter aussi bien un saut complet du
    centroïde par-dessus la ligne qu'une intersection sur une extrémité.
    """
    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    eps = 1e-9
    o1, o2 = orientation(p1, p2, p3), orientation(p1, p2, p4)
    o3, o4 = orientation(p3, p4, p1), orientation(p3, p4, p2)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    return (
        abs(o1) <= eps and on_segment(p1, p3, p2)
        or abs(o2) <= eps and on_segment(p1, p4, p2)
        or abs(o3) <= eps and on_segment(p3, p1, p4)
        or abs(o4) <= eps and on_segment(p3, p2, p4)
    )


def cross_product(a: Point, b: Point) -> float:
    """Produit vectoriel 2D a x b."""
    return a[0] * b[1] - a[1] * b[0]


class LineCrossingTracker:
    """Suit les personnes par point médian bas et détecte le franchissement par produit vectoriel."""

    # Alias de compatibilité pour les intégrations appelant l’ancienne méthode.
    check_intersection = staticmethod(check_intersection)
    _segments_intersect = staticmethod(check_intersection)

    @staticmethod
    def get_bottom_center(bbox: Tuple[float, float, float, float]) -> Point:
        """Point de référence : point médian bas (pieds) (x_center, y_max)."""
        return (float((bbox[0] + bbox[2]) / 2.0), float(bbox[3]))

    def __init__(
        self,
        line_p1: Point = (0.0, 0.6),
        line_p2: Point = (1.0, 0.6),
        max_lost_frames: int = 60,
        history_len: int = 2,
        **_ignored,
    ) -> None:
        self.line_p1 = line_p1
        self.line_p2 = line_p2
        self.max_lost_frames = max_lost_frames
        self.history_len = max(2, history_len)

        # Historique métier : pour chaque display_id, au moins T-1 et T.
        self.track_history: Dict[int, Deque[Point]] = {}
        self.track_lost_counter: Dict[int, int] = {}
        self.counted_in: Set[int] = set()
        self.counted_out: Set[int] = set()
        self.entries = 0
        self.exits = 0
        self._frame_index = 0

    def update(
        self,
        boxes,
        frame_height: int,
        frame_width: Optional[int] = None,
        override_track_ids: Optional[list] = None,
    ) -> Tuple[int, Set[int]]:
        """Met à jour les trajectoires et valide les franchissements vectoriels."""
        self._frame_index += 1
        active_ids: Set[int] = set()
        if frame_width is None:
            frame_width = int(frame_height * 16 / 9)

        line_a = (self.line_p1[0] * frame_width, self.line_p1[1] * frame_height)
        line_b = (self.line_p2[0] * frame_width, self.line_p2[1] * frame_height)
        line_vector = (line_b[0] - line_a[0], line_b[1] - line_a[1])

        if boxes is not None and boxes.id is not None and boxes.xyxy is not None:
            track_ids = override_track_ids if override_track_ids is not None else boxes.id.round().int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            for track_id, bbox in zip(track_ids, xyxy):
                track_id = int(track_id)
                active_ids.add(track_id)
                # Point de référence : point médian bas (les pieds) -> (x_center, y_max)
                bottom_center = (float((bbox[0] + bbox[2]) / 2.0), float(bbox[3]))
                history = self.track_history.setdefault(track_id, deque(maxlen=self.history_len))
                previous = history[-1] if history else None
                history.append(bottom_center)

                if previous is None:
                    continue

                movement = (bottom_center[0] - previous[0], bottom_center[1] - previous[1])
                if abs(movement[0]) + abs(movement[1]) < 1e-6:
                    continue

                # Calcul du produit vectoriel entre AB et AP pour la frame précédente et courante :
                # sign = (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)
                sign_prev = cross_product(line_vector, (previous[0] - line_a[0], previous[1] - line_a[1]))
                sign_curr = cross_product(line_vector, (bottom_center[0] - line_a[0], bottom_center[1] - line_a[1]))

                # Franchissement avec changement de signe du produit vectoriel
                if sign_prev < 0 and sign_curr > 0 and track_id not in self.counted_in:
                    self.counted_in.add(track_id)
                    self.entries += 1
                    print(f"[COMPTAGE] Personne #{track_id} est ENTRÉE (franchissement détecté)")
                elif sign_prev > 0 and sign_curr < 0 and track_id not in self.counted_out:
                    self.counted_out.add(track_id)
                    self.exits += 1
                    print(f"[COMPTAGE] Personne #{track_id} est SORTIE (franchissement détecté)")

        self._cleanup_lost_tracks(active_ids)
        return len(active_ids), active_ids

    def _cleanup_lost_tracks(self, active_ids: Set[int]) -> None:
        """Supprime uniquement les pistes absentes au-delà de max_lost_frames."""
        for track_id in list(self.track_history):
            if track_id in active_ids:
                self.track_lost_counter[track_id] = 0
                continue
            self.track_lost_counter[track_id] = self.track_lost_counter.get(track_id, 0) + 1
            if self.track_lost_counter[track_id] > self.max_lost_frames:
                self.track_history.pop(track_id, None)
                self.track_lost_counter.pop(track_id, None)
                # Les sets restent en mémoire pendant la session pour empêcher
                # un double comptage si un même display_id réapparaît.
