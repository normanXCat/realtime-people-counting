"""Module autonome de stabilisation cinématique du point d'ancrage.

Ce module implémente AnchorStabilizer, qui stabilise la position basse (y_max)
de la bounding box en utilisant le mouvement de la tête (y_min).
En cas d'effondrement brutal de la boîte (occlusion des jambes par un obstacle),
le déplacement des pieds est recalé sur la vitesse de déplacement de la tête.
"""

from __future__ import annotations


class AnchorStabilizer:
    """Stabilisateur d'ancrage cinématique guidé par la tête."""

    def __init__(self, tolerance_px: float = 20.0):
        self.tolerance_px = tolerance_px
        self.tracks_history: dict[int, dict[str, float]] = {}  # {track_id: {"y_min": float, "y_max": float}}

    def get_stabilized_anchor(
        self, track_id: int, bbox: tuple[float, float, float, float] | list[float]
    ) -> tuple[float, float]:
        """Prend la bbox brute [x1, y1, x2, y2] et retourne le point (x_center, y_max_stabilisé)
        sans altérer la bbox d'origine.
        """
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x_center = (x1 + x2) / 2.0
        y_min_raw, y_max_raw = y1, y2

        if track_id not in self.tracks_history:
            self.tracks_history[track_id] = {
                "y_min": y_min_raw,
                "y_max": y_max_raw,
            }
            return (x_center, y_max_raw)

        prev = self.tracks_history[track_id]
        v_head = y_min_raw - prev["y_min"]
        v_feet = y_max_raw - prev["y_max"]

        # Si l'effondrement de la boîte crée un écart anormal par rapport au mouvement de la tête
        if abs(v_feet - v_head) > self.tolerance_px:
            y_max_stabilized = prev["y_max"] + v_head
        else:
            y_max_stabilized = y_max_raw

        # Mise à jour de l'historique
        prev["y_min"] = y_min_raw
        prev["y_max"] = y_max_stabilized

        return (x_center, y_max_stabilized)

    def purge_lost_tracks(self, active_ids: set[int]) -> None:
        """Nettoie l'historique des pistes inactives pour éviter les fuites mémoire."""
        keys_to_remove = [tid for tid in self.tracks_history if tid not in active_ids]
        for tid in keys_to_remove:
            del self.tracks_history[tid]
