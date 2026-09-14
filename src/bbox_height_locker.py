"""Module autonome de verrouillage de la hauteur de bounding box (BBox Height Locking).

Filtre géométrique pour empêcher le rétrécissement anormal du bas de la bounding box
(pieds / y_max) lors d'occlusions partielles.
"""

from typing import Union, Tuple
import numpy as np


class BBoxHeightLocker:
    """Verrouille la hauteur stable d'une personne pour préserver l'ancrage des pieds (y_max)."""

    def __init__(self, min_height_ratio: float = 0.70):
        """
        min_height_ratio : Seuil sous lequel un rétrécissement de boîte est considéré
        comme une occlusion (ex: si la hauteur chute sous 70% de la hauteur stable de la personne).
        """
        self.min_height_ratio = float(min_height_ratio)
        self.tracks_profile: dict[int, dict] = {}  # {track_id: {"h_stable": float, "history_h": list[float]}}

    def process_bbox(
        self,
        track_id: int,
        bbox: Union[Tuple[float, float, float, float], np.ndarray, list],
    ) -> Union[Tuple[float, float, float, float], np.ndarray]:
        """Reçoit [x1, y1, x2, y2] et renvoie [x1, y1, x2, y2_corrigé].
        
        Ne modifie rien d'autre : conserve le type d'entrée (tuple ou numpy.ndarray).
        """
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        current_h = y2 - y1

        # 1. Enregistrement initial de la hauteur de la personne
        if track_id not in self.tracks_profile:
            self.tracks_profile[track_id] = {
                "h_stable": current_h,
                "history_h": [current_h],
            }
            if isinstance(bbox, np.ndarray):
                return np.array([x1, y1, x2, y2], dtype=bbox.dtype)
            return (x1, y1, x2, y2)

        profile = self.tracks_profile[track_id]
        h_stable = profile["h_stable"]

        # 2. Si le bas remonte trop vers le haut (occlusion basse)
        if current_h < (h_stable * self.min_height_ratio):
            # Repousse le bas (y2) pour maintenir la vraie hauteur des pieds
            y2_corrected = y1 + h_stable
            print(f"[LOCKER] id={track_id} h_stable={profile['h_stable']:.1f} y2_raw={y2:.1f} y2_corrected={y2_corrected:.1f}")
        else:
            # Comportement normal : Mise à jour progressive de la hauteur de référence
            y2_corrected = y2
            profile["history_h"].append(current_h)
            if len(profile["history_h"]) > 15:
                profile["history_h"].pop(0)
            profile["h_stable"] = sum(profile["history_h"]) / len(profile["history_h"])

        if isinstance(bbox, np.ndarray):
            return np.array([x1, y1, x2, y2_corrected], dtype=bbox.dtype)
        return (x1, y1, x2, y2_corrected)

    def purge_lost_tracks(self, active_ids: set[int], max_age: int = 90) -> None:
        """Nettoie la mémoire des profils pour les pistes absentes depuis plus de max_age frames."""
        for tid in active_ids:
            if tid in self.tracks_profile:
                self.tracks_profile[tid]["age"] = 0
        for tid, prof in list(self.tracks_profile.items()):
            if tid not in active_ids:
                prof["age"] = prof.get("age", 0) + 1
                if prof["age"] > max_age:
                    del self.tracks_profile[tid]
