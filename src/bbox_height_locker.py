"""Verrouillage de la hauteur de bounding box (BBox Height Locking).

Filtre géométrique pour empêcher le rétrécissement anormal du bas de la bounding
box (pieds / y_max) lors d'occlusions partielles.

Statut après audit : **conservé en expérimental**, désactivé par défaut
(``stabilization.use_bbox_locker: false``). La logique est inchangée ; seul le
``print()`` non conditionnel a été retiré (spec 5.1) et la purge des profils ne
repose plus sur un compteur d'âge en frames (règle 0.2). Le déclenchement est
tracé par l'appelant sous forme d'événement ``STABILIZATION`` (spec 7).
"""

from typing import Union, Tuple
import numpy as np


class BBoxHeightLocker:
    """Verrouille la hauteur stable d'une personne pour préserver l'ancrage des pieds (y_max)."""

    def __init__(self, min_height_ratio: float = 0.70, history_window: int = 15):
        """
        min_height_ratio : Seuil sous lequel un rétrécissement de boîte est considéré
        comme une occlusion (ex: si la hauteur chute sous 70% de la hauteur stable de la personne).

        history_window : nombre d'hauteurs utilisées pour la hauteur de référence lissée
        (``h_stable``). Le pipeline la fixe sur ``anchor.height_history_window_frames`` :
        c'est la **même** fenêtre que celle de la surveillance de chute, pour ne pas
        entretenir un second jeu de seuils redondant. La hauteur stabilisée exposée par
        :meth:`stable_height` sert aussi de source à l'échelle locale de la zone morte,
        au lieu d'une hauteur brute instantanée.
        """
        self.min_height_ratio = float(min_height_ratio)
        if int(history_window) < 1:
            raise ValueError("history_window doit être >= 1")
        self.history_window = int(history_window)
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
        #    Le déclenchement est tracé par l'appelant (événement STABILIZATION),
        #    pas par une impression console : le mode headless doit rester propre
        #    (spec 5.1).
        if current_h < (h_stable * self.min_height_ratio):
            # Repousse le bas (y2) pour maintenir la vraie hauteur des pieds
            y2_corrected = y1 + h_stable
        else:
            # Comportement normal : Mise à jour progressive de la hauteur de référence
            y2_corrected = y2
            profile["history_h"].append(current_h)
            if len(profile["history_h"]) > self.history_window:
                profile["history_h"].pop(0)
            profile["h_stable"] = sum(profile["history_h"]) / len(profile["history_h"])

        if isinstance(bbox, np.ndarray):
            return np.array([x1, y1, x2, y2_corrected], dtype=bbox.dtype)
        return (x1, y1, x2, y2_corrected)

    def stable_height(self, track_id: int) -> float | None:
        """Hauteur de référence **lissée** d'une piste, ou ``None`` si inconnue.

        C'est la hauteur stabilisée par piste qui doit alimenter toute grandeur
        relative à l'échelle locale (zone morte) : la prendre sur la boîte brute
        instantanée ferait varier cette grandeur au rythme du bruit de détection,
        au lieu de suivre la personne.
        """
        profile = self.tracks_profile.get(track_id)
        if profile is None:
            return None
        height = float(profile["h_stable"])
        return height if height > 0 else None

    def purge_lost_tracks(self, active_ids: set[int]) -> None:
        """Libère les profils des pistes techniques absentes de la frame courante.

        Les clés sont les ``track_id`` techniques, attribués par BoT-SORT et
        propres à la frame : un compteur d'âge en frames serait à la fois un
        doublon de l'état du tracker et une durée métier exprimée en frames
        (règle 0.2). La purge est donc immédiate dès qu'une piste disparaît.
        """
        for track_id in list(self.tracks_profile):
            if track_id not in active_ids:
                del self.tracks_profile[track_id]
