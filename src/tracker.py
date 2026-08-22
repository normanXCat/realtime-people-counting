"""
Module de gestion du suivi d'objets et détection de franchissement de ligne virtuelle arbitraire (2 points).

Ce module définit la classe `LineCrossingTracker` qui exploite les identifiants
de tracking fournis par YOLO/ByteTrack pour suivre les trajectoires des objets
et détecter si et quand ils traversent une frontière virtuelle orientée (définie par 2 points).

Version robuste : intègre une bande d'hystérésis (zone morte perpendiculaire à la ligne),
un lissage de la distance signée et un cooldown temporel par ID afin d'éliminer
les faux comptages dus au bruit de détection et aux oscillations d'une personne immobile près de la ligne.
"""

import math
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ultralytics.engine.results import Boxes


class LineCrossingTracker:
    """
    Gère le suivi des identifiants (tracking) et détecte le franchissement
    d'une ligne virtuelle arbitraire à 2 points pour compter les entrées et les sorties.

    La détection de franchissement repose sur trois garde-fous complémentaires :
        1. Bande d'hystérésis : une zone morte perpendiculaire à la ligne évite
           qu'un léger jitter de la bbox ne bascule l'état.
        2. Lissage de trajectoire : l'état est calculé sur la moyenne des
           dernières distances signées à la ligne.
        3. Cooldown par ID : un délai minimal entre deux comptages consécutifs
           pour un même ID empêche les allers-retours rapprochés d'être comptés plusieurs fois.

    Attributes:
        line_p1 (Tuple[float, float]): Coordonnées normalisées (x1, y1) du premier point.
        line_p2 (Tuple[float, float]): Coordonnées normalisées (x2, y2) du second point.
        max_lost_frames (int): Nombre maximal de frames d'inactivité avant d'oublier un ID.
        history_len (int): Taille maximale de l'historique des positions pour les trails.
        band_margin_ratio (float): Demi-largeur de la bande morte, en fraction de la dimension minimale d'image.
        smoothing_window (int): Nombre de positions récentes utilisées pour lisser la distance.
        min_cooldown_frames (int): Nombre minimal de frames entre deux comptages pour un même ID.
        track_history (Dict[int, Deque[Tuple[int, int]]]): Historique des centroids.
        track_states (Dict[int, str]): Dernier état confirmé par rapport à la ligne ('side_a' ou 'side_b').
        track_lost_counter (Dict[int, int]): Nombre de frames d'inactivité cumulées par ID.
        track_last_count_frame (Dict[int, int]): Numéro de frame du dernier comptage validé par ID.
        entries (int): Nombre total d'entrées détectées (passage de side_a vers side_b).
        exits (int): Nombre total de sorties détectées (passage de side_b vers side_a).
    """

    def __init__(
        self,
        line_p1: Tuple[float, float] = (0.0, 0.6),
        line_p2: Tuple[float, float] = (1.0, 0.6),
        max_lost_frames: int = 90,
        history_len: int = 20,
        band_margin_ratio: float = 0.03,
        smoothing_window: int = 5,
        min_cooldown_frames: int = 15,
    ):
        """
        Initialise le tracker de franchissement de ligne arbitraire.

        Args:
            line_p1 (Tuple[float, float]): Coordonnées normalisées (x, y) dans [0.0, 1.0] du point 1.
            line_p2 (Tuple[float, float]): Coordonnées normalisées (x, y) dans [0.0, 1.0] du point 2.
            max_lost_frames (int): Seuil de frames perdues avant oubli. Défaut à 90.
            history_len (int): Taille max de la queue pour la trajectoire. Défaut à 20.
            band_margin_ratio (float): Demi-largeur de la bande morte perpendiculaire à la ligne.
            smoothing_window (int): Nombre de points récents pour la moyenne glissante de distance.
            min_cooldown_frames (int): Delai minimal entre deux comptages pour un même ID.
        """
        self.line_p1: Tuple[float, float] = line_p1
        self.line_p2: Tuple[float, float] = line_p2
        self.max_lost_frames: int = max_lost_frames
        self.history_len: int = history_len
        self.band_margin_ratio: float = band_margin_ratio
        self.smoothing_window: int = smoothing_window
        self.min_cooldown_frames: int = min_cooldown_frames

        # États internes de tracking
        self.track_history: Dict[int, Deque[Tuple[int, int]]] = {}
        self.track_states: Dict[int, str] = {}
        self.track_lost_counter: Dict[int, int] = {}
        self.track_last_count_frame: Dict[int, int] = {}

        # Compteurs globaux de franchissement
        self.entries: int = 0
        self.exits: int = 0

        # Compteur de frames global
        self._frame_index: int = 0

    def update(
        self,
        boxes,
        frame_height: int,
        frame_width: Optional[int] = None,
        override_track_ids: Optional[list] = None,
    ) -> Tuple[int, Set[int]]:
        """
        Met à jour l'état du tracker avec les boîtes de détection de la frame courante.

        Calcule les centroids des boîtes, met à jour l'historique des positions,
        lisse la distance signée à la ligne, compare le côté actuel ('side_a'/'side_b'/bande morte)
        à l'état précédemment confirmé, et met à jour les compteurs entrées/sorties.

        Args:
            boxes (ultralytics.engine.results.Boxes): Boîtes renvoyées par YOLO avec les IDs.
            frame_height (int): La hauteur de la frame actuelle.
            frame_width (int, optional): La largeur de la frame.
            override_track_ids (list, optional): Identifiants remappés par ReID à utiliser à la place de boxes.id.

        Returns:
            Tuple[int, Set[int]]:
                - current_count (int) : Nombre d'objets actuellement visibles.
                - active_ids_in_frame (Set[int]) : Ensemble des identifiants actifs de la frame.
        """
        self._frame_index += 1
        active_ids_in_frame: Set[int] = set()
        current_count: int = 0

        if frame_width is None:
            frame_width = int(frame_height * (16.0 / 9.0))

        # Conversion des points normalisés de la ligne en pixels absolus
        p1_px = (int(self.line_p1[0] * frame_width), int(self.line_p1[1] * frame_height))
        p2_px = (int(self.line_p2[0] * frame_width), int(self.line_p2[1] * frame_height))

        # Longueur du segment de ligne
        line_len = math.hypot(p2_px[0] - p1_px[0], p2_px[1] - p1_px[1])
        if line_len < 1e-5:
            line_len = 1.0

        # Demi-largeur de la bande morte en pixels
        margin_px = max(2.0, self.band_margin_ratio * min(frame_width, frame_height))

        # Traitement des détections
        if boxes is not None and boxes.id is not None and boxes.xyxy is not None:
            if override_track_ids is not None:
                track_ids = override_track_ids
            else:
                track_ids = boxes.id.round().int().cpu().tolist()
            xyxy_list = boxes.xyxy.cpu().numpy()
            current_count = len(set(track_ids))

            for track_id, bbox in zip(track_ids, xyxy_list):
                active_ids_in_frame.add(track_id)
                x1, y1, x2, y2 = bbox

                # Centroid
                cx: int = round((x1 + x2) / 2)
                cy: int = round((y1 + y2) / 2)

                # Mise à jour historique. Le dernier segment est conservé pour
                # vérifier un franchissement géométrique réel, pas seulement un
                # changement de signe dû au jitter de la boîte.
                previous_point = self.track_history[track_id][-1] if track_id in self.track_history and self.track_history[track_id] else None
                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=self.history_len)
                self.track_history[track_id].append((cx, cy))

                # Lissage de la distance signée à la ligne
                smoothed_dist: float = self._smoothed_signed_distance(track_id, p1_px, p2_px, line_len)

                # Zone nette ('side_a', 'side_b', ou None si dans la bande morte)
                zone: Optional[str] = self._zone_from_distance(smoothed_dist, margin_px)

                if track_id not in self.track_states:
                    if zone is not None:
                        self.track_states[track_id] = zone
                    continue

                prev_state: str = self.track_states[track_id]

                if zone is None or zone == prev_state:
                    continue

                # Validation avec cooldown
                last_count_frame = self.track_last_count_frame.get(
                    track_id, -self.min_cooldown_frames
                )
                if (self._frame_index - last_count_frame) < self.min_cooldown_frames:
                    self.track_states[track_id] = zone
                    continue

                segment_crosses = previous_point is not None and self._segments_intersect(
                    previous_point, (cx, cy), p1_px, p2_px
                )
                if not segment_crosses:
                    continue

                if prev_state == "side_a" and zone == "side_b":
                    self.entries += 1
                    self.track_last_count_frame[track_id] = self._frame_index
                    print(f"[COMPTAGE] Personne #{track_id} est ENTRÉE (Côté A -> Côté B)")
                elif prev_state == "side_b" and zone == "side_a":
                    self.exits += 1
                    self.track_last_count_frame[track_id] = self._frame_index
                    print(f"[COMPTAGE] Personne #{track_id} est SORTIE (Côté B -> Côté A)")

                self.track_states[track_id] = zone

        # Nettoyage des pistes obsolètes
        self._cleanup_lost_tracks(active_ids_in_frame)

        return current_count, active_ids_in_frame

    def _smoothed_signed_distance(
        self,
        track_id: int,
        p1_px: Tuple[int, int],
        p2_px: Tuple[int, int],
        line_len: float,
    ) -> float:
        """
        Calcule la moyenne de la distance perpendiculaire signée récente d'un ID par rapport à la ligne.

        Distance signée = ((x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)) / line_len.
        """
        path = self.track_history[track_id]
        window = list(path)[-self.smoothing_window:]
        x1, y1 = p1_px
        x2, y2 = p2_px

        distances = [
            ((x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)) / line_len
            for pt in window
        ]
        return sum(distances) / len(distances)

    @staticmethod
    def _zone_from_distance(dist: float, margin_px: float) -> Optional[str]:
        """
        Détermine le côté de la ligne d'après la distance perpendiculaire signée lissée.

        - dist < -margin_px  -> 'side_a' (au-dessus ou à gauche de la ligne orientée P1->P2)
        - dist > margin_px   -> 'side_b' (en-dessous ou à droite de la ligne orientée P1->P2)
        - sinon              -> None (bande morte / hystérésis)
        """
        if dist < -margin_px:
            return "side_a"
        if dist > margin_px:
            return "side_b"
        return None

    @staticmethod
    def _segments_intersect(a, b, c, d) -> bool:
        """Retourne vrai si les segments AB et CD se coupent."""
        def orient(p, q, r):
            value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            return 0 if abs(value) < 1e-6 else (1 if value > 0 else -1)

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        if o1 != o2 and o3 != o4:
            return True
        return (o1 == 0 and on_segment(a, c, b)) or (o2 == 0 and on_segment(a, d, b)) or (o3 == 0 and on_segment(c, a, d)) or (o4 == 0 and on_segment(c, b, d))

    def _cleanup_lost_tracks(self, active_ids_in_frame: Set[int]) -> None:
        """Purge les danciennes pistes qui ont dépassé `max_lost_frames`."""
        lost_ids = list(self.track_history.keys() - active_ids_in_frame)
        for lid in lost_ids:
            self.track_lost_counter[lid] = self.track_lost_counter.get(lid, 0) + 1
            if self.track_lost_counter[lid] > self.max_lost_frames:
                self.track_history.pop(lid, None)
                self.track_states.pop(lid, None)
                self.track_lost_counter.pop(lid, None)
                self.track_last_count_frame.pop(lid, None)

        for aid in active_ids_in_frame:
            self.track_lost_counter[aid] = 0
