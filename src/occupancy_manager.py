"""Gestionnaire d'occupation robuste avec machine à états, hystérésis et warm-up.

Ce module implémente OccupancyManager, le cœur du système de comptage :
- Intersection vectorielle CCW pour détecter les franchissements
- Machine à états (TrackState) pour chaque personne
- Zone morte (hystérésis) autour de la ligne virtuelle
- Phase de warm-up pour calibrer l'effectif initial
- Gestion des occultations (disparition ≠ sortie)
- Réassociation ReID des pistes perdues

Formule : occupancy = initial_occupancy + total_in + total_new_presences - total_out
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from occupancy_types import LogicalTrack, OriginType, TrackState, Zone


# ---------------------------------------------------------------------------
# Paramètres par défaut (surchargeables via config YAML ou CLI)
# ---------------------------------------------------------------------------
DEFAULT_DEAD_ZONE_MARGIN = 30.0      # Pixels de zone morte autour de la ligne
DEFAULT_INIT_DURATION_FRAMES = 15    # Frames de warm-up (~500ms à 30fps)
DEFAULT_CONFIRMATION_THRESHOLD = 15  # Frames consécutives pour confirmer nouvelle présence
DEFAULT_GRACE_PERIOD_FRAMES = 300    # Frames avant purge d'une piste occultée
DEFAULT_REID_THRESHOLD = 0.78        # Seuil de similarité ReID
DEFAULT_TRAJECTORY_MAXLEN = 60       # Taille max de l'historique de trajectoire


# ---------------------------------------------------------------------------
# Géométrie vectorielle
# ---------------------------------------------------------------------------
def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    """Orientation CCW stricte (counter-clockwise)."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def check_line_crossing(
    p1: tuple[float, float],
    p2: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> bool:
    """Intersection stricte entre le segment [p1→p2] et [line_start→line_end] via CCW."""
    return (
        _ccw(p1, line_start, line_end) != _ccw(p2, line_start, line_end)
        and _ccw(p1, p2, line_start) != _ccw(p1, p2, line_end)
    )


def signed_perpendicular_distance(
    pt: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    """Distance orthogonale signée d'un point par rapport à la ligne.

    Positif = côté intérieur (salle), négatif = côté extérieur.
    """
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        return 0.0
    return float(((pt[0] - line_start[0]) * dy - (pt[1] - line_start[1]) * dx) / norm)


def classify_zone(
    dist: float,
    dead_zone_margin: float,
) -> Zone:
    """Classifie un point selon sa distance signée à la ligne."""
    if abs(dist) <= dead_zone_margin:
        return Zone.MORTE
    return Zone.INTERIEURE if dist > 0 else Zone.EXTERIEURE


def compute_anchor(
    box: np.ndarray,
    is_zenithal: bool = False,
) -> tuple[float, float]:
    """Calcule le point d'ancrage d'une bounding box.

    Perspective standard : centre bas [x_center, y_max] (pieds).
    Vue zénithale : centre de la boîte [x_center, y_center].
    """
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    x_center = (x1 + x2) / 2.0
    if is_zenithal:
        return (x_center, (y1 + y2) / 2.0)
    return (x_center, y2)


# ---------------------------------------------------------------------------
# Couleurs par état
# ---------------------------------------------------------------------------
STATE_COLORS: dict[TrackState, tuple[int, int, int]] = {
    TrackState.PRESENTE: (0, 255, 0),           # Vert
    TrackState.NOUVELLE_PRESENCE: (0, 255, 0),   # Vert
    TrackState.OCCULTEE: (0, 165, 255),          # Orange
    TrackState.EN_ZONE_LIGNE: (0, 255, 255),     # Jaune
    TrackState.SORTIE_CONFIRMEE: (0, 0, 255),    # Rouge
    TrackState.A_RETOURNE: (0, 255, 0),          # Vert
}


class OccupancyManager:
    """Gestionnaire d'occupation par machine à états finis.

    Gère le registre de LogicalTrack, détecte les franchissements
    vectoriels, applique l'hystérésis et calcule l'effectif.

    Args:
        line_p1: Premier point de la ligne (coordonnées normalisées 0–1).
        line_p2: Second point de la ligne (coordonnées normalisées 0–1).
        dead_zone_margin: Épaisseur de la zone morte en pixels.
        init_duration_frames: Nombre de frames de warm-up.
        confirmation_threshold: Frames consécutives pour confirmer une nouvelle présence.
        grace_period_frames: Délai avant purge d'une piste occultée.
        reid_threshold: Seuil minimal de similarité ReID.
        is_zenithal: Utiliser le centre de boîte au lieu du bas.
    """

    def __init__(
        self,
        line_p1: tuple[float, float],
        line_p2: tuple[float, float],
        line2_p1: tuple[float, float] | None = None,
        line2_p2: tuple[float, float] | None = None,
        dead_zone_margin: float = DEFAULT_DEAD_ZONE_MARGIN,
        init_duration_frames: int = DEFAULT_INIT_DURATION_FRAMES,
        confirmation_threshold: int = DEFAULT_CONFIRMATION_THRESHOLD,
        grace_period_frames: int = DEFAULT_GRACE_PERIOD_FRAMES,
        reid_threshold: float = DEFAULT_REID_THRESHOLD,
        is_zenithal: bool = False,
    ) -> None:
        # Ligne virtuelle (normalisée)
        self.line_p1 = line_p1
        self.line_p2 = line_p2
        self.line2_p1 = line2_p1
        self.line2_p2 = line2_p2

        # Paramètres
        self.dead_zone_margin = dead_zone_margin
        self.init_duration_frames = init_duration_frames
        self.confirmation_threshold = confirmation_threshold
        self.grace_period_frames = grace_period_frames
        self.reid_threshold = reid_threshold
        self.is_zenithal = is_zenithal

        # Registre des pistes logiques (clé = logical_id)
        self.tracks: dict[int, LogicalTrack] = {}
        # Mapping track_id technique → logical_id
        self.tech_to_logical: dict[int, int] = {}
        self.next_logical_id: int = 1

        # Compteurs
        self.initial_occupancy: int = 0
        self.total_in: int = 0
        self.total_out: int = 0
        self.total_new_presences: int = 0

        # Phase
        self._warmup_candidates: dict[int, int] = {}  # logical_id → frames consécutives intérieures
        self._warmup_done: bool = False

    # -----------------------------------------------------------------------
    # Propriétés
    # -----------------------------------------------------------------------
    @property
    def occupancy_count(self) -> int:
        """Effectif logique courant."""
        return max(0, self.initial_occupancy + self.total_in + self.total_new_presences - self.total_out)

    @property
    def phase(self) -> str:
        return "WARMUP" if not self._warmup_done else "RUNNING"

    @property
    def visible_count(self) -> int:
        """Nombre de pistes actuellement visibles (non occultées, non sorties)."""
        return sum(
            1 for t in self.tracks.values()
            if t.state not in (TrackState.OCCULTEE, TrackState.SORTIE_CONFIRMEE)
        )

    @property
    def occluded_count(self) -> int:
        """Nombre de pistes actuellement occultées."""
        return sum(1 for t in self.tracks.values() if t.state == TrackState.OCCULTEE)

    # -----------------------------------------------------------------------
    # Ligne en pixels
    # -----------------------------------------------------------------------
    def line_px(self, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Convertit la ligne normalisée en coordonnées pixels."""
        return (
            (self.line_p1[0] * width, self.line_p1[1] * height),
            (self.line_p2[0] * width, self.line_p2[1] * height),
        )

    def line2_px(self, width: int, height: int):
        if self.line2_p1 is None or self.line2_p2 is None:
            return None
        return (
            (self.line2_p1[0] * width, self.line2_p1[1] * height),
            (self.line2_p2[0] * width, self.line2_p2[1] * height),
        )

    def classify_point(self, point, width: int, height: int) -> Zone:
        line1 = self.line_px(width, height)
        if self.line2_p1 is None or self.line2_p2 is None:
            return classify_zone(signed_perpendicular_distance(point, *line1), self.dead_zone_margin)
        line2 = self.line2_px(width, height)
        d1 = signed_perpendicular_distance(point, *line1)
        d2 = signed_perpendicular_distance(point, *line2)
        if d1 > self.dead_zone_margin:
            return Zone.INTERIEURE
        if d2 < -self.dead_zone_margin:
            return Zone.EXTERIEURE
        return Zone.MORTE

    # -----------------------------------------------------------------------
    # Traitement d'une frame
    # -----------------------------------------------------------------------
    def process_frame(
        self,
        candidates: list[tuple[int, int, tuple[float, float], np.ndarray, float]],
        frame: np.ndarray,
        frame_index: int,
        features: dict[int, np.ndarray] | None = None,
    ) -> list[dict]:
        """Traite les détections d'une frame et met à jour l'état d'occupation.

        Args:
            candidates: Liste de (track_id, display_id, anchor_point, box, confidence).
            frame: Image brute de la frame courante.
            frame_index: Indice de la frame.
            features: Dictionnaire optionnel display_id → vecteur d'apparence ReID.

        Returns:
            Liste d'événements générés (dicts avec type, id, direction, etc.).
        """
        h, w = frame.shape[:2]
        l_start, l_end = self.line_px(w, h)
        events: list[dict] = []
        seen_logical_ids: set[int] = set()

        for track_id, display_id, anchor, box, conf in candidates:
            dist = signed_perpendicular_distance(anchor, l_start, l_end)
            zone = self.classify_point(anchor, w, h)

            # -- Association technique → logique --
            logical_id = self._get_or_create_logical(
                track_id, display_id, anchor, frame_index, conf, features, initial_zone=zone,
            )
            seen_logical_ids.add(logical_id)
            track = self.tracks[logical_id]

            # Mettre à jour la position et la confiance
            prev_pos = track.last_position
            track.last_position = anchor
            track.last_seen_frame = frame_index
            track.confidence = conf
            track.trajectory_history.append(anchor)
            if len(track.trajectory_history) > DEFAULT_TRAJECTORY_MAXLEN:
                track.trajectory_history = track.trajectory_history[-DEFAULT_TRAJECTORY_MAXLEN:]

            # Mettre à jour le vecteur ReID si disponible
            if features and display_id in features:
                feat = features[display_id]
                if feat is not None:
                    track.feature_vector = feat

            # -- Phase warm-up --
            if not self._warmup_done:
                if frame_index <= self.init_duration_frames:
                    self._handle_warmup(logical_id, zone, frame_index)
                    continue
                else:
                    self._finalize_warmup()

            # -- Réactivation d'une piste occultée --
            if track.state == TrackState.OCCULTEE:
                if zone == Zone.INTERIEURE:
                    track.state = TrackState.PRESENTE
                    print(f"[REAPPEAR] ID {logical_id} réapparu côté intérieur → PRESENTE")
                elif zone == Zone.EXTERIEURE:
                    # Réapparition côté extérieur → maintenir comme sortie
                    track.state = TrackState.SORTIE_CONFIRMEE
                else:
                    track.state = TrackState.EN_ZONE_LIGNE
                continue

            # -- Gestion de la zone morte --
            if zone == Zone.MORTE and self.line2_p1 is None:
                if track.state not in (TrackState.EN_ZONE_LIGNE, TrackState.SORTIE_CONFIRMEE):
                    track.state = TrackState.EN_ZONE_LIGNE
                elif track.state == TrackState.SORTIE_CONFIRMEE:
                    track.pending_direction = "IN"
                continue

            # -- Détection de franchissement vectoriel (CCW) --
            if prev_pos is not None and prev_pos != anchor:
                crossed = check_line_crossing(prev_pos, anchor, l_start, l_end)
                # Avec un flux temps réel, une frame peut être perdue et le
                # point peut passer directement de l'extérieur à l'intérieur.
                # La transition signée complète reste alors un franchissement.
                prev_dist = signed_perpendicular_distance(prev_pos, l_start, l_end)
                if track.state == TrackState.SORTIE_CONFIRMEE and zone == Zone.INTERIEURE:
                    crossed = crossed or (prev_dist < -self.dead_zone_margin and dist > self.dead_zone_margin)
            else:
                crossed = False

            # -- Machine à états --
            if self.line2_p1 is not None and self.line2_p2 is not None:
                l2 = self.line2_px(w, h)
                crossed2 = prev_pos is not None and check_line_crossing(prev_pos, anchor, *l2)
                evt = self._transition_double(track, logical_id, zone, crossed, crossed2, frame_index)
            else:
                evt = self._transition(track, logical_id, zone, crossed, dist, frame_index)
            if evt:
                events.append(evt)

        # -- Gestion des pistes disparues (occultation) --
        self._handle_missing_tracks(seen_logical_ids, frame_index)

        return events

    def _transition_double(self, track, logical_id, zone, crossed1, crossed2, frame_index):
        """Valide un passage seulement après les deux lignes du sas."""
        if track.state == TrackState.SORTIE_CONFIRMEE:
            if crossed2:
                track.pending_direction = "IN"
                track.state = TrackState.EN_ZONE_LIGNE
            return None
        if track.state == TrackState.PRESENTE and crossed1:
            track.pending_direction = "OUT"
            track.state = TrackState.EN_ZONE_LIGNE
            return None
        if track.state == TrackState.EN_ZONE_LIGNE:
            if track.pending_direction == "OUT":
                if zone == Zone.INTERIEURE and crossed1:
                    track.pending_direction = None
                    track.state = TrackState.PRESENTE
                elif crossed2 and zone == Zone.EXTERIEURE and track.counted_in_occupancy:
                    track.pending_direction = None
                    track.state = TrackState.SORTIE_CONFIRMEE
                    track.counted_in_occupancy = False
                    track.is_counted_out = True
                    self.total_out += 1
                    print(f"[COUNT] ID {logical_id} → OUT (sas, Total OUT: {self.total_out})")
                    return {"type": "OUT", "id": logical_id, "frame": frame_index, "reason": "sas"}
            elif track.pending_direction == "IN":
                if zone == Zone.EXTERIEURE and crossed2:
                    track.pending_direction = None
                    track.state = TrackState.SORTIE_CONFIRMEE
                elif crossed1 and zone == Zone.INTERIEURE:
                    track.pending_direction = None
                    track.state = TrackState.PRESENTE
                    track.counted_in_occupancy = True
                    track.is_counted_out = False
                    self.total_in += 1
                    print(f"[COUNT] ID {logical_id} → IN (sas, Total IN: {self.total_in})")
                    return {"type": "IN", "id": logical_id, "frame": frame_index, "reason": "sas"}
        return None

    # -----------------------------------------------------------------------
    # Machine à états interne
    # -----------------------------------------------------------------------
    def _transition(
        self,
        track: LogicalTrack,
        logical_id: int,
        zone: Zone,
        crossed: bool,
        dist: float,
        frame_index: int,
    ) -> Optional[dict]:
        """Applique les transitions d'état et retourne un événement si comptage."""
        state = track.state

        # --- Entrée depuis l'extérieur (sortie confirmée ou piste venant de l'extérieur) ---
        if state == TrackState.SORTIE_CONFIRMEE:
            if zone == Zone.INTERIEURE and (crossed or track.pending_direction == "IN"):
                track.state = TrackState.A_RETOURNE if track.is_counted_out else TrackState.PRESENTE
                track.counted_in_occupancy = True
                track.is_counted_out = False
                track.pending_direction = None
                self.total_in += 1
                label = "retour" if track.state == TrackState.A_RETOURNE else "entrée"
                print(f"[COUNT] ID {logical_id} → IN ({label}, Total IN: {self.total_in})")
                return {"type": "IN", "id": logical_id, "frame": frame_index, "reason": label}
            return None

        # --- Personne en zone intérieure ---
        if zone == Zone.INTERIEURE:
            if crossed and state == TrackState.EN_ZONE_LIGNE and not track.counted_in_occupancy:
                # Traversée complète de la zone morte depuis l'extérieur
                track.state = TrackState.PRESENTE
                track.counted_in_occupancy = True
                self.total_in += 1
                print(f"[COUNT] ID {logical_id} → IN (Total IN: {self.total_in})")
                return {"type": "IN", "id": logical_id, "frame": frame_index, "reason": "entrée"}
            elif state == TrackState.EN_ZONE_LIGNE:
                # Retour en arrière depuis la zone morte vers l'intérieur sans franchissement
                track.state = TrackState.PRESENTE
            elif state == TrackState.NOUVELLE_PRESENCE:
                track.state = TrackState.PRESENTE

            # Confirmation d'une nouvelle présence intérieure
            if not track.counted_in_occupancy and track.origin == OriginType.APPARITION_INTERIEURE:
                track.consecutive_interior_frames += 1
                if track.consecutive_interior_frames >= self.confirmation_threshold:
                    track.state = TrackState.NOUVELLE_PRESENCE
                    track.counted_in_occupancy = True
                    self.total_new_presences += 1
                    print(f"[NEW] ID {logical_id} → NOUVELLE PRESENCE confirmée (Total NEW: {self.total_new_presences})")
                    return {"type": "NEW", "id": logical_id, "frame": frame_index}

            return None

        # --- Personne en zone extérieure ---
        if zone == Zone.EXTERIEURE:
            if crossed and state in (TrackState.PRESENTE, TrackState.EN_ZONE_LIGNE, TrackState.A_RETOURNE, TrackState.NOUVELLE_PRESENCE):
                # Franchissement validé : OUT
                if track.counted_in_occupancy:
                    track.state = TrackState.SORTIE_CONFIRMEE
                    track.counted_in_occupancy = False
                    track.is_counted_out = True
                    self.total_out += 1
                    print(f"[COUNT] ID {logical_id} → OUT (Total OUT: {self.total_out})")
                    return {"type": "OUT", "id": logical_id, "frame": frame_index}
                else:
                    # Personne jamais comptabilisée comme présente → sortie sans décrémentation
                    track.state = TrackState.SORTIE_CONFIRMEE
                    print(f"[SKIP] ID {logical_id} → OUT ignoré (jamais comptabilisé)")
            elif not crossed and state == TrackState.EN_ZONE_LIGNE:
                # Apparition directe côté extérieur depuis la zone morte sans croisement
                track.state = TrackState.SORTIE_CONFIRMEE

            # Entrée par franchissement depuis l'extérieur
            if crossed and state == TrackState.SORTIE_CONFIRMEE:
                # Ce cas est géré en haut de la méthode
                pass

            return None

        return None

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------
    def _handle_warmup(self, logical_id: int, zone: Zone, frame_index: int) -> None:
        """Accumule les détections stables pendant le warm-up."""
        if zone == Zone.INTERIEURE:
            self._warmup_candidates[logical_id] = self._warmup_candidates.get(logical_id, 0) + 1
        else:
            # Réinitialiser le compteur si la personne quitte l'intérieur
            self._warmup_candidates.pop(logical_id, None)

    def _finalize_warmup(self) -> None:
        """Finalise le warm-up : compter les personnes stables comme effectif initial."""
        stable_threshold = max(1, min(5, self.init_duration_frames // 3))
        for logical_id, count in self._warmup_candidates.items():
            if count >= stable_threshold and logical_id in self.tracks:
                self.tracks[logical_id].state = TrackState.PRESENTE
                self.tracks[logical_id].origin = OriginType.INITIALISATION
                self.tracks[logical_id].counted_in_occupancy = True
                self.initial_occupancy += 1
        self._warmup_done = True
        self._warmup_candidates.clear()
        print(f"[WARMUP] Phase terminée. Effectif initial : {self.initial_occupancy}")

    # -----------------------------------------------------------------------
    # Association et création de pistes
    # -----------------------------------------------------------------------
    def _get_or_create_logical(
        self,
        track_id: int,
        display_id: int,
        anchor: tuple[float, float],
        frame_index: int,
        conf: float,
        features: dict[int, np.ndarray] | None,
        initial_zone: Zone = Zone.INTERIEURE,
    ) -> int:
        """Retourne le logical_id associé au track_id, ou en crée un nouveau."""
        # Vérifier si le track_id a déjà un logical_id
        if track_id in self.tech_to_logical:
            return self.tech_to_logical[track_id]

        # Tentative de réassociation ReID avec une piste occultée
        if features and display_id in features:
            new_feat = features[display_id]
            if new_feat is not None:
                match_id = self._try_reid_match(new_feat, frame_index)
                if match_id is not None:
                    self.tech_to_logical[track_id] = match_id
                    self.tracks[match_id].technical_track_id = track_id
                    print(f"[RE-ID OCCUPANCY] track_id={track_id} réassocié → logical_id={match_id}")
                    return match_id

        # Création d'une nouvelle piste logique
        logical_id = self.next_logical_id
        self.next_logical_id += 1
        self.tech_to_logical[track_id] = logical_id

        feat = features.get(display_id) if features else None
        if initial_zone == Zone.EXTERIEURE:
            init_state = TrackState.SORTIE_CONFIRMEE
            init_origin = OriginType.ENTREE_LIGNE
        elif initial_zone == Zone.MORTE:
            init_state = TrackState.EN_ZONE_LIGNE
            init_origin = OriginType.APPARITION_INTERIEURE
        else:
            init_state = TrackState.PRESENTE
            init_origin = OriginType.APPARITION_INTERIEURE

        track = LogicalTrack(
            logical_id=logical_id,
            technical_track_id=track_id,
            state=init_state,
            origin=init_origin,
            last_position=anchor,
            last_seen_frame=frame_index,
            feature_vector=feat,
            confidence=conf,
        )
        self.tracks[logical_id] = track
        return logical_id

    def _try_reid_match(self, feature: np.ndarray, frame_index: int) -> Optional[int]:
        """Cherche une correspondance ReID parmi les pistes occultées."""
        best_id: Optional[int] = None
        best_score = -1.0
        for lid, track in self.tracks.items():
            if track.state not in (TrackState.OCCULTEE, TrackState.SORTIE_CONFIRMEE):
                continue
            if frame_index - track.last_seen_frame > self.grace_period_frames:
                continue
            if track.feature_vector is None:
                continue
            score = self._cosine_similarity(feature, track.feature_vector)
            if score > best_score:
                best_id, best_score = lid, score

        if best_id is not None and best_score >= self.reid_threshold:
            return best_id
        return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

    # -----------------------------------------------------------------------
    # Gestion des pistes disparues
    # -----------------------------------------------------------------------
    def _handle_missing_tracks(self, seen_ids: set[int], frame_index: int) -> None:
        """Marque comme occultées les pistes non vues, purge les expirées."""
        to_purge: list[int] = []
        for lid, track in self.tracks.items():
            if lid in seen_ids:
                continue
            if track.state == TrackState.SORTIE_CONFIRMEE:
                # Personne déjà sortie : ne pas toucher au compteur
                if frame_index - track.last_seen_frame > self.grace_period_frames:
                    to_purge.append(lid)
                continue
            if track.state != TrackState.OCCULTEE:
                track.state = TrackState.OCCULTEE
            if frame_index - track.last_seen_frame > self.grace_period_frames:
                to_purge.append(lid)

        for lid in to_purge:
            track = self.tracks.pop(lid, None)
            if track:
                # Retirer le mapping technique
                keys_to_remove = [k for k, v in self.tech_to_logical.items() if v == lid]
                for k in keys_to_remove:
                    del self.tech_to_logical[k]

    # -----------------------------------------------------------------------
    # Rendu visuel
    # -----------------------------------------------------------------------
    def draw_overlay(
        self,
        frame: np.ndarray,
        candidates: list[tuple[int, int, tuple[float, float], np.ndarray, float]],
        visible_count: int,
    ) -> None:
        """Dessine la ligne, les bounding boxes colorées par état, et le HUD."""
        h, w = frame.shape[:2]
        l_start, l_end = self.line_px(w, h)

        # -- Ligne virtuelle --
        cv2.line(frame, _to_int(l_start), _to_int(l_end), (0, 255, 0), 3, cv2.LINE_AA)

        # -- Zone morte semi-transparente --
        self._draw_dead_zone(frame, l_start, l_end)

        # -- Bounding boxes colorées par état --
        for track_id, display_id, anchor, box, conf in candidates:
            logical_id = self.tech_to_logical.get(track_id)
            if logical_id and logical_id in self.tracks:
                state = self.tracks[logical_id].state
            else:
                state = TrackState.PRESENTE
            color = STATE_COLORS.get(state, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(anchor[0]), int(anchor[1])), 5, (0, 0, 255), -1)

            label = f"ID:{display_id} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # -- Bounding boxes des pistes occultées (dernière position connue) --
        for lid, track in self.tracks.items():
            if track.state == TrackState.OCCULTEE:
                pos = track.last_position
                color = STATE_COLORS[TrackState.OCCULTEE]
                r = 15
                cx, cy = int(pos[0]), int(pos[1])
                cv2.circle(frame, (cx, cy), r, color, 2)
                cv2.putText(frame, f"?{lid}", (cx - 10, cy - r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # -- HUD --
        self._draw_hud(frame, visible_count)

    def _draw_dead_zone(
        self,
        frame: np.ndarray,
        l_start: tuple[float, float],
        l_end: tuple[float, float],
    ) -> None:
        """Dessine une zone morte semi-transparente autour de la ligne."""
        dx = l_end[0] - l_start[0]
        dy = l_end[1] - l_start[1]
        length = max(float(np.hypot(dx, dy)), 1.0)
        nx, ny = -dy / length, dx / length  # Normale unitaire
        m = self.dead_zone_margin

        pts = np.array([
            [l_start[0] + nx * m, l_start[1] + ny * m],
            [l_end[0] + nx * m, l_end[1] + ny * m],
            [l_end[0] - nx * m, l_end[1] - ny * m],
            [l_start[0] - nx * m, l_start[1] - ny * m],
        ], dtype=np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def _draw_hud(self, frame: np.ndarray, visible_count: int) -> None:
        """Dessine le panneau de contrôle d'occupation."""
        h, w = frame.shape[:2]
        scale = max(0.5, min(1.0, w / 1280.0))

        lines = [
            f"OCCUPANCY: {self.occupancy_count}",
            f"IN: {self.total_in}  |  OUT: {self.total_out}  |  NEW: {self.total_new_presences}",
            f"Visible: {visible_count}  |  Occluded: {self.occluded_count}",
            f"Phase: {self.phase}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 0.5 * scale
        th = max(1, int(1.5 * scale))
        line_h = int(22 * scale)
        pad = int(10 * scale)

        # Mesurer la largeur max du texte
        max_text_w = max(cv2.getTextSize(l, font, f_scale, th)[0][0] for l in lines)
        box_w = max_text_w + 2 * pad
        box_h = len(lines) * line_h + 2 * pad

        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (120, 120, 120), 1)

        colors = [(0, 255, 255), (0, 255, 0), (255, 255, 255), (180, 180, 180)]
        for i, (text, color) in enumerate(zip(lines, colors)):
            y = 10 + pad + (i + 1) * line_h - int(4 * scale)
            cv2.putText(frame, text, (10 + pad, y), font, f_scale, color, th, cv2.LINE_AA)


def _to_int(pt: tuple[float, float]) -> tuple[int, int]:
    return (int(pt[0]), int(pt[1]))
