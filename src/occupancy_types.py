"""Structures de données pour le gestionnaire d'occupation robuste.

Ce module définit les types utilisés par OccupancyManager :
- TrackState : machine à états finis (FSM) de chaque piste logique
- OriginType : comment la personne a été détectée pour la première fois
- Zone : classification spatiale par rapport à la ligne virtuelle
- LogicalTrack : état complet d'une personne suivie
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class TrackState(Enum):
    """État d'une piste logique dans la machine à états."""

    PRESENTE = auto()           # Connue dans la salle
    OCCULTEE = auto()           # Perdue temporairement (ex: sous une table)
    EN_ZONE_LIGNE = auto()      # Dans la zone morte autour de la porte
    SORTIE_CONFIRMEE = auto()   # A franchi OUT et s'est éloignée
    A_RETOURNE = auto()         # Personne sortie qui réentre dans la salle
    NOUVELLE_PRESENCE = auto()  # Apparue dans la salle sans franchissement observé


class OriginType(Enum):
    """Comment la personne a été initialement détectée."""

    INITIALISATION = auto()          # Détectée pendant le warm-up
    ENTREE_LIGNE = auto()            # Entrée par franchissement de la ligne
    APPARITION_INTERIEURE = auto()   # Apparue directement dans la salle


class Zone(Enum):
    """Classification spatiale d'un point par rapport à la ligne virtuelle."""

    INTERIEURE = auto()   # Côté salle
    MORTE = auto()        # Bande tampon autour de la ligne
    EXTERIEURE = auto()   # Côté sortie


@dataclass
class LogicalTrack:
    """État complet d'une personne suivie par le système d'occupation."""

    logical_id: int
    technical_track_id: int
    state: TrackState
    origin: OriginType
    last_position: tuple[float, float]              # (x, y) en pixels
    last_seen_frame: int
    feature_vector: np.ndarray | None = None        # ReID embedding
    consecutive_interior_frames: int = 0
    counted_in_occupancy: bool = False
    is_counted_out: bool = False
    confidence: float = 0.0
    trajectory_history: list[tuple[float, float]] = field(default_factory=list)
    previous_zone: Zone = Zone.INTERIEURE
    pending_direction: str | None = None
    exterior_streak: int = 0
    interior_streak: int = 0
    exit_l1_crossed: bool = False
    entry_l2_crossed: bool = False
    previous_bottom_l1: float | None = None
    previous_top_l2: float | None = None
    previous_top_l1: float | None = None
    previous_bottom_l2: float | None = None
