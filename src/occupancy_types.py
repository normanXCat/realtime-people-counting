"""Structures de données du gestionnaire d'occupation.

Ce module a été nettoyé lors de l'audit (§2.3 et §2.2 h) :

- :class:`TrackState` provient désormais de :mod:`fsm` : les états
  ``EN_ZONE_LIGNE`` (jamais assigné) et ``A_RETOURNE`` (jamais lu) ont disparu
  au profit des états réellement atteints de la spec 4.1.
- :class:`Zone` vit dans :mod:`geometry`, seule implémentation de la géométrie.
- ``LogicalTrack`` est remplacé par :class:`PersonTrack`, réduit aux champs
  effectivement lus ou écrits. Les onze champs jamais utilisés
  (``previous_zone``, ``exterior_streak``, ``interior_streak``,
  ``exit_l1_crossed``, ``entry_l2_crossed``, ``previous_bottom_l1``,
  ``previous_top_l2``, ``previous_top_l1``, ``previous_bottom_l2``,
  ``crossing_in_streak``, ``crossing_out_streak``) ainsi que
  ``trajectory_history`` (écrit mais jamais lu) ont été supprimés.

L'identifiant porteur du comptage est ``person_id`` (spec 3.1). Le
``technical_track_id`` est conservé à titre de traçabilité uniquement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from fsm import Action, Approach, Decision, FsmContext, TrackState  # noqa: F401 (réexport)

__all__ = [
    "TrackState",
    "Action",
    "Approach",
    "FsmContext",
    "Decision",
    "OriginType",
    "PersonTrack",
    "OccupancySnapshot",
]


class OriginType(Enum):
    """Comment la personne est entrée dans l'effectif."""

    INITIALISATION = auto()          # présente au démarrage (warm-up)
    ENTREE_LIGNE = auto()            # entrée par franchissement confirmé
    APPARITION_INTERIEURE = auto()   # apparue directement dans la salle (NEW)
    REASSOCIATION = auto()           # récupérée par la galerie long terme


@dataclass
class PersonTrack:
    """État complet d'une personne suivie par le système d'occupation.

    Attributes:
        person_id: identifiant logique stable, **seul** porteur du comptage.
        state: état courant de la machine à états (:mod:`fsm`).
        inside_occupancy: la personne est-elle comptée comme présente ?
            Passe à vrai sur ``COMPTE_ENTREE`` / ``NOUVELLE_PRESENCE`` /
            ``EMPLACE_INITIAL``, à faux sur ``COMPTE_SORTIE``.

    Le statut « purgée sans sortie observée » (borne haute de l'occupation,
    spec 4.4) n'est pas dupliqué ici : il est détenu par
    ``OccupancyManager``, car il doit survivre à la disparition de cette
    structure (libération de la galerie).
    """

    person_id: int
    state: TrackState = TrackState.INITIALISATION
    origin: OriginType = OriginType.APPARITION_INTERIEURE
    technical_track_id: int = -1
    anchor: tuple[float, float] = (0.0, 0.0)
    previous_anchor: tuple[float, float] | None = None
    bbox: np.ndarray | None = None
    bbox_height: float = 0.0
    last_seen_frame: int = 0
    last_seen_s: float = 0.0
    feature_vector: np.ndarray | None = None
    confidence: float = 0.0
    inside_occupancy: bool = False
    provisional: bool = False
    approach: Approach = Approach.INTERIEUR
    pending_crossing: str | None = None
    interior_streak: int = 0
    crossed: int = 0
    occlusion_events: int = 0
    anchor_suspend_streak: int = 0
    last_rule: str = ""
    last_zone: str = ""
    last_distance: float = 0.0
    #: Diagnostic : ensemble des track_id techniques ayant porté ce person_id.
    aliases: set[int] = field(default_factory=set)

    @property
    def is_active(self) -> bool:
        """La personne est-elle encore susceptible de produire un comptage ?"""
        return self.state is not TrackState.ABSENTE

    @property
    def is_visible(self) -> bool:
        return self.state in (
            TrackState.INITIALISATION,
            TrackState.EXTERIEUR,
            TrackState.PRESENTE,
            TrackState.EN_ZONE_MORTE,
            TrackState.ENTREE_EN_COURS,
            TrackState.SORTIE_EN_COURS,
        )

    @property
    def lifetime_s(self) -> float:
        return self.last_seen_s


@dataclass(frozen=True)
class OccupancySnapshot:
    """Occupation publiée à intervalle régulier (spec 4.4)."""

    confirmed: int
    uncertain: int
    range_low: int
    range_high: int
    visible: int
    occluded: int
    initial: int
    total_in: int
    total_out: int
    total_new: int

    def to_fields(self) -> dict[str, object]:
        return {
            "occupancy_confirmed": self.confirmed,
            "occupancy_uncertain": self.uncertain,
            "occupancy_range": [self.range_low, self.range_high],
            "visible_count": self.visible,
            "occluded_count": self.occluded,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_new": self.total_new,
            "identities_live": self.confirmed + self.uncertain,
        }
