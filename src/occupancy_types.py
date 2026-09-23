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
    height_history: list[float] = field(default_factory=list)
    height_drop_streak: int = 0
    anchor_unreliable_reason: str | None = None
    last_reliable_anchor: tuple[float, float] | None = None
    drop_previous_height: float = 0.0
    drop_current_height: float = 0.0
    drop_ratio: float = 0.0
    #: La piste a-t-elle été observée dans la zone **extérieure** ? Condition qui
    #: interdit ``NEW`` (une personne vue dehors entre, elle n'apparaît pas).
    observed_outside: bool = False
    #: Le report de ``NEW`` (personne comptée occultée à proximité) a déjà été
    #: journalisé : évite une ligne par frame tant que la situation dure.
    new_deferred_reported: bool = False
    last_rule: str = ""
    last_zone: str = ""
    last_distance: float = 0.0
    #: Assistance tête (présence uniquement)
    head_presence_first_s: float | None = None
    head_presence_expired: bool = False
    current_head_point: tuple[float, float] | None = None
    current_head_confidence: float | None = None
    #: Index de la frame où ``current_head_point`` a été observé. Les keypoints
    #: sont extraits de la même détection que la boîte : sans boîte, pas de point
    #: tête. Ce champ rend explicite la **fraîcheur** de la donnée, sans quoi une
    #: piste non observée peut maintenir sa présence sur un point tête figé lors
    #: de la dernière frame où la personne était vue — donnée qui n'a plus aucun
    #: lien avec l'image courante.
    head_point_frame_index: int | None = None
    #: Diagnostic : ensemble des track_id techniques ayant porté ce person_id.
    aliases: set[int] = field(default_factory=set)
    #: Post-Occlusion Recovery (étape 1) : l'identité, restée ``OCCULTEE`` pour
    #: la FSM, est visible sur cette frame grâce à une détection brute
    #: rattachée. N'influe que sur la visibilité, jamais sur le comptage.
    recovery_active: bool = False

    @property
    def is_active(self) -> bool:
        """La personne est-elle encore susceptible de produire un comptage ?"""
        return self.state is not TrackState.ABSENTE

    @property
    def is_visible(self) -> bool:
        if self.recovery_active and self.state is TrackState.OCCULTEE:
            return True
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
    """Occupation publiée à intervalle régulier (spec 4.4).

    Trois valeurs distinctes, jamais confondues (spec 6 du prompt) :

    - ``confirmed`` = ``operational`` : l'effectif **opérationnel**, tenu par le
      bilan ``occupation initiale + IN + NEW - OUT``. Il conserve une personne
      occultée et ne diminue donc **que** sur une sortie confirmée par la machine
      à états — jamais sur une occultation, une expiration de grâce, une absence
      de détection ou une purge technique ;
    - ``observed`` : la part de cet effectif **actuellement observable** ;
    - ``uncertain`` : la part dont l'observation est perdue. Invariant vérifié à
      chaque frame : ``observed + uncertain == operational`` ;
    - ``range_low``/``range_high`` : l'encadrement publié, ``[observed,
      operational]``.
    """

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
    observed: int = 0
    operational: int = 0

    def to_fields(self) -> dict[str, object]:
        return {
            "occupancy_confirmed": self.confirmed,
            "occupancy_observed": self.observed,
            "occupancy_uncertain": self.uncertain,
            "occupancy_range": [self.range_low, self.range_high],
            "occupancy_operational": self.operational,
            "visible_count": self.visible,
            "occluded_count": self.occluded,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_new": self.total_new,
            "identities_live": self.confirmed + self.uncertain,
        }
