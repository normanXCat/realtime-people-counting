"""Journal d'événements structuré (JSON Lines) et schéma versionné.

Deux responsabilités, volontairement séparées de la logique métier :

1. **Schéma** (:data:`EVENT_SCHEMA`) : liste explicite des champs obligatoires
   par type d'événement (spec 6.2). :func:`validate_event` refuse tout
   événement incomplet ou mal typé. Les événements de franchissement portent
   obligatoirement un champ ``direction`` exploitable par le script
   d'évaluation.
2. **Écriture incrémentale** (:class:`EventLogger`) : une ligne JSON par
   événement, ``flush`` après chaque événement (ou par petit lot configurable).
   Une interruption brutale ne doit jamais faire perdre les événements déjà
   produits (spec 6.1).

Aucune décision métier ne doit être silencieuse (règle 0.4) : c'est ici que se
matérialise l'exigence « produire un événement explicite plutôt que choisir
arbitrairement ».
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

SCHEMA_VERSION = 1


class EventSink(Protocol):
    """Contrat minimal d'un collecteur d'événements.

    Implémenté par :class:`EventLogger` (écriture disque incrémentale) et par
    :class:`ListSink` (collecte en mémoire pour les tests et les runs hors
    ligne). Les composants métier ne dépendent que de ce contrat : ils ne
    connaissent ni fichier, ni format de sérialisation.
    """

    def emit(
        self, event_type: str, timestamp_s: float, frame_index: int, **fields: Any
    ) -> dict[str, Any]:  # pragma: no cover - protocole
        ...

    def write(self, event: Mapping[str, Any]) -> None:  # pragma: no cover - protocole
        ...

#: Directions autorisées pour les événements de franchissement (spec 6.2).
DIRECTIONS = ("in", "out", "indeterminate")

#: Champs présents dans **tous** les événements.
COMMON_REQUIRED: tuple[str, ...] = (
    "schema_version",
    "event_id",
    "session_id",
    "type",
    "timestamp_s",
    "frame_index",
)


@dataclass(frozen=True)
class EventSpec:
    """Champs obligatoires et optionnels d'un type d'événement."""

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    #: Si vrai, ``direction`` est obligatoire et doit valoir in/out/indeterminate.
    crossing: bool = False
    #: Variantes acceptées : **au moins un** de ces groupes de champs doit être
    #: entièrement présent. Permet de décrire un événement à deux formes (ex. un
    #: échange de paire et une correction de groupe à 3+ pistes) sans affaiblir la
    #: validation : chaque variante reste contrainte, seule l'alternative change.
    any_of: tuple[tuple[str, ...], ...] = ()


_CROSSING_FIELDS = ("person_id", "technical_track_id", "direction")

EVENT_SCHEMA: dict[str, EventSpec] = {
    # -- Cycle de vie de la session ---------------------------------------
    "SESSION_START": EventSpec(
        required=("config_path", "source", "model_path", "model_sha256"),
        optional=(
            "fps_source", "resolved_config_path", "line_policy", "mode",
            "tracker_high_thresh", "tracker_low_thresh", "tracker_new_thresh",
            "warmup_min_frames", "warmup_seconds",
        ),
    ),
    "SESSION_END": EventSpec(
        required=("frames_processed", "duration_s", "fps_mean"),
        optional=(
            "fps_median", "fps_min", "occupancy_confirmed", "occupancy_observed",
            "occupancy_uncertain", "occupancy_operational", "total_in", "total_out",
            "total_new", "latency_ms",
        ),
    ),
    "SOURCE_END_OF_STREAM": EventSpec(
        required=("frames_processed",), optional=("duration_s",)
    ),
    "SOURCE_NO_DETECTION": EventSpec(
        required=("consecutive_frames", "elapsed_s"), optional=("reconnect_attempt",)
    ),
    "SOURCE_ERROR": EventSpec(required=("error",), optional=("frames_processed",)),
    "WARMUP_END": EventSpec(
        required=("initial_occupancy",),
        optional=(
            "budget_frames", "fps", "warmup_frames_observed", "warmup_elapsed_seconds",
            "warmup_min_frames",
        ),
    ),
    "CONFIG_WARNING": EventSpec(
        required=("code", "details"), optional=("source",)
    ),
    # -- Ligne virtuelle (définie manuellement, jamais par défaut) --------
    "LINE_VALIDATED": EventSpec(
        required=("p1", "p2", "inside_side", "frame_resolution"),
        optional=("length_normalized", "display_scale", "confirmed_at", "calibration_path"),
    ),
    "LINE_CALIBRATION_CANCELLED": EventSpec(
        required=("reason",), optional=("frames_processed",)
    ),
    "LINE_CALIBRATION_UNAVAILABLE": EventSpec(
        required=("reason",), optional=("frames_processed",)
    ),
    # -- Franchissements (portent obligatoirement une direction) ----------
    "IN": EventSpec(required=_CROSSING_FIELDS, optional=("reason", "occupancy_after"), crossing=True),
    "OUT": EventSpec(required=_CROSSING_FIELDS, optional=("reason", "occupancy_after"), crossing=True),
    # NEW est une apparition à l'intérieur : direction = "in" (spec 6.2).
    "NEW": EventSpec(
        required=_CROSSING_FIELDS,
        optional=("reason", "origin", "occupancy_after", "stable_frames"),
        crossing=True,
    ),
    "AMBIGUOUS_CROSSING": EventSpec(
        required=("person_id", "direction", "reason"),
        optional=("technical_track_id", "pending_direction", "last_position", "policy"),
        crossing=True,
    ),
    # -- Cycle de vie d'une identité --------------------------------------
    "OCCLUDED": EventSpec(
        required=("person_id",), optional=("technical_track_id", "last_position", "state_before")
    ),
    # PURGE est **toujours** une purge technique : elle ne vaut jamais sortie.
    # `purge_kind` = "technical" et `is_exit` = false le rendent explicite pour
    # l'audit, la borne haute de l'occupation restant détenue par ailleurs.
    "PURGE": EventSpec(
        required=("person_id", "reason"),
        optional=(
            "last_position", "last_state", "occupancy_impact", "grace_period_frames",
            "lifetime_s", "purge_kind", "is_exit",
        ),
    ),
    "REID_MATCH": EventSpec(
        required=("person_id", "technical_track_id", "similarity"),
        optional=(
            "runner_up_similarity", "candidates_evaluated", "distance", "allowed_distance",
            "winner_margin",
        ),
    ),
    "REID_NEW": EventSpec(
        required=("person_id", "technical_track_id", "reason"),
        optional=("candidates_evaluated", "best_similarity", "deferred"),
    ),
    "REID_AMBIGUOUS": EventSpec(
        required=("technical_track_id", "best_similarity", "runner_up_similarity", "safety_margin"),
        optional=(
            "person_id", "candidates", "resolution", "distance", "allowed_distance",
            "reason", "descriptor_reliable",
        ),
    ),
    # Un nouvel identifiant technique est rattaché à une identité connue :
    # c'est la trace explicite d'une perte/reprise de piste (spec 3 du prompt).
    "TECHNICAL_ID_CHANGED": EventSpec(
        required=("person_id", "technical_track_id", "decision"),
        optional=(
            "previous_technical_track_id", "similarity", "distance", "allowed_distance",
            "reason", "descriptor_reliable",
        ),
    ),
    # Garde-fou « inversement d'identifiant » (swap) : une piste technique
    # **active en continu** (jamais perdue selon BoT-SORT) change brutalement
    # d'apparence. Ce phénomène se produit à l'intérieur du matching du tracker,
    # avant qu'un identifiant ne disparaisse : aucun TECHNICAL_ID_CHANGED ni
    # TRACK_ID_ABSENT ne le signale. L'événement est un **diagnostic** : le swap
    # n'est pas corrigé automatiquement (sur-corriger un faux positif serait
    # pire), il est seulement rendu mesurable.
    "TRACK_ID_APPEARANCE_DISCONTINUITY": EventSpec(
        required=(
            "technical_track_id",
            "similarity_to_previous_descriptor",
            "reason",
        ),
        optional=(
            "person_id_before", "gap_frames", "previous_descriptor_frame_index",
            "descriptor_age_s", "threshold", "consecutive_frames",
        ),
    ),
    # Correction active des inversions d'identité (swap). Deux formes :
    #   - **paire** (groupe de 2) : le croisement d'apparence entre deux pistes
    #     connues dépasse la marge de sécurité. Champs ``*_a`` / ``*_b`` ;
    #   - **groupe** (3+ pistes mutuellement proches) : résolution globale N×N,
    #     un événement par piste corrigée, avec ``group_size``. Une permutation
    #     circulaire n'est pas décomposable en décisions par paires.
    # Dans les deux formes, ``similarity_direct`` / ``similarity_crossed`` portent la
    # similarité **totale** avant/après correction, et ``margin_applied`` la marge.
    "IDENTITY_SWAP_CORRECTED": EventSpec(
        required=("similarity_direct", "similarity_crossed", "margin_applied", "reason"),
        any_of=(
            (
                "technical_track_id_a",
                "technical_track_id_b",
                "person_id_a_before",
                "person_id_a_after",
                "person_id_b_before",
                "person_id_b_after",
            ),
            (
                "technical_track_id",
                "person_id_before",
                "person_id_after",
                "group_size",
            ),
        ),
        optional=(
            "frame",
            "technical_track_id_a",
            "technical_track_id_b",
            "person_id_a_before",
            "person_id_a_after",
            "person_id_b_before",
            "person_id_b_after",
            "technical_track_id",
            "person_id_before",
            "person_id_after",
            "group_size",
            "group_person_ids",
        ),
    ),
    # Rattachement d'une observation **non trackée** à une identité de galerie
    # (personne assise/immobile dont la réapparition n'atteint pas
    # ``tracker.new_track_thresh``). Événement dédié, distinct de ``REID_MATCH`` :
    # l'impact du chemin de secours reste mesurable séparément et le flag
    # ``reid.long_term.gallery_rescue_enabled`` le désactive sans toucher au reste.
    "REID_GALLERY_RESCUE": EventSpec(
        required=(
            "person_id",
            "technical_track_id",
            "similarity",
            "confidence",
            "reason",
        ),
        optional=(
            "previous_technical_track_id",
            "runner_up_similarity",
            "winner_margin",
            "candidates_evaluated",
            "distance",
            "allowed_distance",
            "threshold_applied",
            "absence_s",
            "strict_reappearance",
        ),
    ),
    # Apparence refusée pour la galerie (jamais remplacée en silence).
    "REID_DESCRIPTOR_REJECTED": EventSpec(
        required=("person_id", "reason"),
        optional=("technical_track_id", "confidence", "bbox_height", "bbox_width", "sharpness", "threshold"),
    ),
    # -- Diagnostics de détection / association (perte de piste) ----------
    "DETECTION_ABSENT": EventSpec(
        required=("reason",),
        optional=("consecutive_frames", "detections", "frames_observed"),
    ),
    "TRACK_ID_ABSENT": EventSpec(
        required=("detections", "reason"),
        optional=("consecutive_frames", "frames_observed"),
    ),
    "DETECTION_LOW_CONFIDENCE": EventSpec(
        required=("detections", "reason"),
        optional=("max_confidence", "track_high_thresh", "consecutive_frames", "frames_observed"),
    ),
    "DETECTION_REJECTED_GEOMETRY": EventSpec(
        required=("reason",),
        optional=(
            "bbox_wh_ratio", "confidence", "frame", "bbox", "min_ratio", "max_ratio",
            "technical_track_id",
        ),
    ),
    # Détection d'une boîte contenant probablement deux personnes (fusion de silhouettes).
    "POSSIBLE_MULTI_PERSON_BOX": EventSpec(
        required=(
            "person_id",
            "current_width",
            "stable_width",
            "ratio",
            "threshold",
        ),
        optional=("frame", "technical_track_id", "bbox"),
    ),
    # -- Cohérence ---------------------------------------------------------
    "INCONSISTENT_STATE": EventSpec(
        required=("kind",),
        optional=("person_id", "technical_track_id", "details", "resolution"),
    ),
    # -- Occupation --------------------------------------------------------
    # ``occupancy_confirmed`` est l'effectif **opérationnel** (bilan
    # initial + IN + NEW - OUT) : il ne diminue que sur une sortie confirmée.
    # ``occupancy_operational`` le répète sous son nom explicite et
    # ``occupancy_observed`` donne la part actuellement observable, de sorte que
    # ``occupancy_observed + occupancy_uncertain == occupancy_confirmed``.
    "OCCUPANCY_SNAPSHOT": EventSpec(
        required=("occupancy_confirmed", "occupancy_uncertain", "occupancy_range"),
        optional=(
            "visible_count", "occluded_count", "total_in", "total_out", "total_new",
            "identities_live", "occupancy_operational", "occupancy_observed",
        ),
    ),
    # -- Traçabilité machine à états et stabilisation ----------------------
    "STATE_TRANSITION": EventSpec(
        required=("person_id", "from_state", "to_state", "rule"),
        optional=("action", "reason", "zone", "distance_signed"),
    ),
    "ANCHOR_UNRELIABLE": EventSpec(
        required=("person_id", "reason"),
        optional=(
            "margin_px", "bbox", "consecutive_frames",
            "previous_height", "current_height", "drop_ratio", "frame",
        ),
    ),
    "STABILIZATION": EventSpec(
        required=("person_id", "raw_bbox", "corrected_bbox", "reason", "threshold"),
        optional=("component", "frame_index_source"),
    ),
    # -- Assistance tête (présence uniquement, jamais franchissement) -------
    "PRESENCE_MAINTAINED_BY_HEAD": EventSpec(
        required=("person_id", "reason", "head_confidence", "feet_anchor_status"),
        optional=("frame", "head_point"),
    ),
    "PRESENCE_EXTENSION_EXPIRED": EventSpec(
        required=("person_id", "duration_s", "max_extension_seconds"),
        optional=("frame", "reason"),
    ),
}


class SchemaError(ValueError):
    """Événement refusé par le schéma : il ne doit pas être journalisé tel quel."""


def validate_event(event: Mapping[str, Any]) -> None:
    """Vérifie qu'un événement respecte le schéma versionné.

    Raises:
        SchemaError: champ obligatoire manquant, type inattendu, ``type``
            inconnu ou ``direction`` absente/invalide.
    """
    missing_common = [key for key in COMMON_REQUIRED if key not in event]
    if missing_common:
        raise SchemaError(f"champs communs manquants : {missing_common}")

    if event["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {event['schema_version']!r} != {SCHEMA_VERSION}"
        )

    event_type = event["type"]
    if event_type not in EVENT_SCHEMA:
        raise SchemaError(
            f"type d'événement inconnu {event_type!r} "
            f"(types connus : {sorted(EVENT_SCHEMA)})"
        )

    spec = EVENT_SCHEMA[event_type]
    missing = [key for key in spec.required if event.get(key) is None]
    if missing:
        raise SchemaError(f"{event_type} : champs obligatoires manquants {missing}")

    if spec.any_of and not any(
        all(event.get(key) is not None for key in variant) for variant in spec.any_of
    ):
        raise SchemaError(
            f"{event_type} : aucun jeu de champs requis n'est complet "
            f"(attendu l'un de {[list(variant) for variant in spec.any_of]})"
        )

    if spec.crossing:
        direction = event.get("direction")
        if direction not in DIRECTIONS:
            raise SchemaError(
                f"{event_type} : direction {direction!r} invalide "
                f"(attendu l'un de {DIRECTIONS})"
            )

    if not isinstance(event["timestamp_s"], (int, float)):
        raise SchemaError(f"{event_type} : timestamp_s doit être numérique")
    if not isinstance(event["frame_index"], int):
        raise SchemaError(f"{event_type} : frame_index doit être un entier")

    for field_name, field_value in event.items():
        if isinstance(field_value, float) and (
            field_value != field_value or field_value in (float("inf"), float("-inf"))
        ):
            raise SchemaError(f"{event_type} : {field_name} n'est pas un flottant fini")


class EventLogger:
    """Écrit les événements en JSON Lines, de façon incrémentale.

    Le fichier est ouvert en append : relancer une session dans le même dossier
    n'écrase pas les événements précédents. Chaque écriture est suivie d'un
    ``flush`` (ou d'un flush par lot si ``flush_every_n_events`` > 1), et le
    flux est synchronisé sur disque à la fermeture.
    """

    def __init__(
        self,
        path: str | Path,
        session_id: str,
        flush_every_n_events: int = 1,
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.flush_every_n_events = max(1, int(flush_every_n_events))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._pending = 0
        self._count = 0
        self._next_event_id = 1

    # -- API ---------------------------------------------------------------
    @property
    def event_count(self) -> int:
        return self._count

    def emit(
        self,
        event_type: str,
        timestamp_s: float,
        frame_index: int,
        **fields: Any,
    ) -> dict[str, Any]:
        """Construit, valide et écrit un événement. Retourne l'événement écrit."""
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"{self.session_id}-{self._next_event_id:06d}",
            "session_id": self.session_id,
            "type": event_type,
            "timestamp_s": round(float(timestamp_s), 4),
            "frame_index": int(frame_index),
        }
        event.update(fields)
        validate_event(event)
        self.write(event)
        self._next_event_id += 1
        return event

    def write(self, event: Mapping[str, Any]) -> None:
        """Écrit un événement déjà construit (validation non refaite)."""
        self._handle.write(json.dumps(event, ensure_ascii=False, default=_json_default))
        self._handle.write("\n")
        self._pending += 1
        self._count += 1
        if self._pending >= self.flush_every_n_events:
            self.flush()

    def write_many(self, events: Iterable[Mapping[str, Any]]) -> None:
        for event in events:
            self.write(event)

    def flush(self) -> None:
        if self._pending:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._pending = 0

    def close(self) -> None:
        if not self._handle.closed:
            self.flush()
            self._handle.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ListSink:
    """Collecteur d'événements en mémoire, au même contrat que :class:`EventLogger`.

    Utilisé par les tests et par les exécutions hors ligne : les événements sont
    construits et validés exactement comme sur disque, sans écrire de fichier.
    """

    def __init__(self, session_id: str = "local") -> None:
        self.session_id = session_id
        self.events: list[dict[str, Any]] = []
        self._next_event_id = 1

    def emit(
        self, event_type: str, timestamp_s: float, frame_index: int, **fields: Any
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"{self.session_id}-{self._next_event_id:06d}",
            "session_id": self.session_id,
            "type": event_type,
            "timestamp_s": round(float(timestamp_s), 4),
            "frame_index": int(frame_index),
        }
        event.update(fields)
        validate_event(event)
        self.events.append(event)
        self._next_event_id += 1
        return event

    def write(self, event: Mapping[str, Any]) -> None:
        validate_event(event)
        self.events.append(dict(event))

    def write_many(self, events: Iterable[Mapping[str, Any]]) -> None:
        for event in events:
            self.write(event)

    def of_type(self, *types: str) -> list[dict[str, Any]]:
        wanted = set(types)
        return [event for event in self.events if event["type"] in wanted]

    def count(self, event_type: str) -> int:
        return sum(1 for event in self.events if event["type"] == event_type)

    def flush(self) -> None:  # compatibilité de contrat
        return None

    def close(self) -> None:  # compatibilité de contrat
        return None

    def __enter__(self) -> "ListSink":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def read_events(path: str | Path, types: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Relit un journal JSON Lines (utilisé par les tests et l'évaluation)."""
    wanted = set(types) if types is not None else None
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"{path}:{line_number} n'est pas une ligne JSON valide : {exc}"
                ) from exc
            validate_event(event)
            if wanted is None or event["type"] in wanted:
                events.append(event)
    return events


def _json_default(value: Any) -> Any:
    """Sérialise les types numpy/Path rencontrés dans les champs optionnels."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Type non sérialisable dans un événement : {type(value)!r}")
