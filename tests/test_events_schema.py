"""Niveau 1 — Journal d'événements : schéma versionné et écriture incrémentale.

Couvre les sprec 6.1 (écriture incrémentale, flush), 6.2 (schéma versionné,
champ ``direction`` obligatoire sur les franchissements) et 6.3 (session
identifiée, tout est reconstituable).
"""

from __future__ import annotations

import json

import pytest

from events import (
    EVENT_SCHEMA,
    SCHEMA_VERSION,
    EventLogger,
    ListSink,
    SchemaError,
    read_events,
    validate_event,
)


def base_event(event_type: str = "IN", **fields) -> dict:
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "session-000001",
        "session_id": "session",
        "type": event_type,
        "timestamp_s": 0.5,
        "frame_index": 1,
    }
    event.update(fields)
    return event


def minimal_event(event_type: str) -> dict:
    """Construit un événement minimal valide pour un type donné."""
    spec = EVENT_SCHEMA[event_type]
    required = {name: 1 for name in spec.required}
    # Un type à variantes (ex. ``IDENTITY_SWAP_CORRECTED`` paire / groupe) est
    # minimalement valide avec la **première** variante complétée.
    if spec.any_of:
        required.update({name: 1 for name in spec.any_of[0]})
    if spec.crossing:
        required["direction"] = "in"
    # Les champs de comptage doivent rester numériques et bornés.
    if "occupancy_range" in required:
        required["occupancy_range"] = [0, 0]
    return base_event(event_type, **required)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("event_type", sorted(EVENT_SCHEMA))
def test_every_declared_type_has_a_minimal_valid_event(event_type):
    validate_event(minimal_event(event_type))


@pytest.mark.parametrize("missing", ["schema_version", "event_id", "session_id", "type", "timestamp_s", "frame_index"])
def test_common_fields_are_mandatory(missing):
    event = minimal_event("IN")
    del event[missing]
    with pytest.raises(SchemaError):
        validate_event(event)


def test_unknown_event_type_is_rejected():
    with pytest.raises(SchemaError) as error:
        validate_event(base_event("TELEPORT"))
    assert "inconnu" in str(error.value)


def test_wrong_schema_version_is_rejected():
    event = minimal_event("IN")
    event["schema_version"] = 2
    with pytest.raises(SchemaError):
        validate_event(event)


@pytest.mark.parametrize("event_type", ["IN", "OUT", "NEW", "AMBIGUOUS_CROSSING"])
def test_crossing_events_require_a_valid_direction(event_type):
    event = minimal_event(event_type)
    del event["direction"]
    with pytest.raises(SchemaError):
        validate_event(event)
    event = minimal_event(event_type)
    event["direction"] = "dehors"
    with pytest.raises(SchemaError):
        validate_event(event)
    for direction in ("in", "out", "indeterminate"):
        event = minimal_event(event_type)
        event["direction"] = direction
        validate_event(event)


@pytest.mark.parametrize(
    "event_type,missing",
    [
        ("IN", "person_id"),
        ("OUT", "technical_track_id"),
        ("NEW", "person_id"),
        ("PURGE", "reason"),
        ("REID_MATCH", "similarity"),
        ("REID_AMBIGUOUS", "safety_margin"),
        ("INCONSISTENT_STATE", "kind"),
        ("OCCUPANCY_SNAPSHOT", "occupancy_confirmed"),
        ("OCCUPANCY_SNAPSHOT", "occupancy_range"),
        ("STATE_TRANSITION", "rule"),
        ("ANCHOR_UNRELIABLE", "reason"),
        ("STABILIZATION", "corrected_bbox"),
        ("SOURCE_ERROR", "error"),
        ("SESSION_START", "model_sha256"),
        ("SESSION_END", "duration_s"),
    ],
)
def test_type_specific_required_fields(event_type, missing):
    event = minimal_event(event_type)
    del event[missing]
    with pytest.raises(SchemaError):
        validate_event(event)


def test_identity_swap_corrected_accepts_both_variants():
    """Forme « paire » et forme « groupe » (3+ pistes) sont toutes deux valides."""
    pair = base_event(
        "IDENTITY_SWAP_CORRECTED",
        technical_track_id_a=1,
        technical_track_id_b=2,
        person_id_a_before=1,
        person_id_a_after=2,
        person_id_b_before=2,
        person_id_b_after=1,
        similarity_direct=0.5,
        similarity_crossed=0.9,
        margin_applied=0.08,
        group_size=2,
        reason="cross_appearance_correction",
    )
    validate_event(pair)

    group = base_event(
        "IDENTITY_SWAP_CORRECTED",
        technical_track_id=8,
        person_id_before=2,
        person_id_after=3,
        group_size=3,
        group_person_ids=[2, 3, 1],
        similarity_direct=0.0,
        similarity_crossed=3.0,
        margin_applied=0.08,
        reason="group_appearance_correction",
    )
    validate_event(group)


def test_identity_swap_corrected_rejects_an_incomplete_variant():
    """Ni paire complète ni groupe complet : l'événement est refusé."""
    incomplete = base_event(
        "IDENTITY_SWAP_CORRECTED",
        technical_track_id_a=1,  # paire incomplète (pas de b)
        similarity_direct=0.5,
        similarity_crossed=0.9,
        margin_applied=0.08,
        reason="cross_appearance_correction",
    )
    with pytest.raises(SchemaError) as error:
        validate_event(incomplete)
    assert "aucun jeu de champs requis n'est complet" in str(error.value)


def test_non_finite_float_is_rejected():
    event = minimal_event("IN")
    event["similarity"] = float("nan")
    with pytest.raises(SchemaError):
        validate_event(event)


def test_frame_index_must_be_an_integer():
    event = minimal_event("IN")
    event["frame_index"] = 1.5
    with pytest.raises(SchemaError):
        validate_event(event)


# ---------------------------------------------------------------------------
# Écriture incrémentale
# ---------------------------------------------------------------------------
def test_events_are_flushed_immediately(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, session_id="s1", flush_every_n_events=1)
    logger.emit("IN", 0.5, 1, person_id=1, technical_track_id=1, direction="in")
    # Le fichier est lisible sur disque avant toute fermeture explicite.
    events = read_events(path)
    assert len(events) == 1
    logger.close()


def test_flush_batching_delays_visibility(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, session_id="s1", flush_every_n_events=5)
    for index in range(3):
        logger.emit("IN", 0.1 * index, index, person_id=1, technical_track_id=1, direction="in")
    assert path.read_text(encoding="utf-8") == ""
    for index in range(3, 5):
        logger.emit("IN", 0.1 * index, index, person_id=1, technical_track_id=1, direction="in")
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 5
    logger.close()
    assert len(read_events(path)) == 5


def test_logger_appends_and_never_overwrites(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")
    with EventLogger(path, session_id="s1") as logger:
        logger.emit("SESSION_START", 0.0, 0, config_path="c", source="s", model_path="m", model_sha256="h")
    with EventLogger(path, session_id="s2") as logger:
        logger.emit("SESSION_END", 1.0, 10, frames_processed=10, duration_s=1.0, fps_mean=10.0)
    events = read_events(path)
    assert [event["session_id"] for event in events] == ["s1", "s2"]
    assert [event["type"] for event in events] == ["SESSION_START", "SESSION_END"]


def test_event_ids_are_unique_and_ordered(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLogger(path, session_id="abc") as logger:
        for index in range(4):
            logger.emit("IN", 0.1 * index, index, person_id=1, technical_track_id=1, direction="in")
    events = read_events(path)
    identifiers = [event["event_id"] for event in events]
    assert identifiers == ["abc-000001", "abc-000002", "abc-000003", "abc-000004"]


def test_event_count_and_close_are_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, session_id="s")
    logger.emit("IN", 0.0, 0, person_id=1, technical_track_id=1, direction="in")
    logger.close()
    logger.close()
    assert logger.event_count == 1


def test_interrupted_session_keeps_written_events(tmp_path):
    """Une interruption brutale ne doit pas perdre les événements produits."""
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, session_id="s", flush_every_n_events=1)
    logger.emit("OUT", 1.0, 5, person_id=1, technical_track_id=1, direction="out")
    # Pas de close() : simulation d'un arrêt brutal.
    del logger
    assert len(read_events(path)) == 1


def test_read_events_validates_each_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"type": "IN"}) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        read_events(path)


def test_read_events_can_filter_by_type(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLogger(path, session_id="s") as logger:
        logger.emit("IN", 0.0, 0, person_id=1, technical_track_id=1, direction="in")
        logger.emit("OUT", 0.1, 1, person_id=1, technical_track_id=1, direction="out")
    assert [event["type"] for event in read_events(path, types=["OUT"])] == ["OUT"]


def test_unsupported_value_is_reported(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, session_id="s")
    with pytest.raises(TypeError):
        logger.emit("IN", 0.0, 0, person_id=object(), technical_track_id=1, direction="in")
    logger.close()


# ---------------------------------------------------------------------------
# ListSink (contrat identique, en mémoire)
# ---------------------------------------------------------------------------
def test_list_sink_obeys_the_same_contract():
    sink = ListSink("memory")
    sink.emit("NEW", 0.0, 1, person_id=1, technical_track_id=1, direction="in")
    assert sink.count("NEW") == 1
    assert sink.of_type("NEW")[0]["session_id"] == "memory"
    with pytest.raises(SchemaError):
        sink.emit("NEW", 0.0, 1, person_id=1)  # direction manquante
