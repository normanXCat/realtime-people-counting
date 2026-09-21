"""Diagnostics de perte de piste : classification, débit, et câblage dans `main`.

Deux niveaux sont testés :

1. le module : classification d'une frame (aucune détection / boîtes sans
   identifiant technique / détections faibles), absence d'invention
   d'identifiant, limitation de débit et compteurs publiés ;
2. `main` : une frame dont les boîtes n'ont **pas** d'``id`` ne doit produire
   aucun ``IN``/``OUT``/``NEW`` — seulement ``TRACK_ID_ABSENT``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import main as entry_point  # noqa: E402
from calibration import ValidatedLine  # noqa: E402
from config import DiagnosticsConfig, load_config  # noqa: E402
from events import ListSink, read_events  # noqa: E402
from track_diagnostics import (  # noqa: E402
    BOXES_EMPTY,
    BOXES_MISSING,
    BOXES_PRESENT,
    SITUATION_DETECTION_ABSENT,
    SITUATION_LOW_CONFIDENCE,
    SITUATION_NOMINAL,
    SITUATION_TRACK_ID_ABSENT,
    DetectionDiagnostics,
    DetectionSnapshot,
    box_count,
)

FRAME_SIDE = 600


# ---------------------------------------------------------------------------
# Niveau 1 : le module
# ---------------------------------------------------------------------------
class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def int(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values

    def tolist(self):
        return [int(value) for value in self._values]


class _FakeBoxes:
    """Double minimal d'un jeu de boîtes Ultralytics (sans ``__len__``)."""

    def __init__(self, detections=(), with_ids=True):
        self.id = _FakeTensor([item[0] for item in detections]) if with_ids else None
        self.xyxy = _FakeTensor([item[1] for item in detections])
        self.conf = _FakeTensor([item[2] for item in detections])


def test_box_count_handles_missing_and_partial_objects():
    assert box_count(None) == 0
    boxes = _FakeBoxes([(1, [0, 0, 10, 10], 0.9)])
    assert box_count(boxes) == 1
    assert box_count(_FakeBoxes(with_ids=False)) == 0


def test_classify_distinguishes_every_situation():
    diagnostics = DetectionDiagnostics(track_high_thresh=0.4)
    assert diagnostics.classify(
        DetectionSnapshot(BOXES_MISSING, 0, False)
    ) == SITUATION_DETECTION_ABSENT
    assert diagnostics.classify(
        DetectionSnapshot(BOXES_EMPTY, 0, False)
    ) == SITUATION_DETECTION_ABSENT
    assert diagnostics.classify(
        DetectionSnapshot(BOXES_PRESENT, 2, False)
    ) == SITUATION_TRACK_ID_ABSENT
    assert diagnostics.classify(
        DetectionSnapshot(BOXES_PRESENT, 2, True, (0.12, 0.3))
    ) == SITUATION_LOW_CONFIDENCE
    assert diagnostics.classify(
        DetectionSnapshot(BOXES_PRESENT, 2, True, (0.9, 0.3))
    ) == SITUATION_NOMINAL


def test_track_id_absent_event_is_explicit_and_bounded():
    sink = ListSink("diag")
    diagnostics = DetectionDiagnostics(
        DiagnosticsConfig(enabled=True, min_interval_seconds=1.0),
        event_sink=sink,
        track_high_thresh=0.4,
    )
    snapshot = DetectionSnapshot(BOXES_PRESENT, 1, False)
    diagnostics.observe(snapshot, 0.0, 1)
    diagnostics.observe(snapshot, 0.2, 2)   # même épisode : pas de seconde ligne
    diagnostics.observe(snapshot, 0.4, 3)
    assert sink.count("TRACK_ID_ABSENT") == 1
    assert sink.events[0]["reason"] == "boxes_present_without_tracker_ids"
    assert sink.events[0]["detections"] == 1
    assert sink.events[0]["frame_index"] == 1

    diagnostics.observe(snapshot, 1.5, 4)   # au-delà de l'intervalle : rappel
    assert sink.count("TRACK_ID_ABSENT") == 2
    assert sink.events[-1]["consecutive_frames"] == 4


def test_leading_edge_is_always_reported():
    sink = ListSink("diag")
    diagnostics = DetectionDiagnostics(
        DiagnosticsConfig(min_interval_seconds=10.0), event_sink=sink
    )
    diagnostics.observe(DetectionSnapshot(BOXES_MISSING, 0, False), 0.0, 1)
    diagnostics.observe(DetectionSnapshot(BOXES_PRESENT, 1, True, (0.9,)), 0.01, 2)
    diagnostics.observe(DetectionSnapshot(BOXES_MISSING, 0, False), 0.02, 3)
    assert [event["type"] for event in sink.events] == [
        "DETECTION_ABSENT",
        "DETECTION_ABSENT",
    ]
    assert sink.events[0]["reason"] == "boxes_missing"
    assert sink.events[1]["reason"] == "boxes_missing"


def test_empty_box_set_is_reported_as_no_boxes_at_all():
    sink = ListSink("diag")
    diagnostics = DetectionDiagnostics(event_sink=sink)
    diagnostics.observe(DetectionSnapshot(BOXES_EMPTY, 0, False), 0.0, 1)
    assert sink.events[0]["type"] == "DETECTION_ABSENT"
    assert sink.events[0]["reason"] == "no_boxes_in_frame"
    assert sink.events[0]["detections"] == 0


def test_low_confidence_event_carries_the_threshold():
    sink = ListSink("diag")
    diagnostics = DetectionDiagnostics(
        DiagnosticsConfig(min_interval_seconds=1.0), event_sink=sink,
        track_high_thresh=0.4,
    )
    diagnostics.observe(
        DetectionSnapshot(BOXES_PRESENT, 2, True, (0.12, 0.35)), 0.0, 1
    )
    event = sink.of_type("DETECTION_LOW_CONFIDENCE")[0]
    assert event["reason"] == "only_low_confidence_detections"
    assert event["max_confidence"] == pytest.approx(0.35)
    assert event["track_high_thresh"] == pytest.approx(0.4)


def test_counters_are_published_even_when_events_are_disabled():
    diagnostics = DetectionDiagnostics(
        DiagnosticsConfig(enabled=False), event_sink=ListSink("diag")
    )
    snapshot = DetectionSnapshot(BOXES_MISSING, 0, False)
    for index in range(5):
        diagnostics.observe(snapshot, float(index), index + 1)
    summary = diagnostics.summary()
    assert summary["enabled"] is False
    assert summary["frames_observed"] == 5
    assert summary["frames_by_situation"][SITUATION_DETECTION_ABSENT] == 5


def test_summary_counts_frames_without_tracker_ids():
    diagnostics = DetectionDiagnostics(event_sink=ListSink("diag"))
    diagnostics.observe(DetectionSnapshot(BOXES_PRESENT, 3, False), 0.0, 1)
    diagnostics.observe(DetectionSnapshot(BOXES_PRESENT, 3, False), 0.5, 2)
    diagnostics.observe(DetectionSnapshot(BOXES_PRESENT, 3, True, (0.9,)), 1.0, 3)
    summary = diagnostics.summary()
    assert summary["frames_without_tracker_ids"] == 2
    assert summary["frames_by_situation"][SITUATION_NOMINAL] == 1


# ---------------------------------------------------------------------------
# Niveau 3 : câblage dans `main`
# ---------------------------------------------------------------------------
def validated_line() -> ValidatedLine:
    return ValidatedLine(
        p1=(0.2, 0.4),
        p2=(0.8, 0.4),
        inside_side="negative",
        frame_width=FRAME_SIDE,
        frame_height=FRAME_SIDE,
        display_scale=1.0,
    )


class _FakeResult:
    def __init__(self, frame, boxes):
        self.orig_img = frame.copy()
        self.speed = {"inference": 1.0, "postprocess": 0.5}
        self.boxes = boxes


def install_fake_ultralytics(monkeypatch, boxes_factory, frames: int = 8):
    module = types.ModuleType("ultralytics")
    image = np.zeros((FRAME_SIDE, FRAME_SIDE, 3), dtype=np.uint8)

    class _FakeModel:
        def __init__(self, path):
            self.path = path

        def track(self, **_kwargs):
            for _ in range(frames):
                yield _FakeResult(image, boxes_factory())

    module.YOLO = _FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def run_main(tmp_path: Path, monkeypatch, session_id: str, boxes_factory) -> Path:
    monkeypatch.setattr(
        entry_point, "perform_line_selection", lambda config, source, **kwargs: validated_line()
    )
    install_fake_ultralytics(monkeypatch, boxes_factory)
    code = entry_point.main(
        [
            "--source", "clip.mp4",
            "--output-root", str(tmp_path),
            "--session-id", session_id,
            "--no-show",
        ]
    )
    assert code == 0
    return tmp_path / session_id


def test_boxes_without_tracker_ids_produce_no_counting(tmp_path, monkeypatch):
    """Le cas central : des boîtes existent, aucun ``id`` n'est fourni."""
    root = run_main(
        tmp_path,
        monkeypatch,
        "session-sans-id",
        lambda: _FakeBoxes([(0, [250, 350, 350, 550], 0.8)], with_ids=False),
    )
    events = read_events(root / "events.jsonl")
    types = [event["type"] for event in events]
    assert "TRACK_ID_ABSENT" in types
    assert [event_type for event_type in types if event_type in ("IN", "OUT", "NEW")] == []
    absent = [event for event in events if event["type"] == "TRACK_ID_ABSENT"]
    assert absent[0]["reason"] == "boxes_present_without_tracker_ids"
    assert absent[0]["detections"] == 1
    # Aucun identifiant technique n'est inventé : aucun événement d'identité.
    assert not [event for event in events if event["type"].startswith("REID_")]

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostics"]["frames_without_tracker_ids"] == 8
    assert summary["counters"]["occupancy_operational"] == 0
    assert summary["counters"]["total_in"] == 0
    assert summary["counters"]["total_new"] == 0


def test_low_confidence_boxes_are_reported_without_being_dropped(tmp_path, monkeypatch):
    """Les détections faibles restent utilisées : elles sont seulement tracées."""
    root = run_main(
        tmp_path,
        monkeypatch,
        "session-faible",
        lambda: _FakeBoxes([(1, [250, 350, 350, 550], 0.12)]),
    )
    events = read_events(root / "events.jsonl")
    low = [event for event in events if event["type"] == "DETECTION_LOW_CONFIDENCE"]
    assert low, "une détection faible doit être journalisée"
    assert low[0]["max_confidence"] == pytest.approx(0.12, abs=1e-6)


def test_absent_boxes_are_reported_as_such(tmp_path, monkeypatch):
    root = run_main(tmp_path, monkeypatch, "session-vide", lambda: None)
    events = read_events(root / "events.jsonl")
    absent = [event for event in events if event["type"] == "DETECTION_ABSENT"]
    assert absent
    assert absent[0]["reason"] == "boxes_missing"


def test_default_session_reports_no_configuration_warning(tmp_path, monkeypatch):
    root = run_main(tmp_path, monkeypatch, "session-coherente", lambda: None)
    events = read_events(root / "events.jsonl")
    # Lot 3.a : l'avertissement de repli du descripteur profond dépend de
    # l'environnement (poids ONNX non versionné), pas de la cohérence de la
    # configuration que ce test vérifie ; il est éprouvé dans
    # tests/test_reid_deep_lot3a.py.
    assert not [
        event for event in events
        if event["type"] == "CONFIG_WARNING"
        and event.get("code") != "reid_deep_descriptor_unavailable"
    ]
    start = events[0]
    assert start["tracker_low_thresh"] == pytest.approx(
        load_config().tracker.track_low_thresh
    )
