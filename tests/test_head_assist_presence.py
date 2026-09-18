"""Tests fonctionnels de l'assistance de présence par détection de tête."""

from __future__ import annotations

import numpy as np
import pytest

from geometry import VirtualLine
from occupancy_manager import Detection, OccupancyManager
from occupancy_types import TrackState


def _make_head_assist_manager(make_config, sink, max_extension=2.0, min_conf=0.5):
    cfg = make_config({
        "line": {"inside_side": "negative"},
        "timing": {
            "warmup_seconds": 0.3,
            "warmup_min_frames": 4,
            "confirmation_seconds": 0.2,
            "grace_period_seconds": 1.0,
            "fps_estimate_window": 4,
        },
        "presence": {
            "head_assist": {
                "enabled": True,
                "min_keypoint_confidence": min_conf,
                "max_presence_extension_seconds": max_extension,
                "keypoints_used": ["nose", "left_eye", "right_eye"],
            }
        },
    })
    line = VirtualLine(p1=(0.0, 0.5), p2=(1.0, 0.5), inside_side="negative")
    mgr = OccupancyManager(cfg, event_sink=sink, line=line)
    mgr.set_frame_budget(cfg.timing.frame_budget(10.0))
    return mgr, cfg


def test_head_assist_maintains_presence_on_height_drop(make_config, sink):
    """1. Chute de hauteur avec tête confiante -> reste PRESENTE, émet PRESENCE_MAINTAINED_BY_HEAD."""
    manager, _ = _make_head_assist_manager(make_config, sink, max_extension=5.0)
    frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # Initialisation de la personne à l'intérieur (hauteur 200, ancre y=450)
    box_inside = np.array([260.0, 250.0, 340.0, 450.0])
    for f in range(1, 10):
        t = f * 0.1
        manager.process_frame([Detection(1, box_inside, 0.9)], frame, t, f)

    track = manager.tracks[1]
    assert track.state is TrackState.PRESENTE
    occ_before = manager.occupancy_confirmed

    # Chute de hauteur (h=100 au lieu de 200), mais tête confiante (conf=0.8)
    drop_box = np.array([260.0, 350.0, 340.0, 450.0])
    head_pt = (300.0, 360.0)
    for f in range(10, 15):
        t = f * 0.1
        manager.process_frame(
            [Detection(1, drop_box, 0.9, head_point=head_pt, head_confidence=0.8)],
            frame,
            t,
            f,
        )

    # La piste reste en PRESENTE
    assert track.state is TrackState.PRESENTE
    assert manager.occupancy_confirmed == occ_before

    # Événement PRESENCE_MAINTAINED_BY_HEAD émis
    events = [e for e in sink.events if e["type"] == "PRESENCE_MAINTAINED_BY_HEAD"]
    assert len(events) >= 1
    assert events[0]["person_id"] == 1
    assert events[0]["head_confidence"] == 0.8
    assert events[0]["feet_anchor_status"] == "height_drop"


def test_head_assist_transitions_to_occluded_when_head_not_confident(make_config, sink):
    """2. Chute de hauteur avec tête non confiante (< min_conf) -> bascule vers OCCULTEE."""
    manager, _ = _make_head_assist_manager(make_config, sink, min_conf=0.6)
    frame = np.zeros((600, 600, 3), dtype=np.uint8)

    box_inside = np.array([260.0, 250.0, 340.0, 450.0])
    for f in range(1, 10):
        t = f * 0.1
        manager.process_frame([Detection(1, box_inside, 0.9)], frame, t, f)

    track = manager.tracks[1]
    assert track.state is TrackState.PRESENTE

    # Chute de hauteur, tête sous le seuil (conf=0.3 < 0.6)
    drop_box = np.array([260.0, 350.0, 340.0, 450.0])
    head_pt = (300.0, 360.0)
    manager.process_frame(
        [Detection(1, drop_box, 0.9, head_point=head_pt, head_confidence=0.3)],
        frame,
        1.0,
        10,
    )

    # Bascule vers OCCULTEE
    assert track.state is TrackState.OCCULTEE
    types = [e["type"] for e in sink.events]
    assert "OCCLUDED" in types
    assert "PRESENCE_MAINTAINED_BY_HEAD" not in types


def test_head_assist_extension_expires_after_max_seconds(make_config, sink):
    """3. Au-delà de max_presence_extension_seconds -> PRESENCE_EXTENSION_EXPIRED émis, bascule vers OCCULTEE."""
    max_extension = 0.5  # 5 frames à 10 ips
    manager, _ = _make_head_assist_manager(make_config, sink, max_extension=max_extension)
    frame = np.zeros((600, 600, 3), dtype=np.uint8)

    box_inside = np.array([260.0, 250.0, 340.0, 450.0])
    for f in range(1, 10):
        t = f * 0.1
        manager.process_frame([Detection(1, box_inside, 0.9)], frame, t, f)

    track = manager.tracks[1]
    assert track.state is TrackState.PRESENTE

    # Chute de hauteur continue pendant 1.0 s (> 0.5 s) avec tête confiante
    drop_box = np.array([260.0, 350.0, 340.0, 450.0])
    head_pt = (300.0, 360.0)
    for f in range(10, 25):
        t = f * 0.1
        manager.process_frame(
            [Detection(1, drop_box, 0.9, head_point=head_pt, head_confidence=0.85)],
            frame,
            t,
            f,
        )

    # La durée a expiré, bascule vers OCCULTEE
    assert track.state is TrackState.OCCULTEE
    expired_events = [e for e in sink.events if e["type"] == "PRESENCE_EXTENSION_EXPIRED"]
    assert len(expired_events) >= 1
    assert expired_events[0]["person_id"] == 1
    assert expired_events[0]["max_extension_seconds"] == pytest.approx(max_extension)


def test_head_point_crossing_line_never_triggers_crossing(make_config, sink):
    """4. Garde-fou strict : head_point qui traverse la ligne ne déclenche JAMAIS de comptage IN/OUT."""
    manager, _ = _make_head_assist_manager(make_config, sink, max_extension=10.0)
    frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # Personne à l'intérieur : pieds immobiles à y=450 (intérieur)
    box_inside = np.array([260.0, 250.0, 340.0, 450.0])
    for f in range(1, 10):
        t = f * 0.1
        manager.process_frame([Detection(1, box_inside, 0.9)], frame, t, f)

    assert manager.tracks[1].state is TrackState.PRESENTE
    in_before = manager.total_in
    out_before = manager.total_out

    # Simuler le head_point qui se déplace de y=350 vers y=150 (traverse la ligne y=300 vers l'extérieur)
    # pendant que l'ancre pieds reste IMMOBILE à y=450 (intérieur)
    for f in range(10, 20):
        t = f * 0.1
        simulated_head_y = 350.0 - (f - 10) * 20.0  # de 350 à 150 (franchit la ligne)
        manager.process_frame(
            [Detection(1, box_inside, 0.9, head_point=(300.0, simulated_head_y), head_confidence=0.95)],
            frame,
            t,
            f,
        )

    # Aucun franchissement n'a été déclenché
    assert manager.total_in == in_before
    assert manager.total_out == out_before
    crossing_events = [e for e in sink.events if e["type"] in ("IN", "OUT")]
    assert len(crossing_events) == 0
