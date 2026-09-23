"""Tests garantissant la non-régression quand head_assist est **explicitement**

désactivé (ablation), et l'activation par défaut dans la configuration livrée."""

from __future__ import annotations

import numpy as np
import pytest

from config import load_config, override_config
from events import ListSink
from geometry import VirtualLine
from identity_manager import IdentityManager, Observation
from occupancy_manager import Detection, OccupancyManager
from occupancy_types import TrackState


def test_default_config_head_assist_is_enabled():
    config = load_config()
    assert config.presence.head_assist.enabled is True
    assert config.model.path == "models/yolo11n.pt"


def test_observations_head_fields_default_to_none():
    obs = Observation(
        technical_track_id=1,
        bbox=np.array([10.0, 20.0, 50.0, 100.0]),
        confidence=0.9,
        anchor=(30.0, 100.0),
        bbox_height=80.0,
    )
    assert obs.head_point is None
    assert obs.head_confidence is None


def test_identity_manager_assign_maintains_none_head_fields():
    sink = ListSink("test_session")
    im = IdentityManager(event_sink=sink)

    obs = Observation(
        technical_track_id=1,
        bbox=np.array([10.0, 20.0, 50.0, 100.0]),
        confidence=0.9,
        anchor=(30.0, 100.0),
        bbox_height=80.0,
    )
    assignments = im.assign([obs], frame=None, timestamp_s=1.0, frame_index=1)
    assert len(assignments) == 1
    assert assignments[0].head_point is None
    assert assignments[0].head_confidence is None


def test_disabled_head_assist_never_emits_presence_maintained_by_head(test_config, sink):
    """Quand head_assist est désactivé, aucun événement PRESENCE_MAINTAINED_BY_HEAD n'est émis."""
    config = override_config(test_config, {"presence": {"head_assist": {"enabled": False}}})
    assert config.presence.head_assist.enabled is False

    line = VirtualLine(p1=(0.0, 0.5), p2=(1.0, 0.5), inside_side="negative")
    manager = OccupancyManager(config, event_sink=sink, line=line)
    manager.set_frame_budget(config.timing.frame_budget(10.0))

    frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # Initialiser une personne à l'intérieur (y >= 330, negative = sous la ligne)
    box_inside = np.array([260.0, 250.0, 340.0, 450.0])  # hauteur 200, ancre y=450
    for f in range(1, 10):
        t = f * 0.1
        manager.process_frame([Detection(1, box_inside, 0.9)], frame, t, f)

    track = manager.tracks.get(1)
    assert track is not None
    assert track.state is TrackState.PRESENTE

    # Chute de hauteur soudaine (hauteur 100 au lieu de 200)
    drop_box = np.array([260.0, 350.0, 340.0, 450.0])
    for f in range(10, 15):
        t = f * 0.1
        manager.process_frame([Detection(1, drop_box, 0.9)], frame, t, f)

    # Vérifier qu'aucun événement d'assistance tête n'a été produit
    types = [e["type"] for e in sink.events]
    assert "PRESENCE_MAINTAINED_BY_HEAD" not in types
    assert "PRESENCE_EXTENSION_EXPIRED" not in types
