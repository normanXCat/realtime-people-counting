"""Tests du relevé par seconde (`src/second_trace.py`) — outillage de mesure.

Ce module ne compte rien : il **lit** l'état du gestionnaire d'occupation. Les
tests vérifient donc deux choses seulement : que la seconde vidéo est dérivée
de l'index de frame (et non d'une horloge), et que ce qui est écrit reflète
exactement ce que le pipeline expose.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fsm import TrackState  # noqa: E402
from second_trace import (  # noqa: E402
    SecondTraceUnavailable,
    SecondTracer,
    second_of_frame,
    source_fps,
)


class _FakeTrack:
    def __init__(self, person_id, technical_id, state, bbox, seen_frame):
        self.person_id = person_id
        self.technical_track_id = technical_id
        self.state = state
        self.bbox = bbox
        self.inside_occupancy = True
        self.last_seen_frame = seen_frame
        self.confidence = 0.75

    @property
    def is_visible(self):
        return self.state is not TrackState.OCCULTEE


class _FakeOccupancy:
    def __init__(self, tracks):
        self.tracks = tracks
        self.occupancy_operational = 3
        self.occupancy_observed = 2
        self.occupancy_uncertain = 1
        self.visible_count = 1
        self.occluded_count = 1


def test_seconde_derivee_de_l_index_de_frame():
    """La frame 1 ouvre la seconde 0, la frame 31 la seconde 1, à 30 ips."""
    assert second_of_frame(1, 30.0) == 0
    assert second_of_frame(31, 30.0) == 1
    assert second_of_frame(1051, 30.0) == 35
    # Entre deux secondes : rien n'est relevé.
    assert second_of_frame(2, 30.0) is None
    assert second_of_frame(30, 30.0) is None


def test_seconde_avec_un_fps_non_entier():
    """Un FPS fractionnaire ne doit ni dupliquer ni sauter une seconde."""
    fps = 29.97
    secondes = [second_of_frame(i, fps) for i in range(1, 400)]
    releves = [s for s in secondes if s is not None]
    assert releves == sorted(set(releves))  # strictement croissant, sans doublon
    assert releves[:4] == [0, 1, 2, 3]


def test_source_camera_refusee():
    """Sur caméra, il n'existe pas de seconde vidéo : le relevé est refusé."""
    with pytest.raises(SecondTraceUnavailable):
        source_fps(0)


def test_trace_ecrit_une_ligne_et_une_image_par_seconde(tmp_path):
    tracks = {
        7: _FakeTrack(7, 3, TrackState.PRESENTE, np.array([10.0, 20.0, 60.0, 180.0]), 31),
        9: _FakeTrack(9, -1, TrackState.OCCULTEE, np.array([70.0, 20.0, 120.0, 180.0]), 12),
    }
    occupancy = _FakeOccupancy(tracks)
    tracer = SecondTracer(tmp_path, fps=30.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    assert tracer.observe(31, frame, occupancy, [3, 5]) == 1
    assert tracer.observe(32, frame, occupancy, [3, 5]) is None  # hors seconde
    tracer.close()

    lignes = (tmp_path / "trace_per_second.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lignes) == 1
    row = json.loads(lignes[0])
    assert row["second"] == 1 and row["frame_index"] == 31
    assert row["pistes_vues"] == 2 and row["technical_ids_in_frame"] == [3, 5]
    assert row["visible_count"] == 1 and row["occupancy_operational"] == 3
    par_id = {p["person_id"]: p for p in row["persons"]}
    # La piste vue à cette frame est distinguée de celle dont la boîte est figée :
    # sans cette distinction, une boîte périmée passerait pour une observation.
    assert par_id[7]["seen_this_frame"] is True
    assert par_id[9]["seen_this_frame"] is False
    assert par_id[9]["state"] == "OCCULTEE"
    assert par_id[7]["bbox"] == [10.0, 20.0, 60.0, 180.0]
    assert (tmp_path / "trace_frames" / "s01.jpg").exists()


def test_trace_ne_modifie_pas_l_etat_du_pipeline(tmp_path):
    """Garde-fou de périmètre : l'outillage lit, il ne touche à rien."""
    track = _FakeTrack(1, 1, TrackState.PRESENTE, np.array([0.0, 0.0, 10.0, 20.0]), 1)
    occupancy = _FakeOccupancy({1: track})
    avant = (track.state, track.inside_occupancy, track.technical_track_id)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    original = frame.copy()

    tracer = SecondTracer(tmp_path, fps=30.0)
    tracer.observe(1, frame, occupancy, [1])
    tracer.close()

    assert (track.state, track.inside_occupancy, track.technical_track_id) == avant
    # L'annotation se fait sur une copie : la frame du pipeline est intacte.
    assert np.array_equal(frame, original)
