"""Niveau 2 — Warm-up : frontière explicite, deux bornes, effectif initial.

Le warm-up sert à mesurer l'effectif **déjà présent** avant de commencer à
compter. Trois exigences sont vérifiées ici :

- il se termine sur **deux** bornes (frames observées **et** durée écoulée) :
  une seule des deux terminerait trop tôt, à faible FPS (durée écoulée, trois
  frames vues) comme à FPS élevé (15 frames vues en 0,15 s) ;
- la frontière n'est pas ambiguë : la frame qui atteint les deux bornes
  appartient au warm-up, la suivante est la première frame comptée ;
- pendant le warm-up, une personne stable à l'intérieur entre dans l'effectif
  initial ; dehors, dans la zone morte, instable ou en franchissement, elle n'y
  entre pas — et personne n'est compté deux fois.

Repère géométrique : image 600×600, ligne à y = 300, ``inside_side: negative``
(l'intérieur est sous la ligne). Zone morte = 0,15 × 200 = 30 px.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TEST_FPS, make_test_line, person_box
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

LINE_Y = 300.0
OUTSIDE_Y = 250.0
INSIDE_Y = 450.0
TINY_FRAME = np.zeros((600, 600, 3), dtype=np.uint8)


class Sim:
    """Rejoue une séquence de frames à pas fixe contre un OccupancyManager."""

    def __init__(self, manager, frame, dt=1.0 / TEST_FPS):
        self.manager = manager
        self.frame = frame
        self.dt = dt
        self.time = 0.0
        self.index = 0

    def step(self, detections=()):
        self.index += 1
        self.time += self.dt
        self.manager.process_frame(list(detections), self.frame, self.time, self.index)
        return self

    def steps(self, count, detections=()):
        for _ in range(count):
            self.step(detections)
        return self


def detection(track_id: int, y: float, x: float = 300.0, height: float = 200.0) -> Detection:
    return Detection(
        technical_track_id=track_id,
        bbox=person_box(x, y, height=height),
        confidence=0.9,
    )


def build_manager(config, sink, appearance=None) -> OccupancyManager:
    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=appearance,
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=make_test_line(inside_side=config.line.inside_side),
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


@pytest.fixture
def manager(test_config, sink, frame):
    return build_manager(test_config, sink)


def warmup_end(sink):
    events = sink.of_type("WARMUP_END")
    assert len(events) == 1, f"WARMUP_END attendu exactement une fois : {events}"
    return events[0]


# ---------------------------------------------------------------------------
# Frontière : la frame qui atteint les deux bornes appartient au warm-up
# ---------------------------------------------------------------------------
def test_the_completing_frame_belongs_to_the_warmup(manager, frame, sink):
    """Bornes du fixture : 4 frames observées ET 0,3 s → warm-up = frames 1..4."""
    sim = Sim(manager, frame)
    sim.steps(4, [detection(1, INSIDE_Y)])
    assert manager.phase == "WARMUP", "la 4e frame appartient encore au warm-up"
    assert manager.initial_occupancy == 0
    assert sink.count("WARMUP_END") == 0

    sim.step([detection(1, INSIDE_Y)])  # 5e frame : première frame comptée
    assert manager.phase == "RUNNING"
    assert manager.initial_occupancy == 1
    event = warmup_end(sink)
    assert event["warmup_frames_observed"] == 4
    assert event["warmup_min_frames"] == 4
    assert event["warmup_elapsed_seconds"] == pytest.approx(0.3, abs=1e-6)
    assert event["initial_occupancy"] == 1


def test_warmup_end_reports_the_real_initial_occupancy(manager, frame, sink):
    """``initial_occupancy`` n'est jamais publié à zéro quand une personne est là.

    L'événement est émis après la frame qui applique l'effectif initial
    (``EMPLACE_INITIAL``) : sinon il publierait systématiquement 0 et ne
    servirait à rien pour l'audit.
    """
    Sim(manager, frame).steps(6, [detection(1, INSIDE_Y)])
    event = warmup_end(sink)
    assert event["initial_occupancy"] == 1
    assert event["initial_occupancy"] == manager.initial_occupancy
    assert event["warmup_elapsed_seconds"] >= 0.3


@pytest.mark.parametrize("fps", [2.0, 5.0, 60.0])
def test_warmup_requires_both_bounds_for_any_fps(sink, make_config, fps):
    """Aucune des deux bornes ne peut terminer le warm-up seule (spec 1)."""
    config = make_config({
        "timing": {"warmup_seconds": 0.5, "warmup_min_frames": 15},
    })
    manager = build_manager(config, sink)
    manager.set_frame_budget(config.timing.frame_budget(fps))
    sim = Sim(manager, np.zeros((600, 600, 3), dtype=np.uint8), dt=1.0 / fps)

    completed_at = None
    for _ in range(200):
        sim.step([detection(1, INSIDE_Y)])
        if manager.phase == "RUNNING":
            completed_at = sim.index
            break
    assert completed_at is not None, "le warm-up doit finir"
    event = warmup_end(sink)
    # Borne 1 : le nombre de frames observées ne peut pas être inférieur au minimum.
    assert event["warmup_frames_observed"] >= 15
    # Borne 2 : la durée écoulée ne peut pas être inférieure à warmup_seconds.
    assert event["warmup_elapsed_seconds"] >= 0.5 - 1e-9
    # Le warm-up ne peut pas durer moins que la borne dure en frames.
    assert completed_at >= 16


def test_low_fps_does_not_end_the_warmup_on_elapsed_time_alone(sink, make_config):
    """À 2 fps, 1 s s'écoule dès la 3e frame : le warm-up ne doit pas finir là."""
    config = make_config({"timing": {"warmup_seconds": 0.5, "warmup_min_frames": 15}})
    manager = build_manager(config, sink)
    manager.set_frame_budget(config.timing.frame_budget(2.0))
    sim = Sim(manager, TINY_FRAME, dt=0.5)
    sim.steps(4, [detection(1, INSIDE_Y)])
    assert manager.phase == "WARMUP"
    assert manager.initial_occupancy == 0
    assert sink.count("WARMUP_END") == 0
    sim.steps(12, [detection(1, INSIDE_Y)])  # 16e frame
    assert manager.phase == "RUNNING"
    assert warmup_end(sink)["warmup_frames_observed"] == 15


def test_high_fps_does_not_end_the_warmup_on_frame_count_alone(sink, make_config):
    """À 60 fps, 15 frames valent 0,25 s : le warm-up doit continuer."""
    config = make_config({"timing": {"warmup_seconds": 0.5, "warmup_min_frames": 15}})
    manager = build_manager(config, sink)
    manager.set_frame_budget(config.timing.frame_budget(60.0))
    sim = Sim(manager, TINY_FRAME, dt=1.0 / 60.0)
    sim.steps(20, [detection(1, INSIDE_Y)])
    assert manager.phase == "WARMUP", "20 frames à 60 fps ne font que 0,33 s"
    assert sink.count("WARMUP_END") == 0
    sim.steps(15, [detection(1, INSIDE_Y)])
    assert manager.phase == "RUNNING"
    event = warmup_end(sink)
    assert event["warmup_frames_observed"] >= 15
    assert event["warmup_elapsed_seconds"] >= 0.5 - 1e-9


# ---------------------------------------------------------------------------
# Qui entre dans l'effectif initial ?
# ---------------------------------------------------------------------------
def test_person_outside_during_warmup_is_not_counted(manager, frame, sink):
    sim = Sim(manager, frame).steps(6, [detection(1, OUTSIDE_Y)])
    assert manager.initial_occupancy == 0
    assert manager.occupancy_confirmed == 0
    assert manager.tracks[1].state.name == "EXTERIEUR"
    assert sink.count("IN") == 0 and sink.count("NEW") == 0 and sink.count("OUT") == 0


def test_person_in_the_dead_zone_during_warmup_is_not_counted(manager, frame, sink):
    sim = Sim(manager, frame).steps(8, [detection(1, LINE_Y)])
    assert manager.initial_occupancy == 0
    assert manager.occupancy_confirmed == 0
    assert manager.tracks[1].last_zone == "MORTE"
    assert sink.count("IN") == 0 and sink.count("NEW") == 0
    assert warmup_end(sink)["initial_occupancy"] == 0


def test_unstable_track_does_not_contaminate_the_initial_occupancy(manager, frame, sink):
    """Une piste vue une seule frame (puis perdue) reste hors de l'effectif."""
    sim = Sim(manager, frame)
    sim.step([detection(1, INSIDE_Y)])
    sim.steps(8)
    assert manager.initial_occupancy == 0
    assert manager.occupancy_confirmed == 0


def test_crossing_during_warmup_never_produces_in_or_out(manager, frame, sink):
    """Un franchissement pendant le warm-up ne compte pas — et ne double pas."""
    sim = Sim(manager, frame)
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(2, [detection(1, INSIDE_Y)])   # franchissement pendant le warm-up
    assert sink.count("IN") == 0 and sink.count("OUT") == 0
    assert manager.initial_occupancy == 0

    sim.steps(4, [detection(1, INSIDE_Y)])   # après le warm-up
    assert manager.total_in == 1, "la personne est comptée une fois et une seule"
    assert manager.total_out == 0
    assert manager.total_new == 0
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_operational == 1
    # L'entrée n'a pas été capturée pendant le warm-up : elle est signalée.
    kinds = [event["kind"] for event in sink.of_type("INCONSISTENT_STATE")]
    assert kinds.count("entry_side_change_without_crossing") == 1


def test_person_hidden_during_warmup_then_visible_inside_is_counted_once(
    manager, frame, sink
):
    sim = Sim(manager, frame)
    sim.steps(2, [detection(1, INSIDE_Y)])
    sim.steps(3)                              # cachée pendant le warm-up
    assert manager.initial_occupancy == 0
    sim.steps(6, [detection(1, INSIDE_Y)])    # visible à l'intérieur après
    assert manager.total_in == 0
    assert manager.total_new == 1, "comptée une fois, comme une apparition intérieure"
    assert manager.occupancy_confirmed == 1


def test_person_appearing_inside_after_warmup_is_a_confirmed_new(manager, frame, sink):
    sim = Sim(manager, frame)
    sim.steps(5)                              # warm-up sur scène vide
    assert manager.initial_occupancy == 0
    sim.steps(6, [detection(1, INSIDE_Y)])
    assert manager.total_new == 1
    assert manager.total_in == 0
    events = sink.of_type("NEW")
    assert events[0]["stable_frames"] >= 2
    assert events[0]["origin"] == "APPARITION_INTERIEURE"


def test_warmup_event_is_published_exactly_once(manager, frame, sink):
    Sim(manager, frame).steps(30, [detection(1, INSIDE_Y)])
    assert sink.count("WARMUP_END") == 1
    assert manager.initial_occupancy == 1
    assert manager.occupancy_operational == 1
