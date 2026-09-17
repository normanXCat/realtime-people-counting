"""Tests ciblés : détection personne assise/floue + stabilité de l'ancre lors d'une chute de hauteur.

Couvre les deux corrections demandées :
1. Détection personne assise et légèrement floue :
   - Personne assise nette avec ratio W/H élevé (~1.8) acceptée et suivie.
   - Personne assise légèrement floue (confiance modérée >= 0.10) transmise et suivie.
   - Filtre géométrique : accepte ratios plausibles (jusqu'à 2.5), rejette les formes aberrantes
     avec émission de DETECTION_REJECTED_GEOMETRY.
   - Test comparatif avant/après (filtre strict vs configuration assouplie).
   - Diagnostics systématiques YOLO bruts avant filtrage (observe_yolo_detections).

2. Stabilité de l'ancre et chute brutale de hauteur de boîte :
   - Personne debout hauteur stable -> ancre fiable.
   - Chute brutale de hauteur (jambes masquées) -> anchor_reliable=False,
     émission de ANCHOR_UNRELIABLE (reason="height_drop"), pas de franchissement faux positif.
   - Diminution progressive (éloignement) -> pas de fausse détection de chute, ancre fiable.
   - Personne qui s'assoit et reste assise -> suspension temporaire puis réadaptation
     et réactivation de la fiabilité après confirmation.
   - Franchissement au moment où les jambes sont masquées -> décision suspendue
     puis résolue sans perte ni double comptage.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    TEST_FPS,
    StubAppearance,
    identity_vectors_by_x,
    make_test_line,
    person_box,
)
from geometry import (
    check_anchor_height_drop,
    check_detection_geometry,
)
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager
from track_diagnostics import DetectionDiagnostics

INSIDE_Y = 450.0
OUTSIDE_Y = 250.0
VECTORS = {
    150.0: (1.0, 0.0, 0.0),
    300.0: (0.0, 1.0, 0.0),
    450.0: (0.0, 0.0, 1.0),
}


class Sim:
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

    def warmup(self):
        return self.steps(5)


def sitting_person_box(x_center: float, y_bottom: float, height: float = 80.0, width: float = 144.0):
    """Bounding box d'une personne assise : ratio W/H = 144 / 80 = 1.8."""
    return np.array(
        [
            x_center - width / 2.0,
            y_bottom - height,
            x_center + width / 2.0,
            y_bottom,
        ],
        dtype=float,
    )


def build_manager(config, sink, appearance=None):
    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=appearance or StubAppearance(describe_fn=identity_vectors_by_x(VECTORS)),
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
def sim(test_config, sink, frame):
    return Sim(build_manager(test_config, sink), frame)


# ===========================================================================
# 1. Tests Détection personne assise / floue
# ===========================================================================

def test_sitting_person_sharp_good_confidence(sim, sink):
    """1. Personne assise nette avec bonne confiance : W/H ~ 1.8 acceptée et suivie."""
    manager = sim.manager
    sim.warmup()

    # Personne assise à l'extérieur puis traversant vers l'intérieur
    box_out = sitting_person_box(150.0, OUTSIDE_Y, height=80.0, width=144.0)
    box_in = sitting_person_box(150.0, INSIDE_Y, height=80.0, width=144.0)

    sim.steps(2, [Detection(1, box_out, confidence=0.92)])
    sim.steps(2, [Detection(1, box_in, confidence=0.92)])

    assert manager.total_in == 1
    assert manager.occupancy_confirmed == 1
    rejected = sink.of_type("DETECTION_REJECTED_GEOMETRY")
    assert len(rejected) == 0, "Une personne assise nette ne doit pas être rejetée"


def test_sitting_person_blurry_moderate_confidence(sim, sink):
    """2. Personne assise légèrement floue : score bas (ex 0.20-0.30) mais >= seuil, suivie."""
    manager = sim.manager
    sim.warmup()

    box_out = sitting_person_box(150.0, OUTSIDE_Y, height=80.0, width=144.0)
    box_in = sitting_person_box(150.0, INSIDE_Y, height=80.0, width=144.0)

    # Confiance modérée 0.25 (au-dessus du seuil modèle 0.10)
    sim.steps(2, [Detection(1, box_out, confidence=0.25)])
    sim.steps(2, [Detection(1, box_in, confidence=0.25)])

    assert manager.total_in == 1
    assert manager.occupancy_confirmed == 1
    assert 1 in manager.tracks
    assert manager.tracks[1].state.name == "PRESENTE"


def test_geometric_filter_accepts_plausible_and_rejects_aberrant(sim, sink):
    """3. Boîte à ratio inhabituel mais plausible acceptée ; cas aberrants rejetés."""
    # Test unitaire de check_detection_geometry
    # Ratio plausible pour personne assise (1.8 et 2.2)
    valid_18, reason_18, r18 = check_detection_geometry([0, 0, 180, 100], max_aspect_ratio=2.5)
    assert valid_18 is True
    assert reason_18 is None
    assert pytest.approx(r18) == 1.8

    valid_22, reason_22, r22 = check_detection_geometry([0, 0, 220, 100], max_aspect_ratio=2.5)
    assert valid_22 is True

    # Aberrant : ligne horizontale pure (ratio = 350 / 50 = 7.0 > 2.5)
    valid_flat, reason_flat, r_flat = check_detection_geometry([0, 0, 350, 50], max_aspect_ratio=2.5)
    assert valid_flat is False
    assert reason_flat == "aspect_ratio_out_of_range"

    # Aberrant : poteau ultra-fin (ratio = 5 / 200 = 0.025 < 0.1)
    valid_thin, reason_thin, r_thin = check_detection_geometry([0, 0, 5, 200], min_aspect_ratio=0.1)
    assert valid_thin is False

    # Aberrant : boîte trop petite (hauteur < min_height_px)
    valid_small, reason_small, _ = check_detection_geometry([0, 0, 20, 5], min_height_px=10.0)
    assert valid_small is False
    assert reason_small == "box_too_small"

    # Vérification dans OccupancyManager : boîte aberrante filtrée et événement émis
    sim.warmup()
    aberrant_box = np.array([100.0, 100.0, 450.0, 150.0])  # W=350, H=50 -> ratio=7.0
    sim.step([Detection(99, aberrant_box, confidence=0.85)])

    rejected_events = sink.of_type("DETECTION_REJECTED_GEOMETRY")
    assert len(rejected_events) >= 1
    ev = rejected_events[-1]
    assert ev["reason"] == "aspect_ratio_out_of_range"
    assert ev["technical_track_id"] == 99
    assert ev["confidence"] == 0.85
    assert ev["bbox_wh_ratio"] == 7.0
    assert 99 not in sim.manager.tracks


def test_comparative_strict_vs_relaxed_geometric_filter(make_config, sink, frame):
    """4. Test comparatif : rejetée sous filtre strict (1.2), acceptée sous filtre assoupli (2.5)."""
    sitting_box = sitting_person_box(150.0, OUTSIDE_Y, height=80.0, width=144.0)  # ratio = 1.8

    # A. Filtre strict (ex. max_aspect_ratio_wh = 1.2)
    cfg_strict = make_config({"geometry": {"max_aspect_ratio_wh": 1.2}})
    sink_strict = type(sink)("strict")
    sim_strict = Sim(build_manager(cfg_strict, sink_strict), frame)
    sim_strict.warmup()
    sim_strict.step([Detection(1, sitting_box, confidence=0.9)])

    rejected_strict = sink_strict.of_type("DETECTION_REJECTED_GEOMETRY")
    assert len(rejected_strict) == 1
    assert rejected_strict[0]["reason"] == "aspect_ratio_out_of_range"
    assert 1 not in sim_strict.manager.tracks

    # B. Filtre assoupli (max_aspect_ratio_wh = 2.5)
    cfg_relaxed = make_config({"geometry": {"max_aspect_ratio_wh": 2.5}})
    sink_relaxed = type(sink)("relaxed")
    sim_relaxed = Sim(build_manager(cfg_relaxed, sink_relaxed), frame)
    sim_relaxed.warmup()
    sim_relaxed.step([Detection(1, sitting_box, confidence=0.9)])

    rejected_relaxed = sink_relaxed.of_type("DETECTION_REJECTED_GEOMETRY")
    assert len(rejected_relaxed) == 0
    assert 1 in sim_relaxed.manager.tracks


def test_observe_yolo_detections_diagnostics(sink):
    """Vérifie la journalisation systématique des détections brutes et sous seuil."""
    diag = DetectionDiagnostics()
    # 3 détections dont 2 sous le seuil 0.30
    boxes = np.array([[10, 10, 50, 100], [20, 20, 60, 120], [30, 30, 70, 140]])
    confidences = np.array([0.15, 0.45, 0.22])

    diag.observe_yolo_detections(boxes, confidences, confidence_threshold=0.30, frame_index=1)
    assert diag.raw_yolo_detections_count == 3
    assert diag.under_threshold_detections_count == 2

    summary = diag.summary()
    assert summary["raw_yolo_detections"] == 3
    assert summary["under_threshold_detections"] == 2

    diag.observe_rejected_geometry("aspect_ratio_out_of_range", 3.0, 0.8, 1.0, 1)
    assert diag.rejected_geometry_detections_count == 1
    assert diag.summary()["rejected_geometry_detections"] == 1


# ===========================================================================
# 2. Tests Stabilité de l'ancre et chute de hauteur
# ===========================================================================

def test_standing_person_stable_height_anchor_reliable(sim, sink):
    """5. Personne debout dont la hauteur est stable : ancre fiable, pas d'événement."""
    sim.warmup()
    # 15 frames avec hauteur constante de 200px
    for _ in range(15):
        box = person_box(150.0, OUTSIDE_Y, height=200.0, width=80.0)
        sim.step([Detection(1, box, confidence=0.9)])

    track = sim.manager.tracks[1]
    assert track.anchor_unreliable_reason is None
    assert track.height_drop_streak == 0
    assert len(sink.of_type("ANCHOR_UNRELIABLE")) == 0


def test_sudden_height_drop_triggers_unreliable_anchor_and_suspends(sim, sink):
    """6. Personne debout dont les jambes sont masquées : chute brutale, ancre non fiable, pas de franchissement."""
    manager = sim.manager
    sim.warmup()

    # 10 frames de personne debout à l'extérieur (y=250, ligne à 300)
    for _ in range(10):
        box = person_box(150.0, OUTSIDE_Y, height=200.0, width=80.0)
        sim.step([Detection(1, box, confidence=0.9)])

    assert manager.tracks[1].state.name == "EXTERIEUR"

    # Jambes occultées par un obstacle bas : le bas de boîte remonte artificiellement
    # de 250 vers 150 (saut de 100px vers le haut), hauteur passe de 200 à 100 (drop = 50% > 35%)
    occluded_legs_box = person_box(150.0, 150.0, height=100.0, width=80.0)
    sim.step([Detection(1, occluded_legs_box, confidence=0.9)])

    track = manager.tracks[1]
    # L'ancre a été marquée non fiable pour height_drop
    assert track.anchor_unreliable_reason == "height_drop"
    # L'état est resté EXTERIEUR (décision suspendue, aucun franchissement induit)
    assert track.state.name == "EXTERIEUR"
    assert manager.total_in == 0
    assert manager.total_out == 0

    unreliable_events = sink.of_type("ANCHOR_UNRELIABLE")
    assert len(unreliable_events) >= 1
    ev = unreliable_events[0]
    assert ev["reason"] == "height_drop"
    assert ev["person_id"] == 1
    assert ev["previous_height"] == 200.0
    assert ev["current_height"] == 100.0
    assert ev["drop_ratio"] == 0.5


def test_progressive_height_reduction_walking_away_stays_reliable():
    """7. Personne qui s'éloigne : hauteur diminue progressivement, ancre reste fiable."""
    # Simulation de 30 frames avec une diminution progressive de 200px à 150px
    history = []
    heights = np.linspace(200.0, 150.0, 30)

    for h in heights:
        is_drop, drop_ratio, ref_h = check_anchor_height_drop(h, history, max_drop_ratio=0.35)
        assert is_drop is False, f"La diminution progressive à h={h:.1f} ne doit pas être une chute"
        history.append(float(h))
        if len(history) > 30:
            history.pop(0)


def test_sitting_down_and_staying_seated_adapts_and_recovers_reliability(sim, sink):
    """8. Personne qui s'assoit et reste assise : chute initiale -> non fiable, puis réadaptation."""
    sim.warmup()

    # Phase debout (10 frames, H=200)
    for _ in range(10):
        box = person_box(150.0, OUTSIDE_Y, height=200.0, width=80.0)
        sim.step([Detection(1, box, confidence=0.9)])

    # S'assoit : H chute à 100 (drop_confirm_frames = 3 par défaut)
    seated_box = person_box(150.0, OUTSIDE_Y, height=100.0, width=120.0)

    # Frame 1 de la chute : instable, non fiable
    sim.step([Detection(1, seated_box, confidence=0.9)])
    track = sim.manager.tracks[1]
    assert track.anchor_unreliable_reason == "height_drop"

    # Frame 2 : toujours en cours de confirmation
    sim.step([Detection(1, seated_box, confidence=0.9)])
    assert track.anchor_unreliable_reason == "height_drop"

    # Frame 3 : confirmation atteinte (streak >= 3) -> fiabilité rétablie
    sim.step([Detection(1, seated_box, confidence=0.9)])
    assert track.anchor_unreliable_reason is None
    assert track.height_drop_streak == 0

    # Frame 4+ : l'historique a été réadapté à 100, la personne assise est suivie normalement
    sim.step([Detection(1, seated_box, confidence=0.9)])
    assert track.anchor_unreliable_reason is None


def test_crossing_during_height_drop_is_delayed_not_lost(sim, sink):
    """9. Franchissement pendant une chute de hauteur : suspendu puis confirmé, pas perdu."""
    manager = sim.manager
    sim.warmup()

    # Personne debout à l'extérieur (y=250, H=200)
    for _ in range(5):
        box = person_box(150.0, OUTSIDE_Y, height=200.0, width=80.0)
        sim.step([Detection(1, box, confidence=0.9)])

    assert manager.tracks[1].state.name == "EXTERIEUR"

    # La personne avance à l'intérieur (y=450) mais au passage ses jambes sont occultées (H=100)
    box_inside_occluded = person_box(150.0, INSIDE_Y, height=100.0, width=80.0)

    # Frame de transition avec chute de hauteur : la décision est suspendue
    sim.step([Detection(1, box_inside_occluded, confidence=0.9)])
    assert manager.total_in == 0  # Pas encore compté pendant la suspension

    # Deuxième frame avec H=100
    sim.step([Detection(1, box_inside_occluded, confidence=0.9)])
    assert manager.total_in == 0

    # Troisième frame avec H=100 : confirmation de la nouvelle hauteur, l'ancre redevient fiable
    # Le franchissement amorcé/effectif est évalué et confirmé
    sim.step([Detection(1, box_inside_occluded, confidence=0.9)])
    assert manager.total_in == 1
    assert manager.occupancy_confirmed == 1
    assert manager.tracks[1].state.name == "PRESENTE"
