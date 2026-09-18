"""Composants de stabilisation (spec 7) : ``BBoxHeightLocker`` actif, prouvé.

Le ``BBoxHeightLocker`` n'est pas un chemin parallèle : il est appliqué sur la
boîte de chaque piste à chaque frame, **après** BoT-SORT et **avant** tout calcul
de zone, avec la traçabilité exigée par la spec 7 (boîte brute, boîte corrigée,
raison, seuil). Il est actif dans la configuration livrée
(``stabilization.use_bbox_locker: true``) ; l'``AnchorStabilizer`` reste
neutralisé par défaut et n'est couvert ici que pour être prouvé activable.

L'indexation de l'historique et la survie du profil à une occultation / à un
changement d'identifiant technique sont couvertes par
``tests/test_bbox_locker_integration.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from anchor_stabilizer import AnchorStabilizer
from bbox_height_locker import BBoxHeightLocker
from conftest import TEST_FPS, make_test_line, person_box
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

INSIDE_Y = 450.0
OUTSIDE_Y = 250.0


def build(config, sink, locker=None, stabilizer=None):
    identity = IdentityManager(
        long_term=config.reid.long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=make_test_line(inside_side=config.line.inside_side),
        bbox_locker=locker,
        anchor_stabilizer=stabilizer,
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


class Sim:
    def __init__(self, manager, frame):
        self.manager = manager
        self.frame = frame
        self.time = 0.0
        self.index = 0

    def step(self, detections=()):
        self.index += 1
        self.time += 1.0 / TEST_FPS
        self.manager.process_frame(list(detections), self.frame, self.time, self.index)
        return self

    def steps(self, count, detections=()):
        for _ in range(count):
            self.step(detections)
        return self


def det(track_id, y, x=300.0, height=200.0):
    return Detection(track_id, person_box(x, y, height=height), 0.9)


# ---------------------------------------------------------------------------
# Désactivation par défaut
# ---------------------------------------------------------------------------
def test_default_configuration_enables_the_bbox_locker():
    from config import load_config

    config = load_config()
    assert config.stabilization.use_bbox_locker is True
    assert config.stabilization.use_anchor_stabilizer is False


def test_no_stabilization_event_when_disabled(test_config, sink, frame):
    manager = build(test_config, sink)
    sim = Sim(manager, frame).steps(5)
    sim.steps(2, [det(1, OUTSIDE_Y)])
    # Boîte tronquée : sans locker, aucune correction n'est appliquée.
    sim.steps(2, [det(1, 280.0, height=100.0)])
    assert sink.count("STABILIZATION") == 0
    assert manager.total_in == 0


# ---------------------------------------------------------------------------
# BBoxHeightLocker activé
# ---------------------------------------------------------------------------
def test_bbox_locker_corrects_and_is_traced(test_config, sink, frame):
    locker = BBoxHeightLocker(min_height_ratio=0.70)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame).steps(5)
    sim.steps(4, [det(1, OUTSIDE_Y, height=200.0)])
    sink.events.clear()

    # Occlusion basse : hauteur 100 px < 0,70 × 200 px -> le bas est repoussé.
    raw = person_box(300.0, 280.0, height=100.0)
    sim.step([Detection(1, raw, 0.9)])

    events = sink.of_type("STABILIZATION")
    assert len(events) == 1
    event = events[0]
    assert event["component"] == "bbox_locker"
    assert event["reason"] == "hauteur_corrigee_sous_seuil"
    assert event["threshold"] == pytest.approx(0.70)
    assert event["raw_bbox"][3] == pytest.approx(280.0)
    assert event["corrected_bbox"][3] == pytest.approx(380.0)  # 180 (y1) + 200


def test_bbox_locker_changes_the_crossing_decision(test_config, sink, frame):
    """La décision repose sur la boîte corrigée, la boîte brute reste journalisée."""
    locker = BBoxHeightLocker(min_height_ratio=0.70)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame).steps(5)
    sim.steps(4, [det(1, OUTSIDE_Y, height=200.0)])
    # Boîte tronquée dont l'ancre brute (280) est dans la zone morte :
    # l'ancre corrigée (380) est franchement à l'intérieur. Les deux premières
    # frames de la chute suspendent la décision (surveillance de chute alimentée
    # par la hauteur BRUTE) ; la troisième confirme la chute, réadapte
    # l'historique, puis le franchissement est décidé sur l'ancre **corrigée**.
    sim.steps(3, [Detection(1, person_box(300.0, 280.0, height=100.0), 0.9)])

    assert manager.total_in == 1
    event = sink.of_type("STABILIZATION")[-1]
    assert event["raw_bbox"][3] == pytest.approx(280.0)
    assert event["corrected_bbox"][3] == pytest.approx(380.0)
    assert manager.tracks[1].anchor[1] == pytest.approx(380.0)


def test_local_scale_is_fed_with_the_stabilized_height(test_config, sink, frame):
    """L'échelle locale (et donc la zone morte) suit la hauteur **stabilisée**.

    La boîte brute reste journalisée et sert à détecter la chute ; c'est la
    hauteur lissée par ``person_id`` qui fixe l'échelle locale, pour que la zone
    morte ne suive pas le bruit de détection frame par frame.
    """
    locker = BBoxHeightLocker(min_height_ratio=0.70)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame).steps(5)
    sim.steps(10, [det(1, OUTSIDE_Y, height=200.0)])
    sim.steps(2, [det(1, 280.0, height=100.0)])

    assert manager.scale.samples > 0
    assert manager.scale.median_height == pytest.approx(200.0, abs=1.0)
    assert sink.count("STABILIZATION") == 2


# ---------------------------------------------------------------------------
# AnchorStabilizer activé
# ---------------------------------------------------------------------------
def test_anchor_stabilizer_is_traced_when_it_acts(test_config, sink, frame):
    stabilizer = AnchorStabilizer(tolerance_px=5.0)
    manager = build(test_config, sink, stabilizer=stabilizer)
    sim = Sim(manager, frame).steps(5)
    sim.steps(3, [det(1, OUTSIDE_Y, height=200.0)])
    sink.events.clear()
    # Effondrement brutal du bas de boîte sans mouvement de tête : recalage.
    sim.step([Detection(1, np.array([260.0, 100.0, 340.0, 220.0]), 0.9)])
    events = sink.of_type("STABILIZATION")
    assert len(events) == 1
    assert events[0]["component"] == "anchor_stabilizer"
    assert events[0]["reason"] == "ancrage_recalibre_sur_vitesse_tete"
    assert events[0]["threshold"] == pytest.approx(5.0)


def test_anchor_stabilizer_keeps_identity_deterministic():
    stabilizer = AnchorStabilizer(tolerance_px=5.0)
    box = (260.0, 100.0, 340.0, 300.0)
    first = stabilizer.get_stabilized_anchor(1, box)
    second = stabilizer.get_stabilized_anchor(1, box)
    assert first == second == (300.0, 300.0)


def test_stabilizer_caches_are_purged_with_the_frame(test_config, sink, frame):
    locker = BBoxHeightLocker()
    stabilizer = AnchorStabilizer()
    manager = build(test_config, sink, locker=locker, stabilizer=stabilizer)
    sim = Sim(manager, frame).steps(5)
    # 1re frame de la piste : ``IdentityManager`` ne connaît pas encore
    # l'identifiant technique, donc la clé de repli (négative) est utilisée ; la
    # frame suivante bascule sur le ``person_id`` résolu.
    sim.step([det(1, OUTSIDE_Y)])
    assert -2 in locker.tracks_profile  # clé de repli : ``-1 - track_id``
    sim.step([det(1, OUTSIDE_Y)])
    assert 1 in locker.tracks_profile  # clé stable : ``person_id``
    assert 1 in stabilizer.tracks_history
    locker.purge_lost_tracks(set())
    stabilizer.purge_lost_tracks(set())
    assert locker.tracks_profile == {}
    assert stabilizer.tracks_history == {}
