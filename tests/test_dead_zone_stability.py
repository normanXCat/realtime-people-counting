"""Stabilité de la zone morte : elle suit la hauteur **stabilisée**, pas le bruit.

Symptôme corrigé : la zone morte « grandit et rétrécit » dans le temps. Elle est
une fraction de l'échelle locale (``geometry.dead_zone_ratio`` × hauteur médiane
observée) — c'est correct — mais l'échelle locale était alimentée par la hauteur
**brute et instantanée** de chaque boîte. Toute variation normale de détection
(bruit, début d'occlusion, changement de pose) faisait donc bouger la zone morte
frame par frame.

L'échelle locale est désormais alimentée par la hauteur stabilisée du
``BBoxHeightLocker`` (moyenne glissante par ``person_id``, fenêtre
``anchor.height_history_window_frames``) : la **même** source que celle de
l'ancre, sans seconde logique de lissage.

Repère géométrique : image 600×600, ligne à y = 300, ``inside_side: negative``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bbox_height_locker import BBoxHeightLocker
from conftest import TEST_FPS, make_test_line, person_box
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

INSIDE_Y = 450.0
TINY_FRAME = np.zeros((600, 600, 3), dtype=np.uint8)


def det(track_id: int, height: float, y: float = INSIDE_Y, x: float = 300.0) -> Detection:
    return Detection(track_id, person_box(x, y, height=height), 0.95)


class Sim:
    def __init__(self, manager, frame=TINY_FRAME):
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


def build(config, sink, locker=None) -> OccupancyManager:
    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=make_test_line(inside_side=config.line.inside_side),
        bbox_locker=locker,
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


def make_locker(config) -> BBoxHeightLocker:
    """Verrouillage construit **depuis la configuration**, comme ``main``."""
    return BBoxHeightLocker(
        min_height_ratio=config.stabilization.bbox_locker_min_height_ratio,
        history_window=config.anchor.height_history_window_frames,
    )


def dead_zone_px(manager) -> float:
    """Largeur réellement utilisée pour classer la frame courante."""
    value = manager.scale.ratio_to_px(manager.config.geometry.dead_zone_ratio)
    assert value is not None, "l'échelle locale doit être disponible"
    return float(value)


# ---------------------------------------------------------------------------
# Personne immobile, hauteur stable
# ---------------------------------------------------------------------------
def test_immobile_person_keeps_a_constant_dead_zone(test_config, sink):
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager).steps(4)                     # warm-up
    widths = [
        dead_zone_px(sim.steps(1, [det(1, 200.0)]).manager)
        for _ in range(25)
    ]
    assert max(widths) - min(widths) < 0.5, (
        f"la zone morte doit rester quasi constante, écarts={widths}"
    )
    assert widths[-1] == pytest.approx(0.15 * 200.0, abs=1.0)


# ---------------------------------------------------------------------------
# Bruit / occultation basse : la zone morte ne doit pas suivre la boîte brute
# ---------------------------------------------------------------------------
def test_dead_zone_does_not_follow_a_collapsing_box_when_the_locker_is_on(
    test_config, sink
):
    """Hauteur brute alternée 200/120 : la hauteur stabilisée reste ~200.

    120 px est sous ``bbox_locker_min_height_ratio`` × 200 = 140 px : c'est une
    occlusion basse, pas un changement réel de distance. La zone morte doit donc
    rester celle d'une personne debout.
    """
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager).steps(4)
    sim.steps(6, [det(1, 200.0)])

    widths: list[float] = []
    for _ in range(4):
        sim.steps(4, [det(1, 200.0)])
        widths.append(dead_zone_px(manager))
        sim.steps(4, [det(1, 120.0)])
        widths.append(dead_zone_px(manager))

    assert max(widths) - min(widths) < 0.5
    # La zone morte n'a pas rétréci vers celle d'une hauteur de 120 px.
    assert min(widths) > 0.15 * 120.0 + 5.0
    assert widths[-1] == pytest.approx(0.15 * 200.0, abs=1.0)


def test_raw_heights_would_move_the_dead_zone_without_any_stabilization(
    test_config, sink
):
    """Comparaison avant/après sur le scénario identique (preuve du correctif).

    Sans verrouillage, l'échelle locale est la médiane des hauteurs brutes : les
    frames à 120 px tirent la zone morte vers le bas, alors que la personne n'a
    pas changé de distance.
    """
    manager = build(test_config, sink)  # aucun verrouillage : comportement d'avant
    sim = Sim(manager).steps(4)
    sim.steps(6, [det(1, 200.0)])
    before = dead_zone_px(manager)
    sim.steps(20, [det(1, 120.0)])
    after = dead_zone_px(manager)

    assert after < before - 0.5, "sans stabilisation, la zone morte suit la boîte brute"


# ---------------------------------------------------------------------------
# Rapprochement réel : adaptation progressive, sans à-coup
# ---------------------------------------------------------------------------
def test_dead_zone_grows_progressively_when_the_person_really_approaches(
    test_config, sink
):
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager).steps(4)
    sim.steps(20, [det(1, 200.0)])
    start = dead_zone_px(manager)

    widths = []
    for height in np.linspace(200.0, 320.0, 60):
        widths.append(dead_zone_px(sim.steps(1, [det(1, float(height))]).manager))

    assert widths[-1] > start + 2.0, "la zone morte doit finir par suivre la personne"
    deltas = [abs(b - a) for a, b in zip(widths, widths[1:])]
    assert max(deltas) < 1.0, f"aucun à-coup visible : deltas={deltas}"


# ---------------------------------------------------------------------------
# Une seule source de lissage : celle du verrouillage, jamais une seconde
# ---------------------------------------------------------------------------
def test_local_scale_uses_the_stabilized_height_not_the_raw_one(test_config, sink):
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager).steps(4)
    sim.steps(10, [det(1, 200.0)])
    sim.steps(2, [det(1, 120.0)])     # chute : boîte brute 120, corrigée 200

    profile = locker.tracks_profile[1]
    assert profile["h_stable"] == pytest.approx(200.0)
    # L'échelle locale reflète la hauteur stabilisée, pas la boîte brute.
    assert manager.scale.median_height == pytest.approx(200.0, abs=1.0)


def test_history_window_follows_the_anchor_configuration(test_config):
    """La fenêtre de lissage n'est pas un second seuil : c'est celle de l'ancre."""
    locker = make_locker(test_config)
    assert locker.history_window == test_config.anchor.height_history_window_frames
    assert locker.min_height_ratio == pytest.approx(
        test_config.stabilization.bbox_locker_min_height_ratio
    )

    window = 3
    short = BBoxHeightLocker(min_height_ratio=0.70, history_window=window)
    for height in (100.0, 200.0, 300.0, 400.0):
        short.process_bbox(1, person_box(300.0, 500.0, height=height))
    # Seules les ``window`` dernières hauteurs comptent pour la référence.
    assert short.stable_height(1) == pytest.approx((200.0 + 300.0 + 400.0) / 3.0)
    assert short.stable_height(99) is None
    with pytest.raises(ValueError):
        BBoxHeightLocker(history_window=0)
