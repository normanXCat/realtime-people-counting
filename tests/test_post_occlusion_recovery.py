"""Post-Occlusion Recovery, étape 1 : visibilité seule, jamais de comptage."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TEST_FPS, StubAppearance, make_test_line, person_box
from fsm import TrackState
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

INSIDE_Y = 450.0
A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])


def _config(make_config, enabled=True):
    return make_config({
        "line": {"inside_side": "negative"},
        "timing": {
            "warmup_seconds": 0.3, "warmup_min_frames": 4,
            "confirmation_seconds": 0.2, "grace_period_seconds": 1.0,
            "fps_estimate_window": 4,
        },
        "occupancy": {"snapshot_interval_seconds": 0.1},
        "display": {"enabled": False},
        "post_occlusion_recovery": {"enabled": enabled},
    })


def _manager(config, sink, vectors):
    """``vectors`` : centre horizontal -> descripteur (stub, sans réseau)."""

    def describe(_frame, bbox):
        x = float((bbox[0] + bbox[2]) / 2.0)
        key = min(vectors, key=lambda k: abs(k - x))
        return vectors[key] if abs(key - x) <= 15 else None

    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=StubAppearance(describe_fn=describe),
    )
    manager = OccupancyManager(
        config, event_sink=sink, identity_manager=identity,
        line=make_test_line(inside_side="negative"),
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


class Sim:
    def __init__(self, manager):
        self.manager = manager
        self.frame = np.zeros((600, 600, 3), dtype=np.uint8)
        self.t = 0.0
        self.i = 0

    def step(self, detections=(), raw=None):
        self.i += 1
        self.t += 1.0 / TEST_FPS
        self.manager.process_frame(list(detections), self.frame, self.t, self.i, raw_detections=raw)

    def steps(self, n, detections=(), raw=None):
        for _ in range(n):
            self.step(detections, raw)


def _det(track_id, x):
    return Detection(technical_track_id=track_id, bbox=person_box(x, INSIDE_Y), confidence=0.9)


def _raw(*items):
    """Détections brutes ``(x, conf)`` au format Ultralytics ``x1,y1,x2,y2,conf,cls``."""
    return np.array([[*person_box(x, INSIDE_Y), conf, 0.0] for x, conf in items], dtype=float)


def _occlude(sim, people):
    """Warm-up avec ``people`` = [(track_id, x)], puis disparition de tous."""
    dets = [_det(t, x) for t, x in people]
    sim.steps(8, dets)
    manager = sim.manager
    for person_id in manager.tracks:
        # Descripteur de galerie fixé : indépendant de la politique on_demand.
        x = float((manager.tracks[person_id].bbox[0] + manager.tracks[person_id].bbox[2]) / 2.0)
        manager.identities.record(person_id).feature = sim_vectors[round(x)]
    sim.step()  # frame sans détection : OCCULTEE
    for track in manager.tracks.values():
        assert track.state is TrackState.OCCULTEE


sim_vectors: dict[int, np.ndarray] = {}


@pytest.fixture(autouse=True)
def _reset_vectors():
    sim_vectors.clear()
    yield
    sim_vectors.clear()


def _counts(manager):
    return (manager.total_in, manager.total_out, manager.total_new, manager.occupancy_operational)


def test_recovery_makes_identity_visible_without_any_count(make_config, sink):
    sim_vectors.update({300: A})
    manager = _manager(_config(make_config), sink, {300.0: A})
    sim = Sim(manager)
    _occlude(sim, [(1, 300.0)])
    before = _counts(manager)
    assert manager.visible_count == 0
    sim.step(raw=_raw((300.0, 0.4)))
    assert manager.visible_count == 0  # une seule image : pas encore confirmée
    sim.step(raw=_raw((302.0, 0.4)))
    assert manager.visible_count == 1
    (track,) = manager.tracks.values()
    assert track.state is TrackState.OCCULTEE  # la FSM ne voit rien
    # Suivi par recouvrement bien au-delà de la grâce (1 s = 10 frames).
    sim.steps(30, raw=_raw((304.0, 0.4)))
    assert manager.visible_count == 1 and track.state is TrackState.OCCULTEE
    assert _counts(manager) == before
    assert sink.count("POST_OCCLUSION_RECOVERED") == 1
    assert sink.count("IN") == 0 and sink.count("OUT") == 0 and sink.count("NEW") == 0
    # La détection brute cesse : la visibilité cesse, le comptage ne bouge pas.
    sim.step()
    assert manager.visible_count == 0 and _counts(manager) == before


def test_one_missed_frame_is_tolerated_during_confirmation(make_config, sink):
    sim_vectors.update({300: A})
    manager = _manager(_config(make_config), sink, {300.0: A})
    sim = Sim(manager)
    _occlude(sim, [(1, 300.0)])
    sim.step(raw=_raw((300.0, 0.4)))
    sim.step()
    sim.step(raw=_raw((300.0, 0.4)))
    assert sink.count("POST_OCCLUSION_RECOVERED") == 1


def test_no_recovery_without_candidate(make_config, sink):
    sim_vectors.update({300: A})
    manager = _manager(_config(make_config), sink, {300.0: A, 520.0: A})
    sim = Sim(manager)
    _occlude(sim, [(1, 300.0)])
    # Trop loin (rayon = 1 hauteur = 200 px), confiance trop basse, ou
    # recouvrant une boîte suivie : aucune de ces détections n'est candidate.
    sim.steps(3, detections=[_det(9, 120.0)], raw=_raw((520.0, 0.5), (300.0, 0.2), (120.0, 0.5)))
    assert sink.count("POST_OCCLUSION_RECOVERED") == 0
    assert manager.tracks[1].recovery_active is False


def test_dissimilar_detection_is_rejected_once(make_config, sink):
    sim_vectors.update({300: A})
    manager = _manager(_config(make_config), sink, {300.0: A, 340.0: B})
    sim = Sim(manager)
    _occlude(sim, [(1, 300.0)])
    sim.steps(4, raw=_raw((340.0, 0.5)))
    assert sink.count("POST_OCCLUSION_RECOVERED") == 0
    assert sink.count("POST_OCCLUSION_REJECTED") == 1  # pas un événement par image


def test_ambiguous_between_two_occluded_identities(make_config, sink):
    sim_vectors.update({250: A, 350: A})
    manager = _manager(_config(make_config), sink, {250.0: A, 350.0: A, 300.0: A})
    sim = Sim(manager)
    _occlude(sim, [(1, 250.0), (2, 350.0)])
    sim.steps(4, raw=_raw((300.0, 0.5)))
    assert sink.count("POST_OCCLUSION_RECOVERED") == 0
    assert sink.count("POST_OCCLUSION_AMBIGUOUS") == 1
    assert manager.visible_count == 0


def test_exterior_identity_is_not_recoverable(make_config, sink):
    sim_vectors.update({300: A})
    manager = _manager(_config(make_config), sink, {300.0: A})
    sim = Sim(manager)
    _occlude(sim, [(1, 300.0)])
    manager.tracks[1].inside_occupancy = False  # identité non comptée (extérieure)
    sim.steps(4, raw=_raw((300.0, 0.5)))
    assert sink.count("POST_OCCLUSION_RECOVERED") == 0
    assert manager.visible_count == 0


def test_one_to_one_pairing(make_config, sink):
    sim_vectors.update({250: A, 400: B})
    manager = _manager(_config(make_config), sink, {250.0: A, 400.0: B, 260.0: A, 390.0: B})
    sim = Sim(manager)
    _occlude(sim, [(1, 250.0), (2, 400.0)])
    before = _counts(manager)
    sim.steps(3, raw=_raw((260.0, 0.5), (390.0, 0.5)))
    assert sink.count("POST_OCCLUSION_RECOVERED") == 2
    recovered = {e["person_id"]: e["bbox"] for e in sink.events if e["type"] == "POST_OCCLUSION_RECOVERED"}
    assert set(recovered) == {1, 2}
    assert abs((recovered[1][0] + recovered[1][2]) / 2 - 260.0) < 1
    assert abs((recovered[2][0] + recovered[2][2]) / 2 - 390.0) < 1
    assert manager.visible_count == 2 and _counts(manager) == before


def test_single_detection_goes_to_a_single_identity(make_config, sink):
    """Une détection, deux identités semblables mais écart suffisant : une seule récupérée."""
    sim_vectors.update({250: A, 350: np.array([0.6, 0.8, 0.0])})
    manager = _manager(_config(make_config), sink, {250.0: A, 350.0: sim_vectors[350], 260.0: A})
    sim = Sim(manager)
    _occlude(sim, [(1, 250.0), (2, 350.0)])
    sim.steps(3, raw=_raw((260.0, 0.5)))
    assert [e["person_id"] for e in sink.events if e["type"] == "POST_OCCLUSION_RECOVERED"] == [1]
    assert manager.visible_count == 1


def test_disabled_is_identical_to_current_version(make_config, sink):
    from events import ListSink

    results = []
    for enabled in (False, True):
        local = ListSink("pytest")
        sim_vectors.clear()
        sim_vectors.update({300: A})
        manager = _manager(_config(make_config, enabled=enabled), local, {300.0: A})
        sim = Sim(manager)
        _occlude(sim, [(1, 300.0)])
        # Sans détection brute, la récupération activée ne change rien non plus.
        raw = _raw((300.0, 0.5)) if not enabled else None
        sim.steps(20, raw=raw)
        results.append([(e["type"], e.get("person_id")) for e in local.events
                        if e["type"] not in ("OCCUPANCY_SNAPSHOT",)])
        if not enabled:
            assert manager.recovery is None
            assert local.count("POST_OCCLUSION_RECOVERED") == 0
    assert results[0] == results[1]


def test_disabled_by_default_and_published():
    import yaml

    from config import load_config

    config = load_config()
    assert config.post_occlusion_recovery.enabled is False
    dumped = yaml.safe_load(yaml.safe_dump(config.to_dict()))
    assert dumped["post_occlusion_recovery"]["similarity_threshold"] == pytest.approx(0.66)
    assert dumped["post_occlusion_recovery"]["radius_height_ratio"] == pytest.approx(1.0)


def test_alternating_reasons_are_logged_once_per_episode(make_config, sink):
    sim_vectors.update({250: A, 350: A})
    manager = _manager(_config(make_config), sink, {250.0: A, 350.0: A, 300.0: A, 200.0: B})
    sim = Sim(manager)
    _occlude(sim, [(1, 250.0), (2, 350.0)])
    for _ in range(3):  # ambiguïté puis rejet, en alternance
        sim.step(raw=_raw((300.0, 0.5)))
        sim.step(raw=_raw((200.0, 0.5)))
    assert sink.count("POST_OCCLUSION_AMBIGUOUS") == 1
    assert sink.count("POST_OCCLUSION_REJECTED") == 1
