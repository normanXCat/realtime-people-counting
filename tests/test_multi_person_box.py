"""Niveau 1-3 — Détection de boîte contenant probablement deux personnes (multi-person box).

Couvre le signalement explicite des boîtes anormalement larges par rapport à l'historique
stabilisé (croissance anormale de largeur sans croissance de hauteur correspondante),
le maintien du comptage opérationnel, le rejet du crop de galerie pour éviter de corrompre
l'apparence de référence, et la neutralisation de la détection lors d'un rapprochement
réel de la caméra où hauteur et largeur croissent proportionnellement.
"""

from __future__ import annotations

import numpy as np
import pytest

from bbox_height_locker import BBoxHeightLocker
from conftest import TEST_FPS, StubAppearance, make_test_line
from events import ListSink
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager


def unit(*values: float) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    return vec / float(np.linalg.norm(vec))


def det(track_id: int, bbox, confidence: float = 0.9) -> Detection:
    return Detection(
        technical_track_id=track_id,
        bbox=np.asarray(bbox, dtype=float),
        confidence=confidence,
    )


def build_manager(
    test_config,
    sink: ListSink,
    *,
    multi_person_enabled: bool = True,
    max_growth_ratio: float = 1.6,
    appearance=None,
    locker=None,
):
    from config import override_config

    cfg = override_config(
        test_config,
        {
            "geometry": {
                "multi_person_box": {
                    "enabled": multi_person_enabled,
                    "max_growth_ratio": max_growth_ratio,
                }
            },
            "stabilization": {"use_bbox_locker": True},
        },
    )
    identity = IdentityManager(
        long_term=cfg.reid.long_term,
        external_reid=cfg.reid.external_reid,
        grace_period_seconds=cfg.timing.grace_period_seconds,
        event_sink=sink,
        appearance=appearance or StubAppearance(default=unit(1, 0, 0)),
    )
    if locker is None:
        locker = BBoxHeightLocker(
            min_height_ratio=cfg.stabilization.bbox_locker_min_height_ratio,
            history_window=cfg.anchor.height_history_window_frames,
        )
    manager = OccupancyManager(
        cfg,
        event_sink=sink,
        identity_manager=identity,
        line=make_test_line(inside_side=cfg.line.inside_side),
        bbox_locker=locker,
    )
    manager.set_frame_budget(cfg.timing.frame_budget(TEST_FPS))
    return manager, locker, identity


def test_boite_anormalement_large_emet_evenement_maintient_comptage_et_exclut_galerie(
    test_config, sink, frame
):
    """Une boîte dont la largeur croît brutalement émet POSSIBLE_MULTI_PERSON_BOX,
    maintient le suivi/comptage de la personne, et exclut le crop de la mise à jour galerie.
    """
    initial_appearance = unit(1, 0, 0)
    new_corrupted_appearance = unit(0, 1, 0)

    current_app = initial_appearance

    def get_app(f, b):
        return current_app

    appearance = StubAppearance(describe_fn=get_app)
    manager, locker, identity = build_manager(
        test_config, sink, appearance=appearance
    )

    # Établir l'historique stabilisé sur quelques frames (largeur = 40, hauteur = 200)
    # y = 350 -> intérieur (negative side)
    normal_box = [100.0, 150.0, 140.0, 350.0]
    for step in range(1, 6):
        manager.process_frame([det(10, normal_box)], frame, step * 0.1, step)

    assert 1 in identity.records
    record = identity.records[1]
    assert record.feature is not None
    initial_feature = np.copy(record.feature)

    # Frame 6 : la boîte double brutalement de largeur (w = 80 au lieu de 40), hauteur constante (200)
    current_app = new_corrupted_appearance
    wide_box = [100.0, 150.0, 180.0, 350.0]  # width = 80, height = 200
    sink.events.clear()

    manager.process_frame([det(10, wide_box)], frame, 0.6, 6)

    # 1. Événement POSSIBLE_MULTI_PERSON_BOX émis avec les bons champs
    events = sink.of_type("POSSIBLE_MULTI_PERSON_BOX")
    assert len(events) == 1
    ev = events[0]
    assert ev["person_id"] == 1
    assert ev["frame"] == 6
    assert ev["current_width"] == pytest.approx(80.0, abs=1.0)
    assert ev["stable_width"] == pytest.approx(40.0, abs=1.0)
    assert ev["ratio"] == pytest.approx(2.0, abs=0.1)
    assert ev["threshold"] == pytest.approx(1.6)

    # 2. Comptage et suivi maintenus : pas de double comptage, pas de perte
    assert 1 in manager.tracks
    track = manager.tracks[1]
    assert track.technical_track_id == 10
    # La personne est toujours active et suivie
    assert manager.occupancy_operational == 1
    assert manager.occupancy_observed == 1

    # 3. Crop exclu de la mise à jour de galerie : le record.feature n'a pas absorbé new_corrupted_appearance
    np.testing.assert_array_almost_equal(record.feature, initial_feature)
    rejections = sink.of_type("REID_DESCRIPTOR_REJECTED")
    assert any(r.get("reason") == "possible_multi_person_crop" for r in rejections)


def test_boite_large_coherente_avec_rapprochement_camera_aucun_evenement(
    test_config, sink, frame
):
    """Quand largeur ET hauteur augmentent proportionnellement (personne qui s'approche),
    aucun événement POSSIBLE_MULTI_PERSON_BOX n'est émis.
    """
    manager, locker, identity = build_manager(test_config, sink)

    # Historique stabilisé : w = 40, h = 200
    normal_box = [100.0, 150.0, 140.0, 350.0]
    for step in range(1, 6):
        manager.process_frame([det(10, normal_box)], frame, step * 0.1, step)

    sink.events.clear()

    # Rapprochement caméra : hauteur ET largeur doublent (w = 80, h = 400)
    approaching_box = [80.0, 50.0, 160.0, 450.0]
    manager.process_frame([det(10, approaching_box)], frame, 0.6, 6)

    assert sink.count("POSSIBLE_MULTI_PERSON_BOX") == 0


def test_multi_person_box_desactivee_ne_produit_aucun_evenement(
    test_config, sink, frame
):
    """Quand multi_person_box.enabled = False, aucun événement n'est émis et le crop n'est pas rejeté."""
    manager, locker, identity = build_manager(
        test_config, sink, multi_person_enabled=False
    )

    normal_box = [100.0, 150.0, 140.0, 350.0]
    for step in range(1, 6):
        manager.process_frame([det(10, normal_box)], frame, step * 0.1, step)

    sink.events.clear()

    # Boîte qui double de largeur, hauteur constante
    wide_box = [100.0, 150.0, 180.0, 350.0]
    manager.process_frame([det(10, wide_box)], frame, 0.6, 6)

    assert sink.count("POSSIBLE_MULTI_PERSON_BOX") == 0
    rejections = [
        r for r in sink.of_type("REID_DESCRIPTOR_REJECTED")
        if r.get("reason") == "possible_multi_person_crop"
    ]
    assert len(rejections) == 0
