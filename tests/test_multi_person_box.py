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
from geometry import heads_suggest_merge
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager


def unit(*values: float) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    return vec / float(np.linalg.norm(vec))


def det(
    track_id: int,
    bbox,
    confidence: float = 0.9,
    head_points: tuple[tuple[float, float], ...] = (),
) -> Detection:
    return Detection(
        technical_track_id=track_id,
        bbox=np.asarray(bbox, dtype=float),
        confidence=confidence,
        head_points=head_points,
    )


def build_manager(
    test_config,
    sink: ListSink,
    *,
    multi_person_enabled: bool = True,
    max_growth_ratio: float = 1.6,
    head_assist: bool = False,
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
            "presence": {"head_assist": {"enabled": head_assist}},
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


# ---------------------------------------------------------------------------
# Lot B.5b — Second signal : comptage de têtes dans la boîte
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heads,width,expected",
    [
        # 100 px de séparation pour 200 px de large : 100 > 0,4 × 200 = 80.
        (((0.0, 0.0), (100.0, 0.0)), 200.0, True),
        # 70 px < 80 : sous le seuil, pas de conclusion de fusion.
        (((0.0, 0.0), (70.0, 0.0)), 200.0, False),
        # Une seule tête : un point isolé ne signale pas deux personnes.
        (((0.0, 0.0),), 200.0, False),
        ((), 200.0, False),
        # Largeur nulle : aucune séparation relative n'est définie.
        (((0.0, 0.0), (100.0, 0.0)), 0.0, False),
    ],
)
def test_heads_suggest_merge_seuils(heads, width, expected):
    flag, separation = heads_suggest_merge(heads, width, min_separation_ratio=0.4)
    assert flag is expected
    assert separation >= 0.0


def test_deux_tetes_separees_declenchent_le_signal_des_la_premiere_frame(
    test_config, sink, frame
):
    """L'angle mort du détecteur par largeur est comblé (lot B.5b).

    Le signal par largeur exige un historique de largeur d'**avant** la fusion :
    deux personnes assises côte à côte dès la première frame ne le produisent
    jamais, leur largeur fusionnée *étant* la référence. Le comptage de têtes
    n'exige aucun historique.
    """
    manager, _, _ = build_manager(test_config, sink, head_assist=True)

    merged_box = [100.0, 150.0, 300.0, 350.0]   # w = 200
    heads = ((150.0, 180.0), (270.0, 180.0))    # séparation 120 > 0,4 × 200 = 80
    sink.events.clear()

    # UNE seule frame : aucune histoire de largeur n'existe encore.
    manager.process_frame([det(10, merged_box, head_points=heads)], frame, 0.1, 1)

    events = sink.of_type("POSSIBLE_MULTI_PERSON_BOX")
    assert len(events) == 1, "le signal de têtes doit exister dès la frame 1"
    event = events[0]
    assert event["signal"] == "multiple_heads"
    assert event["head_separation_px"] == pytest.approx(120.0, abs=1.0)
    assert event["current_width"] == pytest.approx(200.0, abs=1.0)
    # Le signal par largeur, lui, n'a rien pu mesurer : la référence *est* la
    # largeur fusionnée.
    assert event["ratio"] == pytest.approx(1.0, abs=1e-6)


def test_signal_par_tetes_inactif_quand_l_assistance_tete_est_desactivee(
    test_config, sink, frame
):
    """Sans modèle pose, seul le signal par largeur subsiste (comportement inchangé)."""
    manager, _, _ = build_manager(test_config, sink, head_assist=False)
    assert test_config.presence.head_assist.enabled is False

    merged_box = [100.0, 150.0, 300.0, 350.0]
    heads = ((150.0, 180.0), (270.0, 180.0))
    sink.events.clear()
    for step in range(1, 6):
        manager.process_frame(
            [det(10, merged_box, head_points=heads)], frame, step * 0.1, step
        )

    # Largeur constante : aucun signal par largeur, et le signal de têtes est
    # désactivé faute de modèle pose.
    assert sink.count("POSSIBLE_MULTI_PERSON_BOX") == 0


def test_une_seule_tete_dans_une_boite_large_ne_declenche_rien(
    test_config, sink, frame
):
    """Un point tête isolé ne vaut pas fusion, même dans une boîte large."""
    manager, _, _ = build_manager(test_config, sink, head_assist=True)

    wide_box = [100.0, 150.0, 300.0, 350.0]     # w = 200
    single_head = ((200.0, 180.0),)
    sink.events.clear()
    for step in range(1, 6):
        manager.process_frame(
            [det(10, wide_box, head_points=single_head)], frame, step * 0.1, step
        )

    assert sink.count("POSSIBLE_MULTI_PERSON_BOX") == 0


# ---------------------------------------------------------------------------
# Lot B.5a — Conséquence sur l'occupation : la personne absorbée reste comptée
# ---------------------------------------------------------------------------
def test_personne_absorbee_reste_en_borne_haute_jamais_purgee_comme_une_sortie(
    test_config, sink, frame
):
    """Une personne absorbée dans une boîte fusionnée n'est jamais retirée en silence.

    Elle perd sa détection propre : sans arbitrage explicite, elle disparaîtrait
    du comptage exactement comme si elle était sortie. Elle doit rester comptée
    en borne haute de l'encadrement ``[observée, opérationnelle]`` et son état
    publié comme incertain.
    """
    manager, _, identity = build_manager(test_config, sink, head_assist=True)

    box_a = [60.0, 150.0, 140.0, 350.0]     # w = 80
    box_b = [180.0, 150.0, 260.0, 350.0]    # w = 80
    for step in range(1, 6):
        manager.process_frame([det(10, box_a), det(20, box_b)], frame, step * 0.1, step)

    assert manager.tracks[1].inside_occupancy is True
    assert manager.tracks[2].inside_occupancy is True
    assert manager.occupancy_operational == 2
    assert manager.occupancy_observed == 2

    # Frame 6 : la piste 20 est absorbée (plus de détection propre) et la boîte
    # restante s'élargit en contenant deux têtes séparées.
    merged_box = [60.0, 150.0, 260.0, 350.0]    # w = 200
    heads = ((100.0, 180.0), (220.0, 180.0))    # séparation 120 > 80
    sink.events.clear()
    manager.process_frame([det(10, merged_box, head_points=heads)], frame, 0.6, 6)

    events = sink.of_type("POSSIBLE_MULTI_PERSON_BOX")
    assert len(events) == 1
    event = events[0]
    # Les deux signaux se déclenchent ici (la largeur passe de 80 à 200 px ET la
    # boîte contient deux têtes séparées) : ``signal`` nomme le principal, mais la
    # preuve de têtes reste publiée.
    assert event["signal"] in ("width_growth", "multiple_heads")
    assert event["head_separation_px"] == pytest.approx(120.0, abs=1.0)
    # La personne absorbée est NOMMÉE : déjà comptée, perdue dans le rayon
    # d'ambiguïté (occlusion_ambiguity_ratio × échelle locale).
    assert event["lost_counted_neighbours"] == [2]
    assert event["ambiguity_radius_px"] is not None

    # Aucune sortie inventée : la personne reste comptée.
    assert manager.total_out == 0
    assert sink.count("OUT") == 0
    assert manager.occupancy_operational == 2

    # Au-delà de la fenêtre de galerie (grâce + rétention), elle est libérée de
    # la mémoire mais RESTE dans la borne haute : incertaine, jamais sortie.
    window_frames = int(
        (
            test_config.timing.grace_period_seconds
            + test_config.reid.long_term.gallery_retention_seconds
        )
        * TEST_FPS
    )
    for step in range(6, 6 + window_frames + 10):
        manager.process_frame(
            [det(10, merged_box, head_points=heads)], frame, step * 0.1, step
        )

    assert identity.person_of(20) is None, "la mémoire d'identité est bien libérée"
    assert manager.occupancy_operational == 2, "jamais retirée en silence"
    assert manager.occupancy_observed == 1
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (1, 2)
    assert manager.total_out == 0
    assert sink.count("OUT") == 0


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
