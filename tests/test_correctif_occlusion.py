"""Niveau 1-3 — Corrections « occlusion dense et tête » (lots A et B).

Ce fichier couvre les corrections demandées par
``prompt/prompt_correctif_occlusion_et_tete.md`` :

- **A.4** : incohérence de base de temps ``track_buffer`` (frames) vs galerie ReID
  (secondes) — avertissement, jamais refus ;
- **B.1** : la réadaptation de hauteur d'une personne assise n'est plus neutralisée
  par l'activation de l'assistance tête ;
- **B.2** : le maintien par la tête dans ``_handle_missing`` exige un point tête
  frais (frame courante ou précédente) ;
- **B.3.a** : ``feature_momentum`` est configurable et validé ;
- **B.3.b** : le descripteur de galerie est **gelé** pendant une discontinuité
  d'apparence, sinon la référence dérive vers l'autre personne après un swap ;
- **B.4** : descripteur HSV par bandes sur un crop érodé, paramétré par la config ;
- **B.5** : second signal multi-personnes (nombre de têtes dans la boîte) et
  conséquence de l'absorption sur la purge d'une identité comptée.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import (
    AppearanceContinuityConfig,
    ConfigError,
    LongTermReidConfig,
    SwapCorrectionConfig,
    load_config,
    override_config,
)
from conftest import (
    TEST_FPS,
    StubAppearance,
    identity_vectors_by_x,
    make_test_line,
)
from events import ListSink
from geometry import detect_multi_person_by_head_count, extract_head_point
from identity_manager import (
    HistogramAppearanceExtractor,
    IdentityManager,
    Observation,
)
from main import track_buffer_timebase_warning
from occupancy_manager import Detection, OccupancyManager
from occupancy_types import PersonTrack
from fsm import TrackState

FRAME = np.zeros((600, 600, 3), dtype=np.uint8)


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


def observation(track_id: int, anchor=(100.0, 300.0), feature=None) -> Observation:
    box = np.asarray([80.0, 100.0, 120.0, 300.0], dtype=float)
    return Observation(
        technical_track_id=track_id,
        bbox=box,
        confidence=0.95,
        anchor=tuple(anchor),
        bbox_height=200.0,
        feature=None if feature is None else np.asarray(feature, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# A.4 — base de temps track_buffer (frames) vs galerie ReID (secondes)
# ---------------------------------------------------------------------------
def test_avertissement_de_base_de_temps_seulement_pour_une_camera_live():
    config = load_config()
    assert track_buffer_timebase_warning(config, "clip.mp4") is None
    assert track_buffer_timebase_warning(config, 0) is not None


def test_avertissement_de_base_de_temps_chiffre_au_fps_mesure():
    """150 frames à 3,5 FPS ≈ 42,9 s de temps réel (et non 5 s de contenu)."""
    config = load_config()
    message = track_buffer_timebase_warning(config, 0, fps=3.5)
    assert message is not None
    assert "FRAMES" in message and "SECONDES" in message
    assert "42.9" in message


# ---------------------------------------------------------------------------
# B.1 — personne assise : réadaptation de hauteur, assistance tête ou non
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head_assist", [False, True])
def test_personne_assise_stabilisee_redevient_fiable(make_config, sink, head_assist):
    """Une chute de hauteur confirmée n'est plus un motif permanent d'ancrage non fiable.

    Avant B.1, la branche de réadaptation était neutralisée dès que
    ``presence.head_assist.enabled`` valait ``true`` : une personne assise
    restait ``anchor_unreliable`` indéfiniment, sa coordonnée verticale gelée sur
    ``last_reliable_anchor``, puis basculait en ``OCCULTEE`` à l'expiration de
    ``max_presence_extension_seconds``. Le test vérifie les DEUX configurations.
    """
    config = make_config(
        {
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
                    "enabled": head_assist,
                    "min_keypoint_confidence": 0.5,
                    "max_presence_extension_seconds": 10.0,
                }
            },
        }
    )
    manager = OccupancyManager(config, event_sink=sink, line=make_test_line())
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))

    standing = np.array([260.0, 250.0, 340.0, 450.0])  # hauteur 200
    sitting = np.array([260.0, 350.0, 340.0, 450.0])  # hauteur 100 (chute de 50 %)
    for index in range(1, 10):
        manager.process_frame([Detection(1, standing, 0.9)], FRAME, index * 0.1, index)

    track = manager.tracks[1]
    assert track.state is TrackState.PRESENTE

    for index in range(10, 20):
        manager.process_frame([Detection(1, sitting, 0.9)], FRAME, index * 0.1, index)

    assert track.state is TrackState.PRESENTE
    assert track.height_drop_streak == 0
    assert track.anchor_unreliable_reason is None
    # L'ancre a été réadaptée à la nouvelle hauteur et n'est plus gelée.
    assert track.last_reliable_anchor == pytest.approx((300.0, 450.0))
    assert track.anchor == pytest.approx((300.0, 450.0))


# ---------------------------------------------------------------------------
# B.2 — fraîcheur du point tête dans _handle_missing
# ---------------------------------------------------------------------------
def _manager_with_present_track(make_config, sink, head_index: int):
    config = make_config(
        {
            "line": {"inside_side": "negative"},
            "display": {"enabled": False},
            "presence": {
                "head_assist": {
                    "enabled": True,
                    "min_keypoint_confidence": 0.5,
                    "max_presence_extension_seconds": 10.0,
                    "max_head_staleness_frames": 1,
                }
            },
        }
    )
    manager = OccupancyManager(config, event_sink=sink, line=make_test_line())
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    manager._warmup_done = True
    manager._frame_size = (600, 600)
    manager.scale.observe_many([np.array([260.0, 250.0, 340.0, 450.0])])
    track = PersonTrack(
        person_id=1,
        state=TrackState.PRESENTE,
        technical_track_id=5,
        anchor=(300.0, 450.0),
        inside_occupancy=True,
        last_seen_frame=head_index,
        last_seen_s=head_index * 0.1,
        current_head_point=(300.0, 300.0),
        current_head_confidence=0.9,
        head_point_frame_index=head_index,
    )
    manager.tracks[1] = track
    return manager, track


def test_maintien_par_la_tete_accepte_si_le_point_est_frais(make_config, sink):
    manager, track = _manager_with_present_track(make_config, sink, head_index=10)
    manager._handle_missing(track, manager.budget, 1.1, 11)  # 11 - 10 = 1 frame
    assert sink.count("PRESENCE_MAINTAINED_BY_HEAD") >= 1
    assert track.state is TrackState.PRESENTE


def test_maintien_par_la_tete_refuse_si_le_point_est_perime(make_config, sink):
    manager, track = _manager_with_present_track(make_config, sink, head_index=8)
    manager._handle_missing(track, manager.budget, 1.2, 11)  # 11 - 8 = 3 frames
    assert sink.count("PRESENCE_MAINTAINED_BY_HEAD") == 0
    assert sink.count("PRESENCE_EXTENSION_EXPIRED") == 0


# ---------------------------------------------------------------------------
# B.3.a — feature_momentum configurable
# ---------------------------------------------------------------------------
def test_feature_momentum_est_configurable_et_valide():
    config = load_config()
    assert config.reid.long_term.feature_momentum == pytest.approx(0.85)

    overridden = override_config(
        config, {"reid": {"long_term": {"feature_momentum": 0.5}}}
    )
    manager = OccupancyManager(overridden, line=make_test_line())
    assert manager.identities.feature_momentum == pytest.approx(0.5)

    with pytest.raises(ConfigError):
        override_config(config, {"reid": {"long_term": {"feature_momentum": 1.0}}})
    with pytest.raises(ConfigError):
        override_config(config, {"reid": {"long_term": {"feature_momentum": -0.1}}})


# ---------------------------------------------------------------------------
# B.3.b — gel du descripteur pendant une discontinuité d'apparence
# ---------------------------------------------------------------------------
def test_descripteur_gele_pendant_une_discontinuite(sink):
    """La référence d'apparence ne doit pas absorber l'autre personne après un swap."""
    manager = IdentityManager(
        appearance_continuity=AppearanceContinuityConfig(
            enabled=True,
            similarity_threshold=0.5,
            max_gap_frames=2,
            min_interval_seconds=0.0,
            freeze_gallery_frames=5,
        ),
        event_sink=sink,
    )
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    gelé = manager.records[1].feature.copy()

    # Frame 2 : discontinuité franche (apparence orthogonale).
    manager.assign([observation(7, feature=unit(0, 1, 0))], FRAME, 0.1, 2)
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 1
    assert np.allclose(manager.records[1].feature, gelé)

    # Frames 3 à 6 : apparence CONTINUE mais différente — le gel doit tenir
    # (gel posé jusqu'à la frame 2 + 5 = 7).
    nouvelle = unit(0.8, 0.6, 0.0)
    for index in range(3, 7):
        manager.assign([observation(7, feature=nouvelle)], FRAME, index * 0.1, index)
    assert np.allclose(
        manager.records[1].feature, gelé
    ), "le descripteur doit rester figé pendant toute la fenêtre de gel"

    # Frame 7 (inclusive) encore gelée, puis la mise à jour reprend.
    manager.assign([observation(7, feature=nouvelle)], FRAME, 0.7, 7)
    assert np.allclose(manager.records[1].feature, gelé)
    manager.assign([observation(7, feature=nouvelle)], FRAME, 0.8, 8)
    assert not np.allclose(manager.records[1].feature, gelé)


def test_correction_croisee_possible_apres_cinq_frames_de_swap(sink):
    """Avec le gel, la référence reste celle d'AVANT le swap après 5 frames.

    Sans gel, ``_blend`` absorbait 1 − 0.85⁵ ≈ 56 % de l'apparence de l'autre
    personne en 5 frames, et ``_correct_cross_swaps`` devenait aveugle.
    """
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True,
            similarity_threshold=0.4,
            safety_margin=0.0,
            v_max_ratio=1e6,
            spatial_margin_ratio=1e6,
        ),
        appearance_continuity=AppearanceContinuityConfig(
            enabled=True,
            similarity_threshold=0.5,
            max_gap_frames=2,
            min_interval_seconds=0.0,
            freeze_gallery_frames=10,
        ),
        # Marge volontairement inatteignable : aucune correction immédiate, on
        # laisse 5 frames de swap s'installer avant de tester la correction.
        swap_correction=SwapCorrectionConfig(enabled=True, margin=5.0),
        grace_period_seconds=5.0,
        event_sink=sink,
    )
    desc_a, desc_b = unit(1, 0, 0), unit(0, 1, 0)
    manager.assign(
        [
            observation(7, anchor=(100.0, 300.0), feature=desc_a),
            observation(9, anchor=(200.0, 300.0), feature=desc_b),
        ],
        FRAME, 0.0, 1,
    )
    reference_a = manager.records[1].feature.copy()
    reference_b = manager.records[2].feature.copy()

    for index in range(2, 7):  # 5 frames d'apparence inversée
        manager.assign(
            [
                observation(7, anchor=(150.0, 300.0), feature=desc_b),
                observation(9, anchor=(160.0, 300.0), feature=desc_a),
            ],
            FRAME, index * 0.1, index,
        )
    assert np.allclose(manager.records[1].feature, reference_a)
    assert np.allclose(manager.records[2].feature, reference_b)

    # La marge redevient atteignable : la comparaison croisée utilise la
    # référence intacte et corrige l'inversion.
    manager.swap_correction = SwapCorrectionConfig(enabled=True, margin=0.5)
    manager.assign(
        [
            observation(7, anchor=(150.0, 300.0), feature=desc_b),
            observation(9, anchor=(160.0, 300.0), feature=desc_a),
        ],
        FRAME, 0.7, 7,
    )
    assert sink.count("IDENTITY_SWAP_CORRECTED") == 1
    assert manager.person_of(7) == 2
    assert manager.person_of(9) == 1


# ---------------------------------------------------------------------------
# B.4 — descripteur HSV par bandes
# ---------------------------------------------------------------------------
def test_descripteur_par_bandes_est_parametre_par_la_configuration():
    config = load_config()
    extractor = HistogramAppearanceExtractor.from_config(
        config.reid.long_term.descriptor
    )
    assert extractor.bands == 3
    assert extractor.band_weights == [1.0, 1.0, 0.5]
    assert extractor.horizontal_crop_ratio == pytest.approx(0.15)

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[40:160, 40:160] = (10, 120, 200)
    vector = extractor.describe(image, [40, 40, 160, 160])
    assert vector is not None
    # Dimension = bandes × bacs teinte × bacs saturation.
    assert vector.shape == (3 * 24 * 8,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)

    plain = IdentityManager()
    assert isinstance(plain.appearance, HistogramAppearanceExtractor)
    assert plain.appearance.bands == 3


def test_erosion_horizontale_ignore_le_fond_et_les_voisins():
    """15 % de chaque côté sont rognés : le fond n'entre plus dans le descripteur."""
    extractor = HistogramAppearanceExtractor(horizontal_crop_ratio=0.15)
    interior = (10, 120, 200)
    background = (0, 0, 255)

    with_border = np.full((160, 160, 3), background, dtype=np.uint8)
    with_border[:, 24:136] = interior
    plain = np.full((160, 160, 3), interior, dtype=np.uint8)

    box = [0, 0, 160, 160]
    assert np.allclose(
        extractor.describe(with_border, box), extractor.describe(plain, box)
    )


def test_chaque_bande_est_normalisee_puis_ponderee():
    """Normalisation PAR BANDE puis poids : la bande basse pèse moitié moins."""
    extractor = HistogramAppearanceExtractor(
        horizontal_crop_ratio=0.0, bands=2, band_weights=(1.0, 0.5)
    )
    frame = np.zeros((120, 100, 3), dtype=np.uint8)
    frame[:60] = (0, 0, 255)
    frame[60:] = (0, 255, 0)

    vector = extractor.describe(frame, [0, 0, 100, 120])
    assert vector is not None
    top, bottom = vector[: 24 * 8], vector[24 * 8:]
    # Chaque bande étant normalisée séparément, leur énergie est comparable avant
    # pondération ; la bande basse pondérée 0.5 doit donc porter moitié moins.
    assert float(np.linalg.norm(bottom)) == pytest.approx(
        0.5 * float(np.linalg.norm(top)), rel=1e-3
    )
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_blend_repart_du_descripteur_courant_si_la_dimension_change():
    """Le descripteur change de dimension (B.4) : ``_blend`` ne doit pas casser."""
    manager = IdentityManager()
    old = np.ones(10, dtype=np.float32)
    old /= float(np.linalg.norm(old))
    manager.assign([observation(7, feature=old)], FRAME, 0.0, 1)
    assert manager.records[1].feature.shape == (10,)

    new = np.ones(3 * 24 * 8, dtype=np.float32)
    new /= float(np.linalg.norm(new))
    manager.assign([observation(7, feature=new)], FRAME, 0.1, 2)
    assert manager.records[1].feature.shape == (3 * 24 * 8,)
    assert float(np.linalg.norm(manager.records[1].feature)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# B.5 — boîte multi-personnes : signal par nombre de têtes, conséquence occupation
# ---------------------------------------------------------------------------
def test_signal_par_tetes_detecte_une_boite_fusionnee():
    boxes = [[0.0, 0.0, 200.0, 200.0]]
    heads = [(40.0, 50.0), (160.0, 60.0)]  # séparation 120 > 0.4 × 200
    flags = detect_multi_person_by_head_count(boxes, heads, 0.4)
    assert 0 in flags
    assert flags[0]["heads"] == 2


def test_signal_par_tetes_ignore_deux_tetes_trop_proches():
    boxes = [[0.0, 0.0, 200.0, 200.0]]
    heads = [(100.0, 50.0), (110.0, 60.0)]  # séparation 10 < 0.4 × 200
    assert detect_multi_person_by_head_count(boxes, heads, 0.4) == {}


def test_signal_par_tetes_exige_deux_points_fiables():
    boxes = [[0.0, 0.0, 200.0, 200.0]]
    assert detect_multi_person_by_head_count(boxes, [(50.0, 50.0), None], 0.4) == {}


def _head_assist_manager(make_config, sink, enabled: bool) -> OccupancyManager:
    config = make_config(
        {
            "line": {"inside_side": "negative"},
            "display": {"enabled": False},
            "presence": {
                "head_assist": {
                    "enabled": enabled,
                    "min_keypoint_confidence": 0.5,
                    "max_presence_extension_seconds": 10.0,
                }
            },
        }
    )
    manager = OccupancyManager(config, event_sink=sink, line=make_test_line())
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


def test_boite_fusionnee_detectee_des_la_premiere_frame_par_les_tetes(make_config, sink):
    manager = _head_assist_manager(make_config, sink, enabled=True)
    box = np.array([100.0, 200.0, 300.0, 450.0])
    detections = [
        Detection(1, box, 0.9, head_point=(150.0, 250.0), head_confidence=0.9),
        Detection(2, box, 0.9, head_point=(250.0, 250.0), head_confidence=0.9),
    ]
    manager.process_frame(detections, FRAME, 0.1, 1)

    events = sink.of_type("POSSIBLE_MULTI_PERSON_BOX")
    assert events, "deux têtes dans une même boîte ⇒ fusion dès la première frame"
    assert {event["signal"] for event in events} == {"head_count"}
    assert any(int(event["head_count"]) == 2 for event in events)


def test_signal_par_tetes_inactif_quand_l_assistance_est_desactivee(make_config, sink):
    """Assistance désactivée : le signal par têtes n'existe pas, pas de régression."""
    manager = _head_assist_manager(make_config, sink, enabled=False)
    box = np.array([100.0, 200.0, 300.0, 450.0])
    detections = [
        Detection(1, box, 0.9, head_point=(150.0, 250.0), head_confidence=0.9),
        Detection(2, box, 0.9, head_point=(250.0, 250.0), head_confidence=0.9),
    ]
    manager.process_frame(detections, FRAME, 0.1, 1)
    assert sink.count("POSSIBLE_MULTI_PERSON_BOX") == 0


def test_identite_absorbee_n_est_pas_purgee_en_silence(make_config, sink):
    """B.5.a : une identité perdue près d'une boîte fusionnée reste dans le bilan."""
    manager = _head_assist_manager(make_config, sink, enabled=True)
    manager._warmup_done = True
    manager._frame_size = (600, 600)
    manager.scale.observe_many([np.array([100.0, 100.0, 200.0, 300.0])])

    lost = PersonTrack(
        person_id=1,
        state=TrackState.OCCULTEE,
        technical_track_id=5,
        anchor=(150.0, 450.0),
        inside_occupancy=True,
        last_seen_frame=5,
        last_seen_s=0.5,
    )
    near = PersonTrack(
        person_id=2,
        state=TrackState.PRESENTE,
        technical_track_id=6,
        anchor=(160.0, 450.0),
        inside_occupancy=True,
    )
    manager.tracks = {1: lost, 2: near}
    # La personne 1 avait été comptée avant l'absorption (registre opérationnel).
    manager.initial_occupancy = 1
    manager._multi_person_frame = {2: {"signal": "head_count"}}

    manager._handle_missing(lost, manager.budget, 1.0, 10)

    events = sink.of_type("INCONSISTENT_STATE")
    assert any(
        event["kind"] == "identity_absorbed_by_multi_person_box" for event in events
    )
    assert lost.inside_occupancy is True, "l'identité reste comptée en borne haute"
    assert lost.multi_person_absorption_reported is True
    assert manager.occupancy_operational >= 1


# ---------------------------------------------------------------------------
# Chemin réel des points-clés : tenseurs (N, 17, 2) et (N, 17)
# ---------------------------------------------------------------------------
def test_point_tete_extrait_de_tenseurs_keypoints_reels():
    """Les 4 tests d'origine injectaient ``head_point`` à la main : ce test part
    d'un tenseur de la forme réellement produite par le modèle pose."""
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    conf = np.zeros((1, 17), dtype=np.float32)
    xy[0, 0] = (300.0, 120.0)
    conf[0, 0] = 0.9  # nose
    xy[0, 3] = (280.0, 125.0)
    conf[0, 3] = 0.8  # left_ear
    xy[0, 4] = (320.0, 125.0)
    conf[0, 4] = 0.85  # right_ear

    head_point, head_confidence = extract_head_point(
        xy[0],
        conf[0],
        ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
        0.5,
    )
    assert head_point == pytest.approx((300.0, 123.333,), abs=1e-2)
    assert head_confidence == pytest.approx(0.8)


def test_chemin_complet_tenseur_vers_maintien_de_presence(make_config, sink):
    """Tenseur ``keypoints`` → ``Detection`` → ``PRESENCE_MAINTAINED_BY_HEAD``."""
    config = make_config(
        {
            "line": {"inside_side": "negative"},
            "display": {"enabled": False},
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
                    "min_keypoint_confidence": 0.5,
                    "max_presence_extension_seconds": 5.0,
                }
            },
        }
    )
    manager = OccupancyManager(config, event_sink=sink, line=make_test_line())
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))

    xy = np.zeros((1, 17, 2), dtype=np.float32)
    conf = np.zeros((1, 17), dtype=np.float32)
    xy[0, 0] = (300.0, 300.0)
    conf[0, 0] = 0.95
    head_point, head_confidence = extract_head_point(
        xy[0], conf[0], config.presence.head_assist.keypoints_used, 0.5
    )
    assert head_point is not None

    standing = np.array([260.0, 250.0, 340.0, 450.0])
    for index in range(1, 10):
        manager.process_frame([Detection(1, standing, 0.9)], FRAME, index * 0.1, index)

    truncated = np.array([0.0, 250.0, 80.0, 450.0])
    for index in range(10, 14):
        manager.process_frame(
            [Detection(1, truncated, 0.9, head_point=head_point, head_confidence=head_confidence)],
            FRAME,
            index * 0.1,
            index,
        )

    maintained = sink.of_type("PRESENCE_MAINTAINED_BY_HEAD")
    assert maintained, "un point tête extrait d'un tenseur réel doit maintenir la présence"
    assert maintained[0]["feet_anchor_status"] == "edge"
