"""Niveau 1 — Gestionnaire d'identités : filtre spatio-temporel, seuil, marge de
sécurité, décision ambiguë, purge explicite (spec 3.3, 3.4, 4.5).

Les descripteurs sont injectés explicitement via ``Observation.feature`` : ces
tests ne dépendent ni de torch, ni d'images réelles, et restent déterministes.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import ExternalReidConfig, LongTermReidConfig
from events import ListSink
from identity_manager import (
    DeepAppearanceExtractor,
    HistogramAppearanceExtractor,
    IdentityManager,
    Observation,
)

BOX = [80.0, 100.0, 120.0, 300.0]  # hauteur 200, ancre (100, 300)
FRAME = np.zeros((600, 600, 3), dtype=np.uint8)


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


def observation(track_id: int, anchor=(100.0, 300.0), box=BOX, confidence=0.9, feature=None):
    box = np.asarray(box, dtype=float)
    return Observation(
        technical_track_id=track_id,
        bbox=box,
        confidence=confidence,
        anchor=tuple(anchor),
        bbox_height=float(box[3] - box[1]),
        feature=None if feature is None else np.asarray(feature, dtype=np.float32),
    )


@pytest.fixture
def sink():
    return ListSink("reid")


@pytest.fixture
def manager(sink):
    return IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True, similarity_threshold=0.9, safety_margin=0.05,
            v_max_ratio=1.5, spatial_margin_ratio=0.3,
        ),
        grace_period_seconds=1.0,
        event_sink=sink,
    )


# ---------------------------------------------------------------------------
# Création et verrouillage
# ---------------------------------------------------------------------------
def test_new_track_creates_a_person(manager, sink):
    assignments = manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    assert assignments[0].person_id == 1
    assert assignments[0].matched is False
    assert manager.person_of(7) == 1
    assert sink.count("REID_NEW") == 1
    assert sink.events[-1]["reason"] == "no_plausible_candidate"


def test_existing_technical_track_is_locked(manager, sink):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    sink.events.clear()
    assignments = manager.assign([observation(7, feature=unit(0.9, 0.1, 0))], FRAME, 0.1, 2)
    assert assignments[0].person_id == 1
    assert assignments[0].reason == "technical_track_locked"
    assert sink.count("REID_MATCH") == 0
    assert sink.count("REID_NEW") == 0


def test_unobserved_identity_moves_to_gallery(manager):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    assert manager.live_count == 0
    assert manager.gallery_size == 1


# ---------------------------------------------------------------------------
# Filtre spatio-temporel (spec 3.3 étape 1)
# ---------------------------------------------------------------------------
def test_spatiotemporal_allowance_formula(manager):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    # La dernière observation date de t=0.0 : à t=0.2, dt = 0.2 s.
    # hauteur 200, v_max 1.5, marge 0.3 -> 1.5*200*0.2 + 0.3*200 = 120 px tolérés.
    allowed = 1.5 * 200.0 * 0.2 + 0.3 * 200.0
    assert allowed == pytest.approx(120.0)

    near = manager.assign(
        [observation(8, anchor=(100.0 + allowed, 300.0), feature=unit(1, 0, 0.05))],
        FRAME, 0.2, 3,
    )
    assert near[0].person_id == 1
    assert near[0].matched is True
    assert near[0].allowed_distance == pytest.approx(allowed)
    assert near[0].distance == pytest.approx(allowed)


def test_too_far_reappearance_is_rejected(manager, sink):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    sink.events.clear()
    far = manager.assign(
        [observation(8, anchor=(400.0, 300.0), feature=unit(1, 0, 0))], FRAME, 0.2, 3
    )
    assert far[0].person_id == 2
    assert far[0].reason == "no_plausible_candidate"
    assert sink.count("REID_MATCH") == 0


def test_gallery_never_closes_before_the_fsm_grace(manager):
    """La fenêtre de galerie est strictement plus large que la grâce FSM.

    Sans cette marge, la galerie libérerait une identité avant que la machine à
    états ait pu la purger : l'événement ``PURGE`` et le passage en occupation
    incertaine disparaîtraient silencieusement.
    """
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    # 1.5 s > grâce (1.0 s) mais < grâce + rétention (2.0 s) : encore joignable.
    late = manager.assign(
        [observation(8, anchor=(100.0, 300.0), feature=unit(1, 0, 0))], FRAME, 1.5, 2
    )
    assert late[0].person_id == 1
    assert manager.release_expired(1.5) == []


def test_reappearance_after_gallery_release_is_rejected(manager):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)          # la piste disparaît
    assert manager.release_expired(1.0) == []   # encore dans la grâce
    assert manager.release_expired(2.5) == [1]  # libérée après deux fenêtres
    late = manager.assign(
        [observation(8, anchor=(100.0, 300.0), feature=unit(1, 0, 0))], FRAME, 2.6, 2
    )
    assert late[0].person_id == 2
    assert late[0].reason == "no_plausible_candidate"


def test_purged_identity_stays_reachable_for_one_extra_window(manager, sink):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.mark_purged(
        1, 0.1, 2, reason="grace_expired", last_state="OCCULTEE", occupancy_impact="uncertain"
    )
    # 0.9 s après la purge : encore dans la fenêtre de rétention.
    recovered = manager.assign(
        [observation(8, anchor=(100.0, 300.0), feature=unit(1, 0, 0))], FRAME, 1.0, 3
    )
    assert recovered[0].person_id == 1
    assert recovered[0].matched is True
    assert sink.count("REID_MATCH") == 1


# ---------------------------------------------------------------------------
# Seuil et marge de sécurité (spec 3.3 étapes 4 et 5)
# ---------------------------------------------------------------------------
def test_below_threshold_creates_a_new_person(manager, sink):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    sink.events.clear()
    result = manager.assign(
        [observation(8, anchor=(100.0, 300.0), feature=unit(0.5, 0.866, 0))], FRAME, 0.2, 3
    )
    assert result[0].reason == "below_similarity_threshold"
    assert sink.count("REID_NEW") == 1
    assert sink.events[-1]["best_similarity"] == pytest.approx(0.5, abs=1e-3)


def test_safety_margin_triggers_ambiguous_event(manager, sink):
    """Deux candidats quasi équivalents : jamais de fusion arbitraire."""
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(9, anchor=(300.0, 300.0), feature=unit(0.999, 0.02, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    sink.events.clear()

    # Descripteur très proche des deux candidats -> marge insuffisante.
    result = manager.assign(
        [observation(11, anchor=(200.0, 300.0), feature=unit(1.0, 0.01, 0))], FRAME, 0.2, 3
    )
    assert sink.count("REID_AMBIGUOUS") == 1
    event = sink.of_type("REID_AMBIGUOUS")[0]
    assert event["best_similarity"] - event["runner_up_similarity"] < event["safety_margin"]
    assert len(event["candidates"]) == 2
    # Politique par défaut : identité provisoire, marquée incertaine.
    assert result[0].provisional is True
    assert result[0].reason == "reid_ambiguous_provisional"
    assert manager.records[result[0].person_id].provisional is True


def test_ambiguous_defer_policy_leaves_track_unassigned(sink):
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True, similarity_threshold=0.9, safety_margin=0.05,
            ambiguous_policy="defer",
        ),
        grace_period_seconds=1.0,
        event_sink=sink,
    )
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(9, anchor=(300.0, 300.0), feature=unit(0.999, 0.02, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    sink.events.clear()
    result = manager.assign(
        [observation(11, anchor=(200.0, 300.0), feature=unit(1.0, 0.01, 0))], FRAME, 0.2, 3
    )
    assert result[0].person_id == 0
    assert result[0].reason == "reid_ambiguous_deferred"
    assert sink.count("REID_AMBIGUOUS") == 1


def test_safety_margin_zero_accepts_best_candidate(sink):
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True, similarity_threshold=0.9, safety_margin=0.0
        ),
        grace_period_seconds=1.0,
        event_sink=sink,
    )
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(9, anchor=(300.0, 300.0), feature=unit(0.999, 0.02, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    sink.events.clear()
    result = manager.assign(
        [observation(11, anchor=(200.0, 300.0), feature=unit(1.0, 0.01, 0))], FRAME, 0.2, 3
    )
    assert sink.count("REID_AMBIGUOUS") == 0
    assert result[0].matched is True


def test_long_term_reid_disabled_creates_new_person(sink):
    manager = IdentityManager(
        long_term=LongTermReidConfig(enabled=False), grace_period_seconds=1.0, event_sink=sink
    )
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    result = manager.assign(
        [observation(8, anchor=(100.0, 300.0), feature=unit(1, 0, 0))], FRAME, 0.2, 3
    )
    assert result[0].person_id == 2
    assert result[0].reason == "long_term_reid_disabled"


def test_missing_descriptor_creates_new_person(sink):
    from conftest import StubAppearance

    manager = IdentityManager(
        grace_period_seconds=1.0, event_sink=sink, appearance=StubAppearance(default=None)
    )
    manager.assign([observation(7)], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    result = manager.assign([observation(8, anchor=(100.0, 300.0))], FRAME, 0.2, 3)
    assert result[0].reason == "no_descriptor"


# ---------------------------------------------------------------------------
# Garde-fou de double réassociation (spec 4.5)
# ---------------------------------------------------------------------------
def test_duplicate_claim_is_reported_and_resolved(manager, sink):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    sink.events.clear()
    # Deux pistes techniques distinctes pour le même person_id sur la même frame.
    manager.technical_to_person[8] = 1
    assignments = manager.assign(
        [observation(7, feature=unit(1, 0, 0)), observation(8, feature=unit(1, 0, 0))],
        FRAME, 0.1, 2,
    )
    assert sink.count("INCONSISTENT_STATE") == 1
    assert sink.of_type("INCONSISTENT_STATE")[0]["kind"] == "duplicate_person_claim"
    assert len({assignment.person_id for assignment in assignments}) == 2


# ---------------------------------------------------------------------------
# Purge explicite (spec 3.4)
# ---------------------------------------------------------------------------
def test_purge_is_explicit_and_then_released(manager, sink):
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    sink.events.clear()
    manager.mark_purged(
        1, 0.5, 5, reason="grace_expired", last_state="OCCULTEE",
        occupancy_impact="uncertain", grace_period_frames=10,
    )
    event = sink.of_type("PURGE")[0]
    assert event["person_id"] == 1
    assert event["reason"] == "grace_expired"
    assert event["last_state"] == "OCCULTEE"
    assert event["occupancy_impact"] == "uncertain"
    assert event["last_position"] == [100.0, 300.0]
    assert event["grace_period_frames"] == 10
    # La mémoire est libérée après la fenêtre de rétention, pas avant.
    assert manager.release_expired(1.0) == []
    assert manager.release_expired(2.0) == [1]
    assert manager.person_of(7) is None


# ---------------------------------------------------------------------------
# Extracteurs d'apparence
# ---------------------------------------------------------------------------
def test_histogram_extractor_is_deterministic_and_normalized():
    extractor = HistogramAppearanceExtractor()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:150, 50:150] = (10, 120, 200)
    box = [50, 50, 150, 150]
    first = extractor.describe(frame, box)
    second = extractor.describe(frame, box)
    assert first is not None and second is not None
    assert np.allclose(first, second)
    assert float(np.linalg.norm(first)) == pytest.approx(1.0)


def test_histogram_extractor_refuses_tiny_crops():
    extractor = HistogramAppearanceExtractor()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert extractor.describe(frame, [0, 0, 4, 4]) is None


def test_histogram_extractor_erodes_the_crop_before_describing():
    """Les bords de la boîte (fond, voisins) sont écartés du descripteur.

    C'est le point clé du lot B.4 : sur un crop complet, la couleur des chaises,
    des tables et du mur dominait l'histogramme, et deux personnes de la même
    rangée atteignaient une similarité > 0,9.
    """
    solid_blue = np.zeros((200, 200, 3), dtype=np.uint8)
    solid_blue[:] = (255, 0, 0)  # BGR : bleu franc
    with_red_borders = solid_blue.copy()
    with_red_borders[:, :30] = (0, 0, 255)    # bord gauche « fond »
    with_red_borders[:, 170:] = (0, 0, 255)   # bord droit « voisin »

    eroding = HistogramAppearanceExtractor(horizontal_crop_ratio=0.15, bands=3)
    assert np.allclose(
        eroding.describe(solid_blue, [0, 0, 200, 200]),
        eroding.describe(with_red_borders, [0, 0, 200, 200]),
    ), "15 % de chaque côté : les bordures rouges doivent être hors du crop"

    # Sans rognage, les bordures entrent dans le descripteur : la différence est
    # bien celle qu'on vient d'écarter, pas un artefact du calcul.
    full = HistogramAppearanceExtractor(horizontal_crop_ratio=0.0, bands=3)
    assert not np.allclose(
        full.describe(solid_blue, [0, 0, 200, 200]),
        full.describe(with_red_borders, [0, 0, 200, 200]),
    )


def test_histogram_extractor_is_banded_and_weights_the_low_band():
    """Descripteur par bandes : dimension et pondération configurables."""
    frame = np.zeros((150, 120, 3), dtype=np.uint8)
    frame[:] = (255, 0, 0)  # couleur uniforme : les 3 bandes sont identiques
    bands, bins_hue, bins_saturation = 3, 24, 8
    band_size = bins_hue * bins_saturation

    extractor = HistogramAppearanceExtractor(
        bands=bands, band_weights=(1.0, 1.0, 0.0),
        bins_hue=bins_hue, bins_saturation=bins_saturation,
    )
    descriptor = extractor.describe(frame, [0, 0, 120, 150])

    assert descriptor.shape == (bands * band_size,), "une tranche par bande"
    # Bandes identiques et poids égaux : les deux premières tranches coïncident.
    assert np.allclose(descriptor[:band_size], descriptor[band_size:2 * band_size])
    # La bande basse est pondérée à 0 (elle est la plus occultée par une table).
    assert np.allclose(descriptor[2 * band_size:], 0.0)
    # Le contrat reste « descripteur L2-normalisé ».
    assert float(np.linalg.norm(descriptor)) == pytest.approx(1.0)


def test_histogram_extractor_refuses_a_crop_eroded_under_the_minimum():
    extractor = HistogramAppearanceExtractor(horizontal_crop_ratio=0.45, min_width=40)
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    # 100 px de large rognés de 45 % de chaque côté -> 10 px < min_width 40.
    assert extractor.describe(frame, [0, 0, 100, 200]) is None


def test_histogram_extractor_is_built_from_the_descriptor_config():
    """Les paramètres du descripteur viennent de la configuration, jamais du code."""
    from config import DescriptorConfig

    extractor = HistogramAppearanceExtractor.from_config(
        DescriptorConfig(
            horizontal_crop_ratio=0.2, bands=2, band_weights=(1.0, 0.5),
            bins_hue=8, bins_saturation=4,
        )
    )
    assert extractor.horizontal_crop_ratio == pytest.approx(0.2)
    assert extractor.bands == 2
    assert extractor.band_weights == (1.0, 0.5)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert extractor.describe(frame, [0, 0, 100, 100]).shape == (2 * 8 * 4,)


def test_deep_extractor_refuses_tiny_crop_without_loading_weights():
    extractor = DeepAppearanceExtractor(ExternalReidConfig(enabled=True))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert extractor.describe(frame, [0, 0, 4, 4]) is None
    assert extractor._model is None, "aucun poids ne doit être chargé pour un crop invalide"


def test_external_reid_flag_selects_deep_extractor(monkeypatch):
    from config import ExternalReidConfig as Config

    # Lot 3.a : le descripteur profond est désormais chargé au démarrage, et un
    # modèle absent déclenche le repli sur l'histogramme. Le poids ONNX n'étant
    # pas versionné, on neutralise le chargement pour que ce test vérifie
    # toujours la sélection par le drapeau, indépendamment de l'environnement ;
    # le repli est éprouvé dans tests/test_reid_deep_lot3a.py.
    monkeypatch.setattr(DeepAppearanceExtractor, "load", lambda self: None)
    manager = IdentityManager(external_reid=Config(enabled=True))
    assert isinstance(manager.appearance, DeepAppearanceExtractor)
    plain = IdentityManager(external_reid=Config(enabled=False))
    assert isinstance(plain.appearance, HistogramAppearanceExtractor)


def test_low_confidence_descriptor_does_not_pollute_gallery(manager):
    manager.assign([observation(7, confidence=0.9, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    original = manager.records[1].feature.copy()
    manager.assign([observation(7, confidence=0.2, feature=unit(0, 1, 0))], FRAME, 0.1, 2)
    assert np.allclose(manager.records[1].feature, original)


def test_descriptor_is_blended_and_renormalized(manager):
    """Le descripteur de galerie suit l'apparence courante en moyenne glissante.

    Le changement d'apparence est volontairement **modéré** (similarité ≈ 0,96,
    au-dessus de ``appearance_continuity.similarity_threshold`` = 0,5) : un saut
    franc d'une frame à l'autre est une discontinuité, qui **gèle** désormais le
    descripteur (lot B.3) — c'est le test dédié suivant qui couvre ce cas.
    """
    manager.assign([observation(7, confidence=0.9, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(7, confidence=0.9, feature=unit(1, 0.3, 0))], FRAME, 0.1, 2)
    feature = manager.records[1].feature
    assert float(np.linalg.norm(feature)) == pytest.approx(1.0)
    # Moyenne glissante à 85 % : l'historique domine, sans effacer la nouvelle vue.
    assert 0.0 < float(feature[1]) < float(feature[0])


def test_descriptor_is_frozen_while_appearance_is_discontinuous(manager, sink):
    """Après une discontinuité, ``record.feature`` reste inchangé pendant N frames.

    C'est la correction B.3 : sans ce gel, la référence dérivait à chaque frame
    vers l'apparence de l'AUTRE personne (1 − 0,85⁵ ≈ 56 % au bout de 5 frames),
    ce qui rendait ``_correct_cross_swaps`` aveugle au moment où le swap
    s'installait.
    """
    manager.assign([observation(7, confidence=0.9, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    frozen = manager.records[1].feature.copy()

    # Apparence franchement différente : similarité 0 -> discontinuité.
    freeze_frames = manager.freeze_gallery_frames
    assert freeze_frames >= 3
    for offset in range(1, freeze_frames + 1):
        manager.assign(
            [observation(7, confidence=0.9, feature=unit(0, 1, 0))],
            FRAME,
            0.1 * offset,
            1 + offset,
        )
        assert np.allclose(manager.records[1].feature, frozen), (
            f"le descripteur doit rester gelé à la frame {1 + offset}"
        )
    assert manager.records[1].feature.shape == frozen.shape
    # Le diagnostic reste émis, lui : le gel porte sur la référence, pas sur la mesure.
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") >= 1
    # La position et l'horodatage continuent d'être mis à jour normalement.
    assert manager.records[1].last_seen_s == pytest.approx(0.1 * freeze_frames)


def test_descriptor_resumes_blending_once_the_freeze_expires(manager):
    """Le gel est borné : au-delà de la péremption, la référence suit l'apparence."""
    manager.assign([observation(7, confidence=0.9, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    frozen = manager.records[1].feature.copy()

    # Frame 2 : un swap se manifeste par UN saut d'apparence, pas par une suite de
    # sauts. C'est ce saut unique qui arme le gel.
    manager.assign([observation(7, confidence=0.9, feature=unit(0, 1, 0))], FRAME, 0.2, 2)
    assert np.allclose(manager.records[1].feature, frozen)
    deadline = manager.records[1].appearance_suspect_until_frame
    assert deadline is not None

    # Jusqu'à la péremption incluse, la référence reste gelée même si le saut n'a
    # eu lieu qu'une fois (l'apparence est ensuite stable d'une frame à l'autre).
    for frame_index in range(3, deadline + 1):
        manager.assign(
            [observation(7, confidence=0.9, feature=unit(0, 1, 0))],
            FRAME, 0.1 * frame_index, frame_index,
        )
        assert np.allclose(manager.records[1].feature, frozen)

    # Au-delà : le mélange reprend, la référence suit de nouveau l'apparence.
    manager.assign(
        [observation(7, confidence=0.9, feature=unit(0, 1, 0))],
        FRAME, 0.1 * (deadline + 1), deadline + 1,
    )
    assert not np.allclose(manager.records[1].feature, frozen)
    assert float(manager.records[1].feature[1]) > 0.0


def test_cosine_similarity_bounds():
    assert IdentityManager.cosine_similarity(unit(1, 0, 0), unit(1, 0, 0)) == pytest.approx(1.0)
    assert IdentityManager.cosine_similarity(unit(1, 0, 0), unit(0, 1, 0)) == pytest.approx(0.0)
    assert IdentityManager.cosine_similarity(unit(1, 0, 0), -unit(1, 0, 0)) == pytest.approx(-1.0)
    assert IdentityManager.cosine_similarity(np.zeros(3), unit(1, 0, 0)) == 0.0
