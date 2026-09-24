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


def test_deep_extractor_refuses_tiny_crop_without_loading_weights():
    extractor = DeepAppearanceExtractor(ExternalReidConfig(enabled=True))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert extractor.describe(frame, [0, 0, 4, 4]) is None
    assert extractor._model is None, "aucun poids ne doit être chargé pour un crop invalide"


def test_external_reid_flag_selects_deep_extractor():
    from config import ExternalReidConfig as Config

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
    manager.assign([observation(7, confidence=0.9, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(7, confidence=0.9, feature=unit(0, 1, 0))], FRAME, 0.1, 2)
    feature = manager.records[1].feature
    assert float(np.linalg.norm(feature)) == pytest.approx(1.0)
    # Moyenne glissante à 85 % : l'historique domine, sans effacer la nouvelle vue.
    assert 0.0 < float(feature[1]) < float(feature[0])


def test_cosine_similarity_bounds():
    assert IdentityManager.cosine_similarity(unit(1, 0, 0), unit(1, 0, 0)) == pytest.approx(1.0)
    assert IdentityManager.cosine_similarity(unit(1, 0, 0), unit(0, 1, 0)) == pytest.approx(0.0)
    assert IdentityManager.cosine_similarity(unit(1, 0, 0), -unit(1, 0, 0)) == pytest.approx(-1.0)
    assert IdentityManager.cosine_similarity(np.zeros(3), unit(1, 0, 0)) == 0.0


# ---------------------------------------------------------------------------
# Priorité 2 : TECHNICAL_ID_REPURPOSED (rejeu du cas frame 266)
# ---------------------------------------------------------------------------
def test_technical_id_repurposed_releases_stale_mapping(manager, sink):
    """Cas frame 266 : BoT-SORT réutilise une piste technique pour une personne
    géométriquement incompatible avec le record verrouillé.
    L'ancien mapping est libéré, l'identité légitime n'est pas perdue.
    """
    # 1. Piste 5 initialement associée à la Personne 1 en bas de cadre (y=550)
    manager.assign(
        [observation(5, anchor=(100.0, 550.0), box=[80.0, 350.0, 120.0, 550.0], feature=unit(1, 0, 0))],
        FRAME, 0.0, 1,
    )
    assert manager.person_of(5) == 1
    assert sink.count("REID_NEW") == 1

    # 2. Perte temporaire de piste : Personne 1 réapparaît sous la piste 8 en y=550
    #    (Personne 1 a maintenant pour alias 5 et 8)
    manager.assign([], FRAME, 0.5, 2)
    manager.assign(
        [observation(8, anchor=(100.0, 550.0), box=[80.0, 350.0, 120.0, 550.0], feature=unit(1, 0, 0))],
        FRAME, 1.0, 3,
    )
    assert manager.person_of(8) == 1
    sink.events.clear()

    # 3. Frame 266 (t=2.0) :
    #    - Piste 8 est active en bas (y=550) et réclame légitimement Personne 1.
    #    - Piste 5 réapparaît soudainement en haut (y=310, distance=240 px > 60 px tolérés).
    assignments = manager.assign(
        [
            observation(8, anchor=(100.0, 550.0), box=[80.0, 350.0, 120.0, 550.0], feature=unit(1, 0, 0)),
            observation(5, anchor=(100.0, 310.0), box=[80.0, 110.0, 120.0, 310.0], feature=unit(0, 1, 0)),
        ],
        FRAME, 2.0, 266,
    )

    # L'identité légitime (Personne 1) reste assignée à la piste 8
    assign_8 = next(a for a in assignments if a.technical_track_id == 8)
    assert assign_8.person_id == 1
    assert assign_8.reason == "technical_track_locked"

    # La piste 5 a été détectée comme réutilisée (repurposed)
    assign_5 = next(a for a in assignments if a.technical_track_id == 5)
    assert assign_5.person_id == 2  # Nouvelle identité créée pour la nouvelle personne
    assert manager.person_of(5) == 2
    assert manager.person_of(8) == 1

    # Événement TECHNICAL_ID_REPURPOSED émis distinctement
    assert sink.count("TECHNICAL_ID_REPURPOSED") == 1
    event = sink.of_type("TECHNICAL_ID_REPURPOSED")[0]
    assert event["technical_track_id"] == 5
    assert event["previous_person_id"] == 1
    assert event["new_person_id"] == 2
    assert event["distance"] == pytest.approx(240.0)
    assert event["allowed_distance"] == pytest.approx(60.0)  # 0.3 * 200
    assert manager.technical_id_repurposed == 1

    # Pas d'INCONSISTENT_STATE duplicate_claim
    assert sink.count("INCONSISTENT_STATE") == 0


# ---------------------------------------------------------------------------
# Priorité 3 : PROVISIONAL_IDENTITY_MERGED (recyclage des provisoires)
# ---------------------------------------------------------------------------
def test_provisional_identity_merged_after_confirmation_streak(sink):
    """Casse la boucle de rétroaction : une identité provisoire est fusionnée
    après provisional_merge_confirmation_frames observations cohérentes avec
    une identité de galerie non provisoire.
    """
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True,
            similarity_threshold=0.6,
            safety_margin=0.05,
            v_max_ratio=1.5,
            spatial_margin_ratio=0.3,
            provisional_merge_confirmation_frames=3,
        ),
        grace_period_seconds=5.0,
        event_sink=sink,
    )

    # 1. Deux personnes établies créées (P1 et P2)
    manager.assign(
        [
            observation(1, anchor=(100.0, 300.0), feature=unit(1, 0, 0)),
            observation(2, anchor=(300.0, 300.0), feature=unit(0.6, 0.8, 0)),
        ],
        FRAME, 0.0, 1,
    )
    assert manager.person_of(1) == 1
    assert manager.person_of(2) == 2

    # 2. P1 et P2 disparaissent en galerie
    manager.assign([], FRAME, 0.1, 2)
    sink.events.clear()

    # 3. Piste 10 apparaît : vecteur équidistant de P1 et P2 -> ambiguïté
    mid = unit(0.8, 0.4, 0)
    res = manager.assign(
        [observation(10, anchor=(200.0, 300.0), feature=mid)],
        FRAME, 0.2, 3,
    )
    assert sink.count("REID_AMBIGUOUS") == 1
    assert res[0].person_id == 3
    assert res[0].provisional is True
    assert 3 in manager.provisional_ids

    sink.events.clear()

    # 4. Observation cohérente 1 avec P1 (streak = 1 < 3)
    res1 = manager.assign(
        [observation(10, anchor=(105.0, 300.0), feature=unit(1.0, 0.0, 0))],
        FRAME, 0.3, 4,
    )
    assert res1[0].person_id == 3
    assert res1[0].provisional is True
    assert sink.count("PROVISIONAL_IDENTITY_MERGED") == 0

    # 5. Observation cohérente 2 avec P1 (streak = 2 < 3)
    res2 = manager.assign(
        [observation(10, anchor=(110.0, 300.0), feature=unit(1.0, 0.0, 0))],
        FRAME, 0.4, 5,
    )
    assert res2[0].person_id == 3
    assert res2[0].provisional is True
    assert sink.count("PROVISIONAL_IDENTITY_MERGED") == 0

    # 6. Observation cohérente 3 avec P1 (streak = 3 >= 3) -> FUSION !
    res3 = manager.assign(
        [observation(10, anchor=(115.0, 300.0), feature=unit(1.0, 0.0, 0))],
        FRAME, 0.5, 6,
    )
    assert res3[0].person_id == 1  # Réattribué à P1 !
    assert res3[0].provisional is False
    assert res3[0].reason == "provisional_identity_merged"
    assert sink.count("PROVISIONAL_IDENTITY_MERGED") == 1
    event = sink.of_type("PROVISIONAL_IDENTITY_MERGED")[0]
    assert event["provisional_person_id"] == 3
    assert event["target_person_id"] == 1
    assert event["consistent_observations"] == 3
    assert 10 in event["technical_track_ids"]

    # P3 est retiré des records et des provisoires
    assert 3 not in manager.records
    assert 3 not in manager.provisional_ids
    assert manager.person_of(10) == 1
    assert manager.records[1].live is True
    assert 10 in manager.records[1].aliases
    assert manager.provisional_identities_merged == 1

    # 7. Les observations futures restent sur P1 sans aucune ambiguïté
    sink.events.clear()
    res4 = manager.assign(
        [observation(10, anchor=(120.0, 300.0), feature=unit(1.0, 0.0, 0))],
        FRAME, 0.6, 7,
    )
    assert res4[0].person_id == 1
    assert res4[0].reason == "technical_track_locked"
    assert sink.count("REID_AMBIGUOUS") == 0


def test_provisional_identity_streak_reset_on_inconsistency(sink):
    """Une rupture de cohérence (ex: descripteur ambigu ou incohérent) réinitialise
    le compteur consécutif : aucune fusion hâtive sur signal discontinu.
    """
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True,
            similarity_threshold=0.6,
            safety_margin=0.05,
            v_max_ratio=1.5,
            spatial_margin_ratio=0.3,
            provisional_merge_confirmation_frames=3,
        ),
        grace_period_seconds=5.0,
        event_sink=sink,
    )
    # P1 et P2 en galerie
    manager.assign([
        observation(1, anchor=(100.0, 300.0), feature=unit(1, 0, 0)),
        observation(2, anchor=(300.0, 300.0), feature=unit(0.6, 0.8, 0)),
    ], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)

    # Frame 3: ambiguïté -> P3 provisoire
    mid = unit(0.8, 0.4, 0)
    manager.assign([observation(10, anchor=(200.0, 300.0), feature=mid)], FRAME, 0.2, 3)
    assert 3 in manager.provisional_ids

    # Observation 1 : cohérente avec P1 (streak = 1)
    manager.assign([observation(10, anchor=(105.0, 300.0), feature=unit(1.0, 0.0, 0))], FRAME, 0.3, 4)
    assert manager._provisional_tracking.get(3) == (1, 1)

    # Observation 2 : cohérente avec P1 (streak = 2)
    manager.assign([observation(10, anchor=(110.0, 300.0), feature=unit(1.0, 0.0, 0))], FRAME, 0.4, 5)
    assert manager._provisional_tracking.get(3) == (1, 2)

    # Observation 3 : descripteur redevient ambigu (mid) -> streak réinitialisé à 0 !
    manager.assign([observation(10, anchor=(200.0, 300.0), feature=mid)], FRAME, 0.5, 6)
    assert 3 not in manager._provisional_tracking
    assert sink.count("PROVISIONAL_IDENTITY_MERGED") == 0
    assert 3 in manager.provisional_ids


def test_provisional_identity_unconfirmed_stays_provisional_until_purge(sink):
    """Une provisoire qui n'est jamais confirmée cohérente avec une identité
    existante avant sa purge reste provisoire jusqu'au bout.
    """
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True,
            similarity_threshold=0.6,
            safety_margin=0.05,
            v_max_ratio=1.5,
            spatial_margin_ratio=0.3,
            provisional_merge_confirmation_frames=3,
        ),
        grace_period_seconds=1.0,
        event_sink=sink,
    )
    # P1 et P2
    manager.assign([
        observation(1, anchor=(100.0, 300.0), feature=unit(1, 0, 0)),
        observation(2, anchor=(300.0, 300.0), feature=unit(0.6, 0.8, 0)),
    ], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)

    # Ambiguïté -> P3 provisoire
    mid = unit(0.8, 0.4, 0)
    manager.assign([observation(10, anchor=(200.0, 300.0), feature=mid)], FRAME, 0.2, 3)
    assert 3 in manager.provisional_ids

    # Piste 10 disparaît sans jamais avoir confirmé
    manager.assign([], FRAME, 0.3, 4)

    # P3 est purgé
    manager.mark_purged(3, 1.5, 15, reason="grace_expired", last_state="OCCULTEE", occupancy_impact="uncertain")
    assert 3 not in manager._provisional_tracking
    assert sink.count("PROVISIONAL_IDENTITY_MERGED") == 0
    # Reste provisoire dans ses attributs de record jusqu'à expiration
    assert manager.records[3].provisional is True
