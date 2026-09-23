"""Niveau 1-3 — Correction active des inversions d'identifiant (swap) sous occlusion.

Complément au diagnostic TRACK_ID_APPEARANCE_DISCONTINUITY :
réintroduit la correction active [RE-ID-LOCK] tout en assurant la traçabilité complète
via l'événement IDENTITY_SWAP_CORRECTED.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import (
    AppearanceContinuityConfig,
    LongTermReidConfig,
    SwapCorrectionConfig,
)
from events import ListSink
from identity_manager import IdentityManager, Observation

FRAME = np.zeros((600, 600, 3), dtype=np.uint8)


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


def observation(track_id: int, anchor=(100.0, 300.0), feature=None, box=None):
    box = np.asarray(box if box is not None else [80.0, 100.0, 120.0, 300.0], dtype=float)
    return Observation(
        technical_track_id=track_id,
        bbox=box,
        confidence=0.9,
        anchor=tuple(anchor),
        bbox_height=float(box[3] - box[1]),
        feature=None if feature is None else np.asarray(feature, dtype=np.float32),
    )


def build_manager(sink, *, swap_enabled=True, swap_margin=0.12, **overrides):
    swap_config = SwapCorrectionConfig(
        enabled=swap_enabled,
        margin=swap_margin,
    )
    continuity = AppearanceContinuityConfig(
        enabled=overrides.pop("continuity_enabled", True),
        similarity_threshold=overrides.pop("similarity_threshold", 0.5),
        max_gap_frames=overrides.pop("max_gap_frames", 2),
        min_interval_seconds=overrides.pop("min_interval_seconds", 0.0),
    )
    return IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True,
            similarity_threshold=0.4,
            safety_margin=0.0,
            v_max_ratio=1e6,
            spatial_margin_ratio=1e6,
        ),
        appearance_continuity=continuity,
        swap_correction=swap_config,
        grace_period_seconds=5.0,
        event_sink=sink,
        **overrides,
    )


@pytest.fixture
def sink():
    return ListSink("swap_correction")


def test_deux_pistes_dont_l_apparence_s_inverse_nettement_sont_corrigees(sink):
    """Deux pistes connues dont l'apparence s'inverse nettement (crossed > direct + margin) :
    technical_to_person est corrigé et IDENTITY_SWAP_CORRECTED est émis.
    """
    manager = build_manager(sink, swap_margin=0.12)
    desc_a = unit(1, 0, 0)
    desc_b = unit(0, 1, 0)

    # Frame 1 : affectation initiale normale
    manager.assign(
        [
            observation(7, anchor=(100.0, 300.0), feature=desc_a),
            observation(9, anchor=(200.0, 300.0), feature=desc_b),
        ],
        FRAME,
        0.0,
        1,
    )
    assert manager.person_of(7) == 1
    assert manager.person_of(9) == 2
    sink.events.clear()

    # Frame 2 : inversion nette des apparences (7 porte desc_b, 9 porte desc_a)
    assignments = manager.assign(
        [
            observation(7, anchor=(150.0, 300.0), feature=desc_b),
            observation(9, anchor=(160.0, 300.0), feature=desc_a),
        ],
        FRAME,
        0.1,
        2,
    )

    # Le mapping technical_to_person doit avoir été corrigé
    assert manager.person_of(7) == 2
    assert manager.person_of(9) == 1
    assert assignments[0].person_id == 2
    assert assignments[1].person_id == 1
    assert manager.swap_corrections_applied == 1

    # Un événement IDENTITY_SWAP_CORRECTED doit être émis avec les champs attendus
    events = sink.of_type("IDENTITY_SWAP_CORRECTED")
    assert len(events) == 1
    ev = events[0]
    assert ev["technical_track_id_a"] == 7
    assert ev["technical_track_id_b"] == 9
    assert ev["person_id_a_before"] == 1
    assert ev["person_id_a_after"] == 2
    assert ev["person_id_b_before"] == 2
    assert ev["person_id_b_after"] == 1
    assert ev["similarity_crossed"] > ev["similarity_direct"] + ev["margin_applied"]
    assert ev["margin_applied"] == pytest.approx(0.12)
    assert ev["reason"] == "cross_appearance_correction"


def test_la_correction_croisee_reste_possible_apres_cinq_frames_de_swap(sink):
    """Le descripteur de galerie ne dérive pas pendant un swap (lot B.3).

    ``_correct_cross_swaps`` compare l'apparence **courante** à ``record.feature``.
    Tant que le gel n'existait pas, ``_touch`` mélangeait à chaque frame le
    descripteur courant dans ``record.feature`` : au bout de 5 frames la
    référence contenait 1 − 0,85⁵ ≈ 56 % de l'apparence de l'AUTRE personne, et
    la comparaison directe/croisée devenait aveugle au moment précis où le swap
    s'installait. Ici, cinq frames d'apparence étrangère passent sans corriger
    (croisement nul) : si la référence dérivait, la correction de la frame 7
    serait impossible.
    """
    manager = build_manager(sink, swap_margin=0.12)
    desc_a = unit(1, 0, 0)
    desc_b = unit(0, 1, 0)

    manager.assign(
        [
            observation(7, anchor=(100.0, 300.0), feature=desc_a),
            observation(9, anchor=(200.0, 300.0), feature=desc_b),
        ],
        FRAME,
        0.0,
        1,
    )
    assert manager.person_of(7) == 1
    assert manager.person_of(9) == 2
    reference_1 = manager.records[1].feature.copy()
    reference_2 = manager.records[2].feature.copy()

    # Frames 2 à 6 : apparence orthogonale à a ET b, alternée pour que la
    # discontinuité soit ré-armée à chaque frame. Croisement nul, donc aucune
    # correction possible pendant ces cinq frames.
    distractors = [unit(0, 0, 1), unit(0, 0, -1)]
    for offset in range(5):
        manager.assign(
            [
                observation(7, anchor=(100.0, 300.0), feature=distractors[offset % 2]),
                observation(9, anchor=(200.0, 300.0), feature=distractors[offset % 2]),
            ],
            FRAME,
            0.1 * (offset + 2),
            offset + 2,
        )

    assert manager.swap_corrections_applied == 0
    assert manager.person_of(7) == 1, "la piste reste verrouillée sur son identité"
    assert manager.person_of(9) == 2
    # LES DEUX références sont intactes : c'est ce qui rend la correction possible.
    assert np.allclose(manager.records[1].feature, reference_1), (
        "la référence de galerie ne doit pas dériver pendant un swap"
    )
    assert np.allclose(manager.records[2].feature, reference_2)
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") >= 1

    # Frame 7 : le swap est enfin exploitable -> la correction doit encore aboutir.
    assignments = manager.assign(
        [
            observation(7, anchor=(100.0, 300.0), feature=desc_b),
            observation(9, anchor=(200.0, 300.0), feature=desc_a),
        ],
        FRAME,
        0.7,
        7,
    )

    assert manager.swap_corrections_applied == 1, (
        "après 5 frames de swap, la correction croisée doit rester possible"
    )
    assert manager.person_of(7) == 2
    assert manager.person_of(9) == 1
    assert assignments[0].person_id == 2
    assert assignments[1].person_id == 1
    assert sink.count("IDENTITY_SWAP_CORRECTED") == 1


def test_deux_personnes_qui_se_croisent_sans_echanger_d_apparence_ne_produisent_aucune_correction(sink):
    """Deux personnes qui conservent leur apparence ne déclenchent aucune correction."""
    manager = build_manager(sink)
    left = unit(1, 0, 0)
    right = unit(0, 1, 0)

    for step, (x7, x9) in enumerate([(100, 300), (180, 220), (220, 180), (300, 100)]):
        manager.assign(
            [
                observation(7, anchor=(x7, 300.0), feature=left),
                observation(9, anchor=(x9, 300.0), feature=right),
            ],
            FRAME,
            step * 0.1,
            step + 1,
        )

    assert manager.person_of(7) == 1
    assert manager.person_of(9) == 2
    assert manager.swap_corrections_applied == 0
    assert sink.count("IDENTITY_SWAP_CORRECTED") == 0


def test_ecart_sous_la_marge_ne_declenche_pas_de_correction(sink):
    """crossed > direct mais insuffisant (< margin) : aucune correction de mapping."""
    manager = build_manager(sink, swap_margin=0.12)
    desc_a = unit(1.0, 0.0, 0.0)
    desc_b = unit(0.0, 1.0, 0.0)

    manager.assign(
        [
            observation(7, feature=desc_a),
            observation(9, feature=desc_b),
        ],
        FRAME,
        0.0,
        1,
    )
    sink.events.clear()

    # Apparences où crossed > direct mais de ~0.07 < 0.12
    new_a = unit(1.0, 1.05, 0.0)
    new_b = unit(1.05, 1.0, 0.0)

    direct = float(np.dot(new_a, desc_a) + np.dot(new_b, desc_b))
    crossed = float(np.dot(new_a, desc_b) + np.dot(new_b, desc_a))
    assert crossed > direct
    assert crossed < direct + 0.12

    assignments = manager.assign(
        [
            observation(7, feature=new_a),
            observation(9, feature=new_b),
        ],
        FRAME,
        0.1,
        2,
    )

    assert manager.person_of(7) == 1
    assert manager.person_of(9) == 2
    assert assignments[0].person_id == 1
    assert assignments[1].person_id == 2
    assert manager.swap_corrections_applied == 0
    assert sink.count("IDENTITY_SWAP_CORRECTED") == 0


def test_swap_correction_desactivee_ne_corrige_rien(sink):
    """Quand swap_correction.enabled = False, aucun mapping n'est corrigé."""
    manager = build_manager(sink, swap_enabled=False)
    desc_a = unit(1, 0, 0)
    desc_b = unit(0, 1, 0)

    manager.assign(
        [
            observation(7, feature=desc_a),
            observation(9, feature=desc_b),
        ],
        FRAME,
        0.0,
        1,
    )

    # Inversion franche d'apparence
    assignments = manager.assign(
        [
            observation(7, feature=desc_b),
            observation(9, feature=desc_a),
        ],
        FRAME,
        0.1,
        2,
    )

    # Pas de correction : comportement V2 (maintien du mapping technique)
    assert manager.person_of(7) == 1
    assert manager.person_of(9) == 2
    assert assignments[0].person_id == 1
    assert assignments[1].person_id == 2
    assert manager.swap_corrections_applied == 0
    assert sink.count("IDENTITY_SWAP_CORRECTED") == 0


def test_trois_pistes_avec_un_seul_swap_corrige_uniquement_la_paire(sink):
    """3 pistes simultanées, seules 7 et 8 s'inversent, 9 reste intacte."""
    manager = build_manager(sink, swap_margin=0.12)
    desc_a = unit(1, 0, 0)
    desc_b = unit(0, 1, 0)
    desc_c = unit(0, 0, 1)

    manager.assign(
        [
            observation(7, feature=desc_a),
            observation(8, feature=desc_b),
            observation(9, feature=desc_c),
        ],
        FRAME,
        0.0,
        1,
    )
    assert manager.person_of(7) == 1
    assert manager.person_of(8) == 2
    assert manager.person_of(9) == 3
    sink.events.clear()

    # Frame 2 : swap entre 7 et 8 uniquement, 9 a toujours desc_c
    assignments = manager.assign(
        [
            observation(7, feature=desc_b),
            observation(8, feature=desc_a),
            observation(9, feature=desc_c),
        ],
        FRAME,
        0.1,
        2,
    )

    assert manager.person_of(7) == 2
    assert manager.person_of(8) == 1
    assert manager.person_of(9) == 3

    assert assignments[0].person_id == 2
    assert assignments[1].person_id == 1
    assert assignments[2].person_id == 3

    # Les trois boîtes se touchent : groupe de 3 résolu globalement (algorithme
    # hongrois). La transposition 7↔8 ne déplace que deux pistes ; l'événement
    # est émis par piste corrigée, avec ``group_size``. La 3e ne produit rien.
    assert manager.swap_corrections_applied == 2
    events = sink.of_type("IDENTITY_SWAP_CORRECTED")
    assert len(events) == 2
    assert all(event["group_size"] == 3 for event in events)
    by_track = {event["technical_track_id"]: event for event in events}
    assert set(by_track) == {7, 8}
    assert (by_track[7]["person_id_before"], by_track[7]["person_id_after"]) == (1, 2)
    assert (by_track[8]["person_id_before"], by_track[8]["person_id_after"]) == (2, 1)
    assert all(event["reason"] == "group_appearance_correction" for event in events)
    assert all(event["group_person_ids"] == [2, 1, 3] for event in events)



def _trois_pistes_en_permutation_circulaire(manager):
    desc_a, desc_b, desc_c = unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)
    manager.assign(
        [observation(7, feature=desc_a), observation(8, feature=desc_b), observation(9, feature=desc_c)],
        FRAME, 0.0, 1,
    )
    assert (manager.person_of(7), manager.person_of(8), manager.person_of(9)) == (1, 2, 3)
    # Chaque piste porte l'apparence de la suivante.
    return manager.assign(
        [observation(7, feature=desc_b), observation(8, feature=desc_c), observation(9, feature=desc_a)],
        FRAME, 0.1, 2,
    )


def test_permutation_circulaire_a_trois_pistes_est_corrigee_en_une_fois(sink):
    """Permutation circulaire ``A→B``, ``B→C``, ``C→A`` : corrigée intégralement.

    Porté de la branche du binôme. Par paires, la correction de (7, 8) exclut la
    piste 8 de toute autre décision et laisse le mapping partiellement faux ; la
    résolution globale N×N retrouve la permutation complète.
    """
    manager = build_manager(sink, swap_margin=0.08)
    assignments = _trois_pistes_en_permutation_circulaire(manager)
    sink_events = sink.of_type("IDENTITY_SWAP_CORRECTED")

    assert (manager.person_of(7), manager.person_of(8), manager.person_of(9)) == (2, 3, 1)
    assert [assignment.person_id for assignment in assignments] == [2, 3, 1]
    assert manager.swap_corrections_applied == 3
    assert len(sink_events) == 3
    assert all(event["group_size"] == 3 for event in sink_events)
    assert {event["technical_track_id"] for event in sink_events} == {7, 8, 9}
    assert all(event["group_person_ids"] == [2, 3, 1] for event in sink_events)
    for event in sink_events:
        assert event["similarity_crossed"] > event["similarity_direct"] + event["margin_applied"]


def test_sans_scipy_repli_sur_la_correction_par_paires(sink, monkeypatch):
    """Sans ``scipy`` : aucune résolution de groupe, comportement par paires historique."""
    import identity_manager

    monkeypatch.setattr(identity_manager, "_linear_sum_assignment", None)
    manager = build_manager(sink, swap_margin=0.08)
    _trois_pistes_en_permutation_circulaire(manager)

    events = sink.of_type("IDENTITY_SWAP_CORRECTED")
    assert events and all("technical_track_id_a" in event for event in events)
    assert all(event.get("group_size") is None for event in events)
    # La permutation circulaire n'est que partiellement corrigée par paires.
    assert (manager.person_of(7), manager.person_of(8), manager.person_of(9)) != (2, 3, 1)


def test_groupe_de_trois_pistes_disjointes_reste_traite_par_paires(sink):
    """Boîtes qui ne se touchent pas : pas de groupe, comparaison par paires."""
    manager = build_manager(sink, swap_margin=0.08)
    boxes = {7: [0.0, 100.0, 40.0, 300.0], 8: [200.0, 100.0, 240.0, 300.0], 9: [400.0, 100.0, 440.0, 300.0]}
    desc_a, desc_b, desc_c = unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)
    manager.assign(
        [observation(t, feature=d, box=boxes[t]) for t, d in ((7, desc_a), (8, desc_b), (9, desc_c))],
        FRAME, 0.0, 1,
    )
    sink.events.clear()
    manager.assign(
        [observation(t, feature=d, box=boxes[t]) for t, d in ((7, desc_b), (8, desc_a), (9, desc_c))],
        FRAME, 0.1, 2,
    )
    events = sink.of_type("IDENTITY_SWAP_CORRECTED")
    assert len(events) == 1
    assert (events[0]["technical_track_id_a"], events[0]["technical_track_id_b"]) == (7, 8)
