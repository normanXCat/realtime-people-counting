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


def _merge_manager(sink):
    """Gestionnaire à seuil strict : plancher d'assouplissement très bas."""
    return IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True,
            similarity_threshold=0.8,
            safety_margin=0.0,
            long_absence_seconds=0.1,
            long_absence_similarity_floor=0.3,
            v_max_ratio=1e6,
            spatial_margin_ratio=1e6,
        ),
        grace_period_seconds=5.0,
        event_sink=sink,
    )


def test_reappariement_apres_fusion_exige_une_similarite_stricte(sink):
    """Une boîte multi-personnes fige l'empreinte ; après séparation, un ID
    ancien n'est réattribué que si la similarité dépasse le seuil nominal (aucun
    assouplissement progressif autorisé pendant l'épisode de fusion).
    """
    manager = _merge_manager(sink)
    original = unit(1, 0, 0)
    changed = unit(0.7, 0.713, 0)  # similarité ≈ 0,7 avec l'original

    # Frame 1 : identité 1 établie.
    manager.assign([observation(7, feature=original)], FRAME, 0.0, 1)
    assert manager.person_of(7) == 1
    record = manager.records[1]

    # Frame 2 : la boîte fusionne (deux personnes dans une boîte).
    multi = observation(7, feature=original)
    multi.possible_multi_person = True
    manager.assign([multi], FRAME, 0.1, 2)
    manager.note_multi_person_box(1, 0.1)

    assert record.merge_fingerprint is not None
    np.testing.assert_array_almost_equal(record.merge_fingerprint, original)
    assert record.merge_flagged_at_s == pytest.approx(0.1)

    # Frame 3 : séparation -> nouvelle piste technique. L'apparence a dérivé
    # jusqu'à ≈ 0,7 (au-dessus du plancher 0,3, sous le seuil nominal 0,8).
    sink.events.clear()
    second = manager.assign([observation(9, feature=changed)], FRAME, 1.0, 3)[0]
    assert second.person_id == 2, "pas de réattribution de l'ID 1 sans similarité stricte"
    assert second.reason == "below_strict_similarity_threshold"

    # Frame 4 : une apparence quasi identique est bien réappariée à l'ID 1.
    close = unit(1, 0.05, 0)
    real = manager.assign([observation(11, feature=close)], FRAME, 1.1, 4)[0]
    assert real.person_id == 1
    assert real.reason == "reid_match"
    match_event = sink.of_type("REID_MATCH")[-1]
    assert match_event["strict_reappearance"] is True
    assert match_event["reference"] == "merge_fingerprint"
    assert record.strict_rematches == 1
    # L'épisode de fusion est clos : l'empreinte est libérée.
    assert record.merge_fingerprint is None


def test_sans_fusion_l_assouplissement_progressif_reste_actif(sink):
    """Contre-épreuve : hors fusion, la même apparence dérivée est réacceptée."""
    manager = _merge_manager(sink)
    original = unit(1, 0, 0)
    changed = unit(0.7, 0.713, 0)

    manager.assign([observation(7, feature=original)], FRAME, 0.0, 1)
    # Aucune fusion : pas de note_multi_person_box.
    assignment = manager.assign([observation(9, feature=changed)], FRAME, 1.0, 2)[0]
    assert assignment.person_id == 1
    assert assignment.reason == "reid_match"


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

    # Groupe de 3 résolu globalement : la transposition 7↔8 ne déplace que deux
    # pistes, mais l'événement est émis **par piste corrigée** (avec
    # ``group_size``), conformément à la résolution de groupe. La 3e piste, qui
    # garde son identité, ne produit aucun événement.
    assert manager.swap_corrections_applied == 2
    events = sink.of_type("IDENTITY_SWAP_CORRECTED")
    assert len(events) == 2
    assert all(event["group_size"] == 3 for event in events)
    assert {event["technical_track_id"] for event in events} == {7, 8}
    by_track = {event["technical_track_id"]: event for event in events}
    assert by_track[7]["person_id_before"] == 1
    assert by_track[7]["person_id_after"] == 2
    assert by_track[8]["person_id_before"] == 2
    assert by_track[8]["person_id_after"] == 1
    assert all(event["reason"] == "group_appearance_correction" for event in events)
    assert all(event["group_person_ids"] == [2, 1, 3] for event in events)


def test_permutation_circulaire_a_trois_pistes_est_corrigee_en_une_fois(sink):
    """Permutation circulaire ``A→B``, ``B→C``, ``C→A`` : corrigée intégralement.

    L'implémentation par paires ne résout pas ce cas : elle corrige la paire
    (7, 8) puis exclut la piste 8 de toute décision, laissant 8 et 9 mappées à la
    mauvaise personne (1 piste sur 3 correcte, mapping partiellement erroné).
    La résolution globale N×N doit retrouver la permutation complète.
    """
    manager = build_manager(sink, swap_margin=0.08)
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
    assert (manager.person_of(7), manager.person_of(8), manager.person_of(9)) == (1, 2, 3)
    sink.events.clear()

    # Permutation circulaire : chaque piste porte l'apparence de la suivante.
    assignments = manager.assign(
        [
            observation(7, feature=desc_b),
            observation(8, feature=desc_c),
            observation(9, feature=desc_a),
        ],
        FRAME,
        0.1,
        2,
    )

    assert manager.person_of(7) == 2
    assert manager.person_of(8) == 3
    assert manager.person_of(9) == 1
    assert [assignment.person_id for assignment in assignments] == [2, 3, 1]
    assert manager.swap_corrections_applied == 3

    events = sink.of_type("IDENTITY_SWAP_CORRECTED")
    assert len(events) == 3
    assert all(event["group_size"] == 3 for event in events)
    assert {event["technical_track_id"] for event in events} == {7, 8, 9}
    assert all(event["group_person_ids"] == [2, 3, 1] for event in events)
    # Le gain total (3 - 0 = 3) dépasse largement la marge.
    for event in events:
        assert event["similarity_crossed"] > event["similarity_direct"] + event["margin_applied"]


@pytest.mark.parametrize(
    "margin,expected_corrections",
    [
        (0.01, 2),  # gain 0,02 > marge : le groupe est corrigé
        (0.50, 0),  # gain 0,02 < marge : le bruit ne réécrit pas le mapping
    ],
)
def test_groupe_a_trois_pistes_le_gain_doit_depasser_la_marge(
    sink, margin, expected_corrections
):
    """Même permutation, seule la marge change : c'est bien elle qui tranche.

    Deux apparences très proches s'échangent (gain total ≈ 0,02), la troisième
    reste intacte. Sous la marge, la correction est refusée : un écart de bruit ne
    doit pas réécrire le mapping.
    """
    manager = build_manager(sink, swap_margin=margin)
    desc_a = unit(1, 0, 0)
    desc_b = unit(0.99, 0.1414, 0.0)
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
    sink.events.clear()

    manager.assign(
        [
            observation(7, feature=desc_b),
            observation(8, feature=desc_a),
            observation(9, feature=desc_c),
        ],
        FRAME,
        0.1,
        2,
    )

    assert manager.swap_corrections_applied == expected_corrections
    assert sink.count("IDENTITY_SWAP_CORRECTED") == expected_corrections
    if expected_corrections:
        assert manager.person_of(7) == 2
        assert manager.person_of(8) == 1
        assert manager.person_of(9) == 3
        gains = sink.of_type("IDENTITY_SWAP_CORRECTED")
        assert all(event["group_size"] == 3 for event in gains)
        assert all(
            event["similarity_crossed"] > event["similarity_direct"] + event["margin_applied"]
            for event in gains
        )
    else:
        assert manager.person_of(7) == 1
        assert manager.person_of(8) == 2
        assert manager.person_of(9) == 3


def test_groupe_de_trois_pistes_eloignees_n_est_pas_resolu_ensemble(sink):
    """Contre-épreuve de proximité : trois pistes disjointes restent indépendantes.

    Les trois permutations sont inverses, mais les pistes sont à plus d'une
    hauteur de bbox l'une de l'autre : elles ne forment pas un groupe, donc le
    `group_proximity_ratio` les exclut l'une de l'autre et seule la paire
    chevauchante peut être corrigée.
    """
    manager = build_manager(sink, swap_margin=0.08)
    desc_a = unit(1, 0, 0)
    desc_b = unit(0, 1, 0)
    desc_c = unit(0, 0, 1)
    boxes = {
        7: [0.0, 100.0, 40.0, 300.0],
        8: [500.0, 100.0, 540.0, 300.0],
        9: [1000.0, 100.0, 1040.0, 300.0],
    }

    manager.assign(
        [
            observation(7, anchor=(20.0, 300.0), feature=desc_a, box=boxes[7]),
            observation(8, anchor=(520.0, 300.0), feature=desc_b, box=boxes[8]),
            observation(9, anchor=(1020.0, 300.0), feature=desc_c, box=boxes[9]),
        ],
        FRAME,
        0.0,
        1,
    )
    sink.events.clear()
    manager.assign(
        [
            observation(7, anchor=(20.0, 300.0), feature=desc_b, box=boxes[7]),
            observation(8, anchor=(520.0, 300.0), feature=desc_c, box=boxes[8]),
            observation(9, anchor=(1020.0, 300.0), feature=desc_a, box=boxes[9]),
        ],
        FRAME,
        0.1,
        2,
    )
    # Aucun groupe de plus de deux pistes : aucune résolution globale appliquée.
    assert manager.swap_corrections_applied == 0
    assert manager.person_of(7) == 1
    assert manager.person_of(8) == 2
    assert manager.person_of(9) == 3
    assert sink.count("IDENTITY_SWAP_CORRECTED") == 0
