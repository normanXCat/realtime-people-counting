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

    assert manager.swap_corrections_applied == 1
    events = sink.of_type("IDENTITY_SWAP_CORRECTED")
    assert len(events) == 1
    assert events[0]["technical_track_id_a"] == 7
    assert events[0]["technical_track_id_b"] == 8
