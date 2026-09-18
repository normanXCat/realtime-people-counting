"""Niveau 1-3 — Garde-fou « inversement d'identifiant » (swap) sous occlusion.

Contrairement à une **perte de piste** (déjà couverte par les tests de
réassociation), un **swap** laisse deux identifiants techniques actifs en
continu : aucun identifiant ne disparaît, donc aucun ``TECHNICAL_ID_CHANGED``
ni ``TRACK_ID_ABSENT`` ne se déclenche. Le seul symptôme observable côté
pipeline est une **discontinuité d'apparence** sur une piste qui n'a jamais été
perdue.

Ces tests vérifient que ce cas est mesuré (``TRACK_ID_APPEARANCE_DISCONTINUITY``)
sans jamais réaffecter d'identité automatiquement, et qu'un croisement sans swap
ne produit aucun faux diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import AppearanceContinuityConfig, LongTermReidConfig
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


def build_manager(sink, **overrides):
    """Gestionnaire de test : galerie volontairement permissive pour isoler le garde-fou."""
    continuity = AppearanceContinuityConfig(
        enabled=overrides.pop("enabled", True),
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
        grace_period_seconds=5.0,
        event_sink=sink,
        **overrides,
    )


@pytest.fixture
def sink():
    return ListSink("swap")


# ---------------------------------------------------------------------------
# Croisement sans swap : aucune discontinuité ne doit être inventée
# ---------------------------------------------------------------------------
def test_deux_personnes_qui_se_croisent_ne_produisent_aucun_swap(sink):
    """Deux pistes gardent chacune leur apparence pendant un croisement.

    Les positions se croisent (aucune occlusion réelle, juste proximité) : chaque
    identifiant technique conserve son descripteur, donc aucun
    ``TRACK_ID_APPEARANCE_DISCONTINUITY`` ne doit être émis et aucun
    ``person_id`` ne doit changer.
    """
    manager = build_manager(sink)
    left = unit(1, 0, 0)
    right = unit(0, 1, 0)

    # Les deux personnes se croisent : les ancres se déplacent, pas les identités.
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
    assert manager.appearance_discontinuities == 0
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 0


def test_une_variation_progressive_d_apparence_n_est_pas_un_swap(sink):
    """Une rotation / un changement de pose progressif reste sous le seuil."""
    manager = build_manager(sink, similarity_threshold=0.5)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    # Décalage angulaire faible entre deux frames (similarité ≈ 0,995).
    manager.assign([observation(7, feature=unit(0.9999, 0.01, 0))], FRAME, 0.1, 2)
    manager.assign([observation(7, feature=unit(0.999, 0.03, 0))], FRAME, 0.2, 3)
    assert manager.appearance_discontinuities == 0
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 0


# ---------------------------------------------------------------------------
# Swap interne : une piste continue change brutalement d'apparence
# ---------------------------------------------------------------------------
def test_un_swap_interne_est_journalise_et_non_corrige(sink):
    """Le garde-fou mesure sans réaffecter : l'identité reste celle du tracker."""
    manager = build_manager(sink)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    assert manager.person_of(7) == 1
    sink.events.clear()

    # Frame suivante : la piste 7 « porte » soudain l'apparence d'une autre personne.
    assignments = manager.assign([observation(7, feature=unit(-1, 0, 0))], FRAME, 0.1, 2)

    events = sink.of_type("TRACK_ID_APPEARANCE_DISCONTINUITY")
    assert len(events) == 1
    event = events[0]
    assert event["technical_track_id"] == 7
    assert event["person_id_before"] == 1
    assert event["reason"] == "possible_internal_swap"
    assert event["similarity_to_previous_descriptor"] == pytest.approx(-1.0, abs=1e-3)
    assert event["gap_frames"] == 1
    assert event["threshold"] == pytest.approx(0.5)
    # Aucune correction automatique : l'identité logique n'est pas réaffectée.
    assert assignments[0].person_id == 1
    assert manager.person_of(7) == 1
    assert manager.appearance_discontinuities == 1


def test_la_discontinuite_n_est_pas_repete_e_a_chaque_frame(sink):
    """Tant que la continuité n'est pas rétablie, une seule ligne est écrite."""
    manager = build_manager(sink, min_interval_seconds=0.0)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    new_look = unit(-1, 0, 0)
    manager.assign([observation(7, feature=new_look)], FRAME, 0.1, 2)
    manager.assign([observation(7, feature=new_look)], FRAME, 0.2, 3)
    manager.assign([observation(7, feature=new_look)], FRAME, 0.3, 4)
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 1
    assert manager.appearance_discontinuities == 1


def test_un_retour_a_la_continuite_rearme_le_diagnostic(sink):
    """Une nouvelle discontinuité après retour à la normale est signalée."""
    manager = build_manager(sink, min_interval_seconds=0.0)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    other = unit(-1, 0, 0)
    manager.assign([observation(7, feature=other)], FRAME, 0.1, 2)   # 1re discontinuité
    # Retour à une apparence cohérente avec la nouvelle référence.
    manager.assign([observation(7, feature=other)], FRAME, 0.2, 3)
    back = unit(0, 1, 0)
    manager.assign([observation(7, feature=back)], FRAME, 0.3, 4)    # 2e discontinuité
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 2
    assert manager.appearance_discontinuities == 2


def test_min_interval_seconds_limite_le_debit_par_piste(sink):
    manager = build_manager(sink, min_interval_seconds=1.0)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(7, feature=unit(-1, 0, 0))], FRAME, 0.1, 2)  # émise
    manager.assign([observation(7, feature=unit(0, 1, 0))], FRAME, 0.2, 3)   # sous 1 s -> pas d'émission
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 1


# ---------------------------------------------------------------------------
# Limite du garde-fou : une piste réellement perdue n'est pas un swap
# ---------------------------------------------------------------------------
def test_une_piste_reellement_perdue_ne_produit_pas_de_swap(sink):
    """Au-delà de ``max_gap_frames``, c'est le chemin de réassociation normal.

    Réapparaître avec une autre apparence après une absence réelle est un cas
    déjà couvert par le filtre spatio-temporel et la marge de sécurité : le
    signaler comme un swap interne serait un faux diagnostic.
    """
    manager = build_manager(sink, max_gap_frames=2)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    manager.assign([], FRAME, 0.2, 3)
    manager.assign([], FRAME, 0.3, 4)
    manager.assign([observation(7, feature=unit(-1, 0, 0))], FRAME, 0.4, 5)
    assert manager.appearance_discontinuities == 0
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 0


def test_un_ecart_dans_la_tolerance_est_encore_verifie(sink):
    """Deux frames consécutives manquées (gap = ``max_gap_frames``) restent un swap."""
    manager = build_manager(sink, max_gap_frames=2)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    manager.assign([observation(7, feature=unit(-1, 0, 0))], FRAME, 0.2, 3)
    assert manager.appearance_discontinuities == 1


# ---------------------------------------------------------------------------
# Désactivation et robustesse
# ---------------------------------------------------------------------------
def test_le_garde_fou_desactive_n_emet_rien(sink):
    manager = build_manager(sink, enabled=False)
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(7, feature=unit(-1, 0, 0))], FRAME, 0.1, 2)
    assert manager.appearance_discontinuities == 0
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 0


def test_un_descripteur_absent_ne_declenche_rien(sink):
    """Sans descripteur exploitable, aucun diagnostic n'est inventé.

    Le descripteur est refusé par l'extracteur (crop inexploitable) : aucune
    comparaison n'est possible, donc rien n'est écrit — un garde-fou silencieux
    est préférable à un garde-fou qui devine.
    """
    from conftest import StubAppearance

    manager = build_manager(sink, appearance=StubAppearance(default=None))
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([observation(7, feature=None)], FRAME, 0.1, 2)
    manager.assign([observation(7, feature=None)], FRAME, 0.2, 3)
    assert manager.appearance_discontinuities == 0
    assert sink.count("TRACK_ID_APPEARANCE_DISCONTINUITY") == 0


def test_la_distribution_des_similarites_est_publiee_pour_calibrer_le_seuil(sink):
    """Le seuil se règle sur des mesures, pas sur une intuition.

    La distribution observée doit être disponible pour choisir
    ``similarity_threshold`` : un seuil sous le p1 mesuré ne produit aucun faux
    diagnostic, un seuil au-dessus en produit.
    """
    manager = build_manager(sink, similarity_threshold=0.5)
    assert manager.continuity_similarity_summary()["samples"] == 0

    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    # Trois comparaisons : quasi-identique, puis franchement différent, puis retour.
    manager.assign([observation(7, feature=unit(0.9999, 0.01, 0))], FRAME, 0.1, 2)
    manager.assign([observation(7, feature=unit(0, 1, 0))], FRAME, 0.2, 3)
    manager.assign([observation(7, feature=unit(0, 1, 0))], FRAME, 0.3, 4)

    summary = manager.continuity_similarity_summary()
    assert summary["samples"] == 3
    assert summary["min"] == pytest.approx(0.01, abs=0.02)
    assert summary["median"] is not None and summary["median"] > 0.9
    assert summary["below_threshold"] == 1


def test_la_memoire_des_descripteurs_reste_bornee(sink):
    """Les descripteurs des pistes disparues ne s'accumulent pas indéfiniment."""
    manager = build_manager(sink, max_gap_frames=0)
    for frame in range(1, 21):
        manager.assign(
            [observation(frame, feature=unit(1, 0, 0), anchor=(50.0 * frame, 300.0))],
            FRAME,
            frame * 0.1,
            frame,
        )
    assert len(manager._track_appearance) <= 1
