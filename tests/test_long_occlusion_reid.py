"""Ré-identification après occultation longue (mémoire d'identité, spec 3.3).

Symptôme corrigé : une personne cachée longtemps, dont la piste technique a
expiré, n'était plus jamais réassociée. Trois exigences sont vérifiées ici :

1. la durée de rétention de la galerie d'apparence est **découplée** explicitement
   de la période de grâce d'occupation
   (``reid.long_term.gallery_retention_seconds: 12.0``) : la fenêtre totale
   passe à grâce + rétention = 17 s, ce qui couvre l'occlusion maximale
   mesurée (13,5 s, rapport §4.6) que l'ancienne valeur alignée (10 s)
   laissait sortir de mémoire par construction. Le repli sur
   ``grace_period_seconds`` quand la clé est **omise** reste disponible et est
   testé séparément, dans un manager construit explicitement avec
   ``gallery_retention_seconds=None`` ;
2. le seuil d'apparence peut être assoupli de façon **progressive** selon la
   durée d'absence — mais uniquement si un plancher est déclaré, et jamais en
   dessous de ce plancher (aucun assouplissement subi par défaut) ;
3. une réapparition après une longue occultation correctement réassociée ne
   produit ni ``NEW``, ni changement d'occupation ; au-delà de la rétention, le
   comportement reste explicite : la personne comptée reste ``incertaine``
   (borne haute), aucune sortie n'est inventée et l'ambiguïté est publiée dans
   l'encadrement ``[observée, opérationnelle]`` au lieu d'être tranchée.

Occultations mesurées dans ``results/*/events.jsonl`` : médiane 2,5 s, p90 5,1 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    TEST_FPS,
    StubAppearance,
    identity_vectors_by_x,
    make_test_line,
    person_box,
)
from config import LongTermReidConfig
from events import ListSink
from identity_manager import IdentityManager, Observation
from occupancy_manager import Detection, OccupancyManager

FRAME = np.zeros((600, 600, 3), dtype=np.uint8)
INSIDE_Y = 450.0
OUTSIDE_Y = 250.0
VECTORS = {150.0: (1.0, 0.0, 0.0), 460.0: (0.0, 1.0, 0.0)}


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


@pytest.fixture
def long_config(make_config):
    """Grâce d'occupation allongée : 3 s = 30 frames à 10 fps."""
    return make_config({
        "line": {"inside_side": "negative"},
        "timing": {
            "warmup_seconds": 0.3,
            "warmup_min_frames": 4,
            "confirmation_seconds": 0.2,
            "grace_period_seconds": 3.0,
            "fps_estimate_window": 4,
        },
        "occupancy": {"snapshot_interval_seconds": 0.1},
        "display": {"enabled": False},
    })


class Sim:
    def __init__(self, manager, frame=FRAME):
        self.manager = manager
        self.frame = frame
        self.time = 0.0
        self.index = 0

    def step(self, detections=()):
        self.index += 1
        self.time += 1.0 / TEST_FPS
        self.manager.process_frame(list(detections), self.frame, self.time, self.index)
        return self

    def steps(self, count, detections=()):
        for _ in range(count):
            self.step(detections)
        return self


def det(track_id, y, x=150.0, height=200.0, confidence=0.9):
    return Detection(track_id, person_box(x, y, height=height), confidence)


def build(config, sink, appearance=None) -> OccupancyManager:
    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=appearance
        or StubAppearance(describe_fn=identity_vectors_by_x(VECTORS)),
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=make_test_line(inside_side=config.line.inside_side),
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


def enter(sim, track_id=1, x=150.0):
    sim.steps(2, [det(track_id, OUTSIDE_Y, x=x)])
    sim.steps(2, [det(track_id, INSIDE_Y, x=x)])
    return sim


# ---------------------------------------------------------------------------
# 1. Durée de rétention : alignée sur la grâce, découplable explicitement
# ---------------------------------------------------------------------------
def test_production_retention_covers_the_measured_long_occlusions(config):
    """La config livrée porte une rétention explicite de 12,0 s.

    C'est une vérification de **valeur de production** : elle doit porter sur
    12,0 et non sur ``None``. Le repli sur la grâce est un comportement
    distinct, couvert par le test suivant.
    """
    manager = IdentityManager(
        long_term=config.reid.long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    assert config.reid.long_term.gallery_retention_seconds == pytest.approx(12.0)
    assert manager.purge_retention_seconds == pytest.approx(12.0)
    # Fenêtre totale = grâce + rétention : strictement supérieure à l'occlusion
    # maximale mesurée (13,5 s), sans quoi les occultations longues — celles qui
    # posent problème — resteraient hors mémoire par construction.
    total_window = (
        config.timing.grace_period_seconds + manager.purge_retention_seconds
    )
    assert total_window > 13.5


def test_retention_falls_back_on_the_occupancy_grace_when_omitted():
    """Repli explicite : une rétention omise aligne la mémoire sur la grâce."""
    manager = IdentityManager(
        long_term=LongTermReidConfig(enabled=True, gallery_retention_seconds=None),
        grace_period_seconds=3.0,
    )
    assert manager.purge_retention_seconds == pytest.approx(3.0)


def test_explicit_retention_decouples_the_memory_from_the_occupancy_grace():
    """Une rétention déclarée s'applique réellement, sans attendre la grâce."""
    manager = IdentityManager(
        long_term=LongTermReidConfig(enabled=True, gallery_retention_seconds=0.5),
        grace_period_seconds=1.0,
    )
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    # Fenêtre totale = grâce (1,0 s) + rétention (0,5 s) = 1,5 s.
    assert manager.release_expired(1.4) == []
    assert manager.release_expired(1.6) == [1]


# ---------------------------------------------------------------------------
# 2. Tolérance progressive, strictement encadrée
# ---------------------------------------------------------------------------
def test_threshold_is_constant_unless_a_floor_is_declared():
    manager = IdentityManager(long_term=LongTermReidConfig(similarity_threshold=0.8))
    assert manager.effective_similarity_threshold(0.0) == pytest.approx(0.8)
    assert manager.effective_similarity_threshold(30.0) == pytest.approx(0.8)


def test_threshold_floor_is_progressive_and_never_crossed():
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            similarity_threshold=0.8,
            long_absence_seconds=2.0,
            long_absence_similarity_floor=0.6,
        )
    )
    assert manager.effective_similarity_threshold(0.0) == pytest.approx(0.8)
    assert manager.effective_similarity_threshold(1.0) == pytest.approx(0.7)
    assert manager.effective_similarity_threshold(2.0) == pytest.approx(0.6)
    # Jamais en dessous du plancher, même pour une absence très longue.
    assert manager.effective_similarity_threshold(60.0) == pytest.approx(0.6)


def test_a_floor_equal_to_the_threshold_disables_the_relaxation():
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            similarity_threshold=0.8, long_absence_similarity_floor=0.8
        )
    )
    assert manager.effective_similarity_threshold(5.0) == pytest.approx(0.8)


def test_progressive_tolerance_recovers_a_slightly_changed_appearance():
    """Similarité 0,7 : rejetée au seuil nominal, acceptée avec un plancher à 0,6."""
    changed = unit(0.7, 0.713, 0.05)

    def run(floor):
        manager = IdentityManager(
            long_term=LongTermReidConfig(
                enabled=True,
                similarity_threshold=0.8,
                safety_margin=0.0,
                long_absence_seconds=2.0,
                long_absence_similarity_floor=floor,
            ),
            grace_period_seconds=1.0,
            event_sink=ListSink("reid"),
        )
        manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
        manager.assign([], FRAME, 0.1, 2)
        return manager.assign([observation(8, feature=changed)], FRAME, 2.0, 3)[0]

    strict = run(None)
    assert strict.person_id == 2 and strict.reason == "below_similarity_threshold"

    relaxed = run(0.6)
    assert relaxed.person_id == 1, "une apparence légèrement changée reste la même personne"
    assert relaxed.reason == "reid_match"
    assert relaxed.similarity == pytest.approx(0.7, abs=1e-3)


def test_progressive_tolerance_still_rejects_a_different_person():
    """Le plancher borne l'assouplissement : un autre individu reste distinct."""
    other = unit(0.1, 0.99, 0.0)
    manager = IdentityManager(
        long_term=LongTermReidConfig(
            enabled=True,
            similarity_threshold=0.8,
            safety_margin=0.0,
            long_absence_seconds=2.0,
            long_absence_similarity_floor=0.6,
        ),
        grace_period_seconds=1.0,
        event_sink=ListSink("reid"),
    )
    manager.assign([observation(7, feature=unit(1, 0, 0))], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)
    assignment = manager.assign([observation(8, feature=other)], FRAME, 2.0, 3)[0]
    assert assignment.person_id == 2
    assert assignment.reason == "below_similarity_threshold"


# ---------------------------------------------------------------------------
# 3. Occupation : réapparition longue, aucun NEW, aucun changement d'occupation
# ---------------------------------------------------------------------------
def test_typical_long_occlusion_is_reassociated_without_any_new(long_config, sink):
    """Occultation de 2,5 s (médiane mesurée) : même ``person_id``, occupation stable."""
    manager = build(long_config, sink)
    sim = Sim(manager).steps(5)
    enter(sim)
    assert manager.total_in == 1
    assert manager.occupancy_operational == 1
    person_id = manager.identities.person_of(1)

    sim.steps(25)                                   # 2,5 s sans aucune détection
    assert manager.total_out == 0
    assert manager.occupancy_operational == 1

    sink.events.clear()
    sim.steps(3, [det(42, INSIDE_Y, x=155.0)])      # nouvel ID technique, même personne

    changes = sink.of_type("TECHNICAL_ID_CHANGED")
    assert len(changes) == 1
    assert changes[0]["person_id"] == person_id
    assert sink.count("REID_MATCH") == 1
    assert sink.count("NEW") == 0, "une réapparition réassociée n'est pas une nouvelle présence"
    assert manager.total_new == 0
    assert manager.total_in == 1 and manager.total_out == 0
    assert manager.occupancy_operational == 1
    assert manager.occupancy_uncertain == 0, "la personne est de nouveau observée"
    assert manager.occupancy_range == (1, 1)


def test_long_occlusion_within_the_12s_retention_is_still_reassociated(long_config):
    """Preuve que la rétention de 12 s protège : à 7 s, la mémoire est ACTIVE.

    La fenêtre de galerie du test vaut ici grâce (3 s) + rétention (12 s) = 15 s,
    soit 150 frames à 10 fps. À 7 s d'occultation — au-delà de l'ancienne
    fenêtre alignée (3 + 3 = 6 s), qui aurait déjà libéré la mémoire — la
    personne doit donc encore être ré-associée sous le **même** ``person_id``.
    """
    sink = ListSink("long")
    manager = build(long_config, sink)
    sim = Sim(manager).steps(5)
    enter(sim)
    assert manager.occupancy_operational == 1
    person_id = manager.identities.person_of(1)

    sim.steps(70)                                   # 7 s d'occultation
    assert manager.identities.person_of(1) is not None, (
        "à 7 s, la rétention de 12 s doit encore maintenir la mémoire"
    )
    assert manager.occupancy_operational == 1

    sink.events.clear()
    sim.steps(3, [det(42, INSIDE_Y, x=152.0)])      # nouvel ID technique, même personne

    assert sink.count("REID_MATCH") == 1, "dans la fenêtre : la réassociation a lieu"
    assert sink.count("NEW") == 0
    assert sink.count("TECHNICAL_ID_CHANGED") == 1
    assert manager.identities.person_of(42) == person_id
    assert manager.total_new == 0
    assert manager.occupancy_operational == 1


def test_occlusion_beyond_retention_is_explicit_and_never_silent(long_config):
    """Au-delà de la rétention : comportement explicite, jamais silencieux.

    Fenêtre de galerie = grâce (3 s) + rétention (12 s) = 15 s (150 frames).
    Au-delà, deux exigences sont vérifiées ensemble :

    - la mémoire de galerie est bien libérée (aucune réassociation possible) ;
    - la personne comptée n'est **jamais** retirée en silence : elle reste dans la
      borne haute (``occupancy_uncertain``), la sortie n'est pas confirmée, et
      l'encadrement publié ``[observée, opérationnelle]`` expose l'ambiguïté au
      lieu de la trancher. Une réapparition devient donc une nouvelle identité,
      ce qui est visible dans le journal plutôt que deviné.
    """
    sink = ListSink("long")
    manager = build(long_config, sink)
    sim = Sim(manager).steps(5)
    enter(sim)
    assert manager.occupancy_operational == 1

    # 150 frames = 15 s : la fenêtre totale (3 s de grâce + 12 s de rétention)
    # est franchie, avec une marge de 10 frames.
    sim.steps(160)
    assert manager.occupancy_operational == 1
    assert manager.occupancy_observed == 0
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (0, 1)
    assert sink.count("OUT") == 0
    assert sink.count("PURGE") >= 1, "la purge est toujours journalisée"
    assert manager.identities.person_of(1) is None, "la mémoire de galerie est libérée"

    sink.events.clear()
    sim.steps(6, [det(9, INSIDE_Y, x=152.0)])

    assert sink.count("REID_MATCH") == 0, "hors fenêtre de galerie : aucune réassociation"
    assert manager.total_out == 0, "aucune sortie inventée"
    assert manager.total_new == 1, "la réapparition devient une identité distincte"
    # L'ancienne personne reste dans la borne haute : l'encadrement expose les
    # deux hypothèses au lieu d'en supprimer une.
    assert manager.occupancy_operational == 2
    assert manager.occupancy_observed == 1
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (1, 2)
    assert sink.count("NEW") == 1
    assert sink.count("PURGE") == 0, "aucune purge silencieuse à la réapparition"
