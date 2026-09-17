"""Niveau 3 — Intégration sans caméra.

Des sorties simulées de YOLO/BoT-SORT (ID techniques, boîtes, confiances) sont
injectées dans le pipeline complet. Les scénarios couvrent : croisement de deux
personnes, franchissements simultanés, vêtements similaires (déclenchement
attendu de ``REID_AMBIGUOUS``), occultation courte/longue/complète,
réapparition avec un nouvel ID technique, réapparition trop tardive ou trop
éloignée (doit être rejetée par la contrainte spatiale) et double réassociation.
"""

from __future__ import annotations

import pytest

from conftest import TEST_FPS, StubAppearance, identity_vectors_by_x, make_test_line, person_box
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

INSIDE_Y = 450.0
OUTSIDE_Y = 250.0
LINE_Y = 300.0

VECTORS = {
    150.0: (1.0, 0.0, 0.0),
    300.0: (0.0, 1.0, 0.0),
    450.0: (0.0, 0.0, 1.0),
}


class Sim:
    def __init__(self, manager, frame, dt=1.0 / TEST_FPS):
        self.manager = manager
        self.frame = frame
        self.dt = dt
        self.time = 0.0
        self.index = 0

    def step(self, detections=()):
        self.index += 1
        self.time += self.dt
        self.manager.process_frame(list(detections), self.frame, self.time, self.index)
        return self

    def steps(self, count, detections=()):
        for _ in range(count):
            self.step(detections)
        return self

    def warmup(self):
        return self.steps(5)


def det(track_id, y, x=300.0, confidence=0.9, height=200.0):
    return Detection(track_id, person_box(x, y, height=height), confidence)


def build(config, sink, appearance=None, line=None):
    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=appearance or StubAppearance(describe_fn=identity_vectors_by_x(VECTORS)),
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=line or make_test_line(inside_side=config.line.inside_side),
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


@pytest.fixture
def sim(test_config, sink, frame):
    manager = build(test_config, sink)
    return Sim(manager, frame)


# ---------------------------------------------------------------------------
# Croisements
# ---------------------------------------------------------------------------
def test_two_people_crossing_in_opposite_directions(sim, sink):
    """A entre pendant que B sort, dans la même frame."""
    manager = sim.manager
    # B est déjà dans la salle au démarrage (effectif initial = 1).
    sim.steps(5, [det(2, INSIDE_Y, x=450.0)])
    sim.steps(4, [det(2, INSIDE_Y, x=450.0)])
    assert manager.initial_occupancy == 1
    assert manager.occupancy_confirmed == 1

    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0), det(2, INSIDE_Y, x=450.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0), det(2, OUTSIDE_Y, x=450.0)])
    assert manager.total_in == 1
    assert manager.total_out == 1
    # A est entrée, B est sorti : il reste exactement A.
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_range == (1, 1)
    assert sink.count("INCONSISTENT_STATE") == 0
    assert sink.count("AMBIGUOUS_CROSSING") == 0


def test_simultaneous_entries(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0), det(2, OUTSIDE_Y, x=300.0), det(3, OUTSIDE_Y, x=450.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0), det(2, INSIDE_Y, x=300.0), det(3, INSIDE_Y, x=450.0)])
    assert manager.total_in == 3
    assert manager.occupancy_confirmed == 3
    assert sink.count("IN") == 3
    assert manager.occupancy_range == (3, 3)


def test_people_walking_side_by_side_keep_their_identity(sim):
    manager = sim.manager
    sim.warmup()
    for _ in range(3):
        sim.step([det(1, INSIDE_Y, x=150.0), det(2, INSIDE_Y, x=450.0)])
    identities = {
        manager.identities.person_of(1),
        manager.identities.person_of(2),
    }
    assert len(identities) == 2
    assert manager.total_new == 2


# ---------------------------------------------------------------------------
# Vêtements similaires -> décision ambiguë explicite (spec 3.3 étape 5)
# ---------------------------------------------------------------------------
def test_similar_appearance_triggers_ambiguous_event(test_config, sink, frame):
    """Deux personnes quasi identiques : la réassociation ne doit pas être silencieuse."""
    # Descripteurs identiques pour toutes les positions -> marge de sécurité nulle.
    appearance = StubAppearance(default=(1.0, 0.0, 0.0))
    manager = build(test_config, sink, appearance=appearance)
    sim = Sim(manager, frame).warmup()

    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])
    sim.steps(3, [det(2, INSIDE_Y, x=450.0)])
    assert sink.count("REID_AMBIGUOUS") + sink.count("REID_MATCH") >= 0

    # Les deux pistes disparaissent, puis une nouvelle apparaît entre les deux :
    # elle ressemble autant à l'une qu'à l'autre, la marge de sécurité est nulle.
    sim.steps(2)
    before_ambiguities = sink.count("REID_AMBIGUOUS")
    sim.step([det(9, INSIDE_Y, x=300.0)])
    assert sink.count("REID_AMBIGUOUS") > before_ambiguities
    event = sink.of_type("REID_AMBIGUOUS")[-1]
    assert event["best_similarity"] - event["runner_up_similarity"] < event["safety_margin"]
    assert len(event["candidates"]) >= 2
    assert event["resolution"] == "provisional_person"


def test_ambiguous_identity_is_marked_provisional(test_config, sink, frame):
    appearance = StubAppearance(default=(1.0, 0.0, 0.0))
    manager = build(test_config, sink, appearance=appearance)
    sim = Sim(manager, frame).warmup()
    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])
    sim.steps(3, [det(2, INSIDE_Y, x=450.0)])
    sim.steps(2)
    sim.step([det(9, INSIDE_Y, x=300.0)])
    provisional = [r for r in manager.identities.records.values() if r.provisional]
    assert provisional, "une identité provisoire doit exister après une décision ambiguë"
    drawn = [view for view in manager._views if view.provisional]
    assert drawn, "l'identité provisoire doit être signalée dans le rendu"


# ---------------------------------------------------------------------------
# Occultations
# ---------------------------------------------------------------------------
def test_short_occlusion_keeps_identity(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])
    assert manager.total_in == 1

    sim.steps(3)  # occultation courte, dans la grâce
    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])
    assert manager.total_in == 1, "aucun double comptage après occultation"
    assert sink.count("IN") == 1
    assert manager.occupancy_confirmed == 1
    assert sink.count("INCONSISTENT_STATE") == 0


def test_complete_occlusion_with_new_technical_id_is_reconnected(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])

    sim.steps(4)
    sim.steps(3, [det(42, INSIDE_Y, x=155.0)])  # nouvel ID technique, même personne
    assert sink.count("REID_MATCH") == 1
    assert manager.total_in == 1
    assert manager.total_new == 0
    assert manager.occupancy_confirmed == 1


def test_long_occlusion_ends_in_the_uncertain_bound(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])
    sim.steps(20)  # bien au-delà de la grâce
    # L'occultation ne retire jamais la personne de l'occupation confirmée.
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_observed == 0
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (0, 1)
    assert sink.count("PURGE") == 1
    assert sink.count("AMBIGUOUS_CROSSING") == 0


def test_reappearance_too_late_is_rejected(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])
    sim.steps(50)
    sink.events.clear()
    sim.steps(5, [det(99, INSIDE_Y, x=150.0)])
    assert sink.count("REID_MATCH") == 0, "hors fenêtre de galerie : pas de réassociation"
    assert manager.total_new == 1
    assert manager.occupancy_uncertain == 1


def test_reappearance_too_far_is_rejected(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0)])
    sim.steps(2)  # occultation très courte
    # Réapparition à 600 px : au-delà de 1,5 × 200 × 0,4 + 60 = 180 px tolérés.
    sim.steps(3, [det(9, INSIDE_Y, x=560.0)])
    assert sink.count("REID_MATCH") == 0
    assert manager.identities.person_of(9) != manager.identities.person_of(1)


def test_occlusion_near_the_line_is_not_counted_twice(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(3, [det(1, LINE_Y)])
    assert manager.total_in == 0 and manager.total_out == 0
    sim.steps(4)  # disparition dans la zone morte
    sim.steps(3, [det(9, LINE_Y)])
    assert manager.total_in == 0 and manager.total_out == 0
    assert sink.count("AMBIGUOUS_CROSSING") == 0


# ---------------------------------------------------------------------------
# Double réassociation (spec 4.5)
# ---------------------------------------------------------------------------
def test_duplicate_person_claim_is_reported(test_config, sink, frame):
    appearance = StubAppearance(default=(1.0, 0.0, 0.0))
    manager = build(test_config, sink, appearance=appearance)
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [det(1, INSIDE_Y, x=150.0)])
    person_id = manager.identities.person_of(1)

    # Deux pistes techniques pointant sur la même identité.
    manager.identities.technical_to_person[2] = person_id
    sink.events.clear()
    sim.step([det(1, INSIDE_Y, x=150.0), det(2, INSIDE_Y, x=152.0)])
    assert sink.count("INCONSISTENT_STATE") == 1
    assert sink.of_type("INCONSISTENT_STATE")[0]["kind"] == "duplicate_person_claim"


# ---------------------------------------------------------------------------
# Occupation cohérente sur une séquence complète
# ---------------------------------------------------------------------------
def test_full_session_scenario_counts_are_exact(sim, sink):
    """Séquence mêlant entrées, sorties, occultation et sortie non observée."""
    manager = sim.manager
    sim.warmup()
    # 1 : entrée propre, puis reste visible dans toute la suite.
    with_person_one = lambda others=(): [det(1, INSIDE_Y, x=150.0), *others]  # noqa: E731
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0)])
    sim.steps(3, with_person_one())
    # 2 : entrée puis sortie propre
    sim.steps(2, with_person_one([det(2, OUTSIDE_Y, x=300.0)]))
    sim.steps(3, with_person_one([det(2, INSIDE_Y, x=300.0)]))
    sim.steps(3, with_person_one([det(2, OUTSIDE_Y, x=300.0)]))
    # 3 : entrée puis disparition définitive
    sim.steps(2, with_person_one([det(3, OUTSIDE_Y, x=450.0)]))
    sim.steps(3, with_person_one([det(3, INSIDE_Y, x=450.0)]))
    sim.steps(25, with_person_one())

    assert manager.total_in == 3
    assert manager.total_out == 1
    assert manager.total_new == 0
    assert sink.count("AMBIGUOUS_CROSSING") == 0
    # 1 est toujours présente, 2 est sortie, 3 est dans l'incertitude :
    # l'effectif opérationnel vaut 3 entrées - 1 sortie = 2.
    assert manager.occupancy_confirmed == 2
    assert manager.occupancy_observed == 1
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (1, 2)
    # Chaque purge est journalisée ; seule celle de 3 est incertaine (2 est
    # sortie proprement avant d'être oubliée).
    purges = sink.of_type("PURGE")
    assert len(purges) == 2
    assert sum(1 for event in purges if event["occupancy_impact"] == "uncertain") == 1
