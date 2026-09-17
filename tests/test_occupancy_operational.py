"""Niveau 3 — Occupation opérationnelle / incertaine, purge, faux comptages.

La politique vérifiée ici (spec 6 à 8 du prompt) est volontairement conservatrice :

- une personne comptée à l'intérieur **reste comptée** tant qu'aucune sortie
  confirmée vers l'extérieur n'a été observée : ni une occultation, ni une
  disparition, ni une purge technique ne diminuent ``occupancy_operational`` ;
- ``occupancy_uncertain`` dit seulement que l'observation de personnes déjà
  comptées est perdue ; ``opérationnel == observable + incertain`` est un
  invariant publié, jamais corrigé en silence ;
- une purge est ``purge_kind="technical"`` et ``is_exit=false`` : elle ne peut
  pas produire un faux ``OUT`` ;
- les protections de la FSM restent en place : hésitation, demi-tour dans les
  deux sens, disparition avant confirmation, ancre au bord.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from conftest import (
    TEST_FPS,
    StubAppearance,
    identity_vectors_by_x,
    make_test_line,
    person_box,
)
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

LINE_Y = 300.0
OUTSIDE_Y = 250.0
INSIDE_Y = 450.0
BUFFER_INSIDE = 315.0
BUFFER_OUTSIDE = 285.0

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
        # Invariant publié : l'occupation confirmée EST l'opérationnelle, et
        # l'opérationnelle se décompose en observée + incertaine.
        assert self.manager.occupancy_confirmed == self.manager.occupancy_operational
        assert self.manager.occupancy_operational == (
            self.manager.occupancy_observed + self.manager.occupancy_uncertain
        ), "invariant : opérationnel == observé + incertain, à chaque frame"
        return self

    def steps(self, count, detections=()):
        for _ in range(count):
            self.step(detections)
        return self

    def warmup(self):
        return self.steps(5)


def det(track_id, y, x=150.0, confidence=0.9, height=200.0):
    return Detection(track_id, person_box(x, y, height=height), confidence)


def build(config, sink, appearance=None):
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
        line=make_test_line(inside_side=config.line.inside_side),
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


@pytest.fixture
def sim(test_config, sink, frame):
    return Sim(build(test_config, sink), frame)


def enter(sim, track_id=1, x=150.0, confidence=0.9):
    sim.steps(2, [det(track_id, OUTSIDE_Y, x=x, confidence=confidence)])
    sim.steps(2, [det(track_id, INSIDE_Y, x=x, confidence=confidence)])
    return sim


# ---------------------------------------------------------------------------
# L'occupation opérationnelle ne diminue que sur sortie confirmée
# ---------------------------------------------------------------------------
def test_short_occlusion_keeps_the_operational_occupancy(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    assert manager.occupancy_operational == 1

    sim.steps(4)
    assert manager.occupancy_operational == 1
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_uncertain == 0
    assert manager.total_out == 0
    assert sink.count("OUT") == 0

    sim.steps(3, [det(1, INSIDE_Y)])
    assert manager.occupancy_operational == 1
    assert manager.total_in == 1, "aucun double comptage après l'occultation"


def test_long_occlusion_moves_the_person_to_the_uncertain_part_only(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)

    sim.steps(25)  # au-delà de la grâce : purge puis oubli de la galerie
    assert manager.occupancy_operational == 1, "jamais retirée en silence"
    assert manager.occupancy_confirmed == 1, "aucune décrémentation sur purge"
    assert manager.occupancy_observed == 0
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (0, 1)
    assert manager.total_out == 0
    assert sink.count("OUT") == 0
    assert sink.count("AMBIGUOUS_CROSSING") == 0


def test_only_a_confirmed_exit_decreases_the_occupational_occupancy(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    assert manager.occupancy_operational == 1

    sim.steps(3, [det(1, OUTSIDE_Y)])
    assert manager.total_out == 1
    assert manager.occupancy_operational == 0
    assert manager.occupancy_uncertain == 0
    assert sink.of_type("OUT")[0]["reason"] == "crossing_confirmed"


def test_technical_purge_is_never_an_exit(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    sim.steps(25)
    purges = sink.of_type("PURGE")
    assert len(purges) == 1
    assert purges[0]["purge_kind"] == "technical"
    assert purges[0]["is_exit"] is False
    assert purges[0]["occupancy_impact"] == "uncertain"
    assert manager.total_out == 0


def test_operational_occupancy_is_published_in_snapshots(sim, sink):
    sim.warmup()
    enter(sim)
    sim.steps(20)
    snapshots = sink.of_type("OCCUPANCY_SNAPSHOT")
    assert snapshots
    for snapshot in snapshots:
        assert snapshot["occupancy_confirmed"] == snapshot["occupancy_operational"]
        assert (
            snapshot["occupancy_operational"]
            == snapshot["occupancy_observed"] + snapshot["occupancy_uncertain"]
        )
    assert snapshots[-1]["occupancy_operational"] == 1
    assert snapshots[-1]["occupancy_observed"] == 0


def test_reappearance_inside_restores_the_confirmed_part(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    sim.steps(13)                       # purge, dans la fenêtre de rétention
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_operational == 1

    sim.steps(3, [det(9, INSIDE_Y)])    # nouvel ID technique, même personne
    assert manager.occupancy_uncertain == 0
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_operational == 1
    assert manager.total_in == 1


def test_reappearance_outside_confirms_the_exit(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    sim.steps(4)
    sim.steps(2, [det(9, 265.0)])       # réapparition cohérente, côté sortie
    assert manager.total_out == 1
    assert manager.occupancy_operational == 0
    assert sink.of_type("OUT")[0]["reason"] == "recovery_outside_coherent"


def test_disappearance_alone_never_produces_an_out(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    for _ in range(40):
        sim.step()
    assert sink.count("OUT") == 0
    assert manager.total_out == 0
    assert manager.occupancy_operational == 1


# ---------------------------------------------------------------------------
# Audit : seuls les chemins FSM touchent les compteurs
# ---------------------------------------------------------------------------
def _mutations(source: str, counter: str) -> list[tuple[int, str]]:
    pattern = re.compile(rf"{re.escape(counter)}\s*[-+]=")
    return [
        (index + 1, line.strip())
        for index, line in enumerate(source.splitlines())
        if pattern.search(line)
    ]


def _method_span(source: str, name: str) -> tuple[int, int]:
    """Plage de lignes (1-indexée, borne haute exclue) d'une méthode.

    La signature peut s'étaler sur plusieurs lignes (paramètres, ``-> None:``) :
    on la traverse en suivant la profondeur de parenthèses avant de chercher la
    première ligne moins indentée, qui marque la fin du corps.
    """
    lines = source.splitlines()
    start = next(
        index for index, line in enumerate(lines) if re.match(rf"\s*def {name}\(", line)
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    depth = 0
    header_end = start
    for index in range(start, len(lines)):
        depth += lines[index].count("(") - lines[index].count(")")
        header_end = index
        if depth <= 0 and lines[index].rstrip().endswith(":"):
            break
    for index in range(header_end + 1, len(lines)):
        line = lines[index]
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            return start + 1, index + 1
    return start + 1, len(lines) + 1


def test_only_the_confirmed_fsm_actions_mutate_the_counters():
    """Audit verrouillé : un compteur ne bouge que dans sa fonction dédiée.

    - ``total_out`` : uniquement dans ``_count_out``, elle-même appelée par les
      seules actions FSM de sortie confirmée (``COMPTE_SORTIE``,
      ``CONFIRME_SORTIE_DIFFEREE``) et par ``RECUPERE_EXTERIEUR`` — une
      disparition, une expiration de grâce, une purge technique ou une
      libération de mémoire ne peuvent donc pas produire de ``OUT`` ;
    - ``total_in`` / ``total_new`` / ``initial_occupancy`` : même règle.
    """
    source = (REPO_ROOT / "src" / "occupancy_manager.py").read_text(encoding="utf-8")
    for counter, owner in (
        ("self.total_out", "_count_out"),
        ("self.total_in", "_count_in"),
        ("self.total_new", "_count_new"),
        ("self.initial_occupancy", "_apply_decision"),
    ):
        start, stop = _method_span(source, owner)
        for line_number, line in _mutations(source, counter):
            assert start <= line_number < stop, (
                f"{counter} est modifié hors de {owner}() (ligne {line_number}) : {line}"
            )


def test_no_other_module_mutates_the_counters():
    """Ni l'identité, ni le journal, ni le point d'entrée ne touchent aux compteurs.

    C'est le contrôle qui interdit qu'une purge, une libération de mémoire ou un
    chemin d'erreur de ``main`` fasse indirectement baisser l'occupation.
    """
    for name in ("identity_manager.py", "main.py", "events.py", "occupancy_types.py"):
        source = (REPO_ROOT / "src" / name).read_text(encoding="utf-8")
        for counter in (
            "total_out", "total_in", "total_new", "initial_occupancy", "inside_occupancy",
        ):
            mutations = _mutations(source, counter)
            assert mutations == [], f"{name} modifie {counter} : {mutations}"


def test_no_counter_is_masked_by_a_clamp():
    """Aucun ``max(0, occupation)`` : un décompte faux doit être journalisé.

    Le contrôle porte sur l'arbre syntaxique, pas sur le texte : les docstrings
    qui mentionnent ``max(0, ...)`` ne sont pas des clamps.
    """
    import ast

    occupancy_tokens = ("occupancy", "total_in", "total_out", "total_new")
    for name in ("occupancy_manager.py", "occupancy_types.py", "main.py"):
        source = (REPO_ROOT / "src" / name).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "max" or not node.args:
                continue
            first = node.args[0]
            if not (
                isinstance(first, ast.Constant)
                and isinstance(first.value, (int, float))
                and first.value <= 0
            ):
                continue
            segment = ast.get_source_segment(source, node) or ""
            assert not any(token in segment for token in occupancy_tokens), (
                f"{name}:{node.lineno} borne l'occupation au lieu de signaler "
                f"l'incohérence : {segment}"
            )


def test_purge_paths_cannot_increment_total_out():
    """Vérification comportementale : purge ↔ sortie sont deux chemins disjoints."""
    source = (REPO_ROOT / "src" / "occupancy_manager.py").read_text(encoding="utf-8")
    start, stop = _method_span(source, "_expire")
    expiry = "\n".join(source.splitlines()[start - 1: stop - 1])
    assert "total_out" not in expiry
    assert "total_in" not in expiry
    assert "total_new" not in expiry
    assert "_count_out" not in expiry


def test_confirmed_occupancy_is_the_operational_ledger(manager):
    """La figure publiée est bien le bilan, et l'observée en est une lecture."""
    manager.initial_occupancy = 3
    manager.total_in = 2
    manager.total_new = 1
    manager.total_out = 4
    assert manager.occupancy_confirmed == 2
    assert manager.occupancy_operational == 2
    assert manager.occupancy_observed == 0
    assert manager.occupancy_range == (0, 2)


def test_a_confirmed_exit_decrements_once_for_the_right_person(sim, sink):
    """Un ``OUT`` confirmé décrémente une seule fois, pour le bon ``person_id``."""
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0), det(2, OUTSIDE_Y, x=450.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0), det(2, INSIDE_Y, x=450.0)])
    assert manager.occupancy_operational == 2

    person_one = manager.identities.person_of(1)
    sim.steps(3, [det(1, OUTSIDE_Y, x=150.0), det(2, INSIDE_Y, x=450.0)])
    assert manager.total_out == 1, "une seule décrémentation"
    assert manager.occupancy_operational == 1
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_observed == 1
    out = sink.of_type("OUT")
    assert len(out) == 1
    assert out[0]["person_id"] == person_one
    assert out[0]["occupancy_after"] == 1

    # La seconde personne ressort : la décrémentation la concerne, elle seule.
    sim.steps(3, [det(2, OUTSIDE_Y, x=450.0)])
    assert manager.total_out == 2
    assert manager.occupancy_operational == 0
    assert manager.occupancy_range == (0, 0)


def test_id_change_without_reassociation_never_decrements(sim, sink):
    """Un changement d'ID technique non réassocié ne fait pas baisser l'occupation."""
    manager = sim.manager
    sim.warmup()
    enter(sim)
    assert manager.occupancy_confirmed == 1

    sim.steps(3)                                    # piste perdue
    sim.steps(4, [det(77, INSIDE_Y, x=150.0)])      # nouvel ID, réassociation possible
    assert manager.occupancy_confirmed == 1
    assert manager.total_out == 0
    assert manager.total_new == 0
    assert manager.occupancy_operational == 1


def test_ledger_mismatch_is_reported_not_hidden(manager, sink):
    manager.total_in = 2                     # deux entrées sans aucune piste
    manager._check_consistency(1.0, 10)
    assert [event["kind"] for event in sink.of_type("INCONSISTENT_STATE")] == [
        "occupancy_ledger_mismatch"
    ]


def test_negative_ledger_is_reported(manager, sink):
    manager.total_out = 3
    manager._check_consistency(1.0, 10)
    kinds = [event["kind"] for event in sink.of_type("INCONSISTENT_STATE")]
    assert kinds == ["negative_occupancy_before_clamp"]


# ---------------------------------------------------------------------------
# Faux comptages : hésitation, demi-tours
# ---------------------------------------------------------------------------
def test_hesitation_around_the_line_never_counts(sim, sink):
    manager = sim.manager
    sim.warmup()
    for _ in range(10):
        sim.step([det(1, BUFFER_OUTSIDE)])
        sim.step([det(1, BUFFER_INSIDE)])
    assert manager.total_in == 0 and manager.total_out == 0
    assert manager.occupancy_operational == 0
    assert sink.count("IN") == 0 and sink.count("OUT") == 0


def test_turnaround_at_entry_counts_nothing(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y)])
    for _ in range(4):
        sim.step([det(1, BUFFER_OUTSIDE)])
        sim.step([det(1, BUFFER_INSIDE)])   # franchissement amorcé
        sim.step([det(1, BUFFER_OUTSIDE)])  # demi-tour
        assert manager.tracks[1].pending_crossing is None
    assert manager.total_in == 0 and manager.total_out == 0
    assert manager.occupancy_operational == 0


def test_turnaround_at_exit_keeps_the_person_counted(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    assert manager.occupancy_operational == 1

    for _ in range(4):
        sim.step([det(1, BUFFER_INSIDE)])
        sim.step([det(1, BUFFER_OUTSIDE)])  # sortie amorcée
        sim.step([det(1, INSIDE_Y)])        # demi-tour
        assert manager.tracks[1].pending_crossing is None
    assert manager.total_out == 0
    assert manager.occupancy_operational == 1
    assert sink.count("OUT") == 0


def test_disappearance_before_the_exit_confirmation_is_ambiguous(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    sim.step([det(1, LINE_Y)])
    sim.step([det(1, BUFFER_OUTSIDE)])
    assert manager.tracks[1].state.name == "SORTIE_EN_COURS"

    sim.steps(12)  # disparition au-delà de la grâce
    assert manager.total_out == 0
    assert manager.occupancy_operational == 1, "la personne reste comptée"
    assert manager.occupancy_uncertain == 1
    assert sink.count("AMBIGUOUS_CROSSING") == 1


# ---------------------------------------------------------------------------
# Flou pendant une entrée / une sortie
# ---------------------------------------------------------------------------
def test_blurred_entry_is_counted_once(sim, sink):
    """0,15 > model.confidence (0,10) : la personne n'est plus « invisible »."""
    manager = sim.manager
    sim.warmup()
    enter(sim, confidence=0.15)
    assert manager.total_in == 1
    assert manager.total_new == 0
    assert manager.occupancy_confirmed == 1
    assert sink.count("REID_DESCRIPTOR_REJECTED") >= 1


def test_blurred_exit_is_counted_once(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    sim.steps(3, [det(1, OUTSIDE_Y, confidence=0.15)])
    assert manager.total_out == 1
    assert manager.total_in == 1
    assert manager.occupancy_operational == 0


def test_two_people_side_by_side_are_counted_separately(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0), det(2, OUTSIDE_Y, x=450.0)])
    sim.steps(3, [det(1, INSIDE_Y, x=150.0), det(2, INSIDE_Y, x=450.0)])
    assert manager.total_in == 2
    assert manager.occupancy_operational == 2
    assert len({manager.identities.person_of(1), manager.identities.person_of(2)}) == 2
    assert sink.count("INCONSISTENT_STATE") == 0


# ---------------------------------------------------------------------------
# Garde-fous de NEW (défense en profondeur, en plus des règles de la FSM)
# ---------------------------------------------------------------------------
def test_new_is_refused_while_a_crossing_is_pending(manager, sink):
    from occupancy_types import PersonTrack, TrackState

    track = PersonTrack(person_id=7, state=TrackState.EXTERIEUR, pending_crossing="in")
    manager._count_new(track, 3, 1.0, 10)
    assert manager.total_new == 0
    assert manager.occupancy_operational == 0
    assert sink.of_type("INCONSISTENT_STATE")[0]["kind"] == "new_with_pending_crossing"


def test_new_is_refused_after_an_exterior_observation(manager, sink):
    from occupancy_types import PersonTrack, TrackState

    track = PersonTrack(
        person_id=8, state=TrackState.EXTERIEUR, observed_outside=True
    )
    manager._count_new(track, 4, 1.0, 10)
    assert manager.total_new == 0
    assert sink.of_type("INCONSISTENT_STATE")[0]["kind"] == (
        "new_after_exterior_observation"
    )


def test_new_is_refused_near_an_occluded_counted_person(manager, sink):
    from occupancy_types import PersonTrack, TrackState

    # Échelle locale connue : le rayon d'ambiguïté vaut 0,5 × 200 = 100 px.
    manager.scale.observe(200.0)
    manager.tracks[1] = PersonTrack(
        person_id=1,
        state=TrackState.OCCULTEE,
        inside_occupancy=True,
        anchor=(300.0, 450.0),
        last_seen_s=0.0,
    )
    candidate = PersonTrack(person_id=2, state=TrackState.EXTERIEUR, anchor=(305.0, 455.0))
    manager._count_new(candidate, 5, 1.0, 10)
    assert manager.total_new == 0
    assert candidate.inside_occupancy is False
    event = sink.of_type("INCONSISTENT_STATE")[0]
    assert event["kind"] == "new_with_occluded_candidates"
    assert "1" in event["details"]


def test_new_is_allowed_when_the_occluded_person_is_far_away(manager, sink):
    from occupancy_types import PersonTrack, TrackState

    manager.scale.observe(200.0)
    manager.tracks[1] = PersonTrack(
        person_id=1,
        state=TrackState.OCCULTEE,
        inside_occupancy=True,
        anchor=(300.0, 450.0),
        last_seen_s=0.0,
    )
    candidate = PersonTrack(person_id=2, state=TrackState.EXTERIEUR, anchor=(560.0, 450.0))
    manager._count_new(candidate, 5, 1.0, 10)
    assert manager.total_new == 1
    assert candidate.inside_occupancy is True
    assert sink.count("INCONSISTENT_STATE") == 0


# ---------------------------------------------------------------------------
# Réapparition cohérente : confirmation différée du franchissement
# ---------------------------------------------------------------------------
def test_deferred_exit_confirmation_after_a_short_occlusion(sim, sink):
    """Sortie amorcée puis disparition courte : la réapparition confirme l'OUT."""
    manager = sim.manager
    sim.warmup()
    enter(sim)
    sim.step([det(1, LINE_Y)])
    sim.step([det(1, BUFFER_OUTSIDE)])
    assert manager.tracks[1].state.name == "SORTIE_EN_COURS"
    assert manager.tracks[1].pending_crossing == "out"

    sim.steps(4)                       # disparition courte, dans la grâce
    sim.steps(2, [det(9, 265.0)])      # réapparition cohérente côté sortie
    assert manager.total_out == 1
    assert sink.of_type("OUT")[0]["reason"] == "deferred_confirmation"
    assert manager.occupancy_operational == 0
    assert sink.count("AMBIGUOUS_CROSSING") == 0


def test_anchor_at_the_image_edge_is_suspended_then_recovered(sim, sink):
    manager = sim.manager
    sim.warmup()
    sim.steps(2, [det(1, OUTSIDE_Y)])
    touching = Detection(1, np.array([0.0, 250.0, 80.0, 400.0]), 0.9)
    sim.steps(2, [touching])
    assert manager.total_in == 0
    assert sink.of_type("ANCHOR_UNRELIABLE")[0]["reason"] == "edge_margin"
    sim.steps(2, [det(1, INSIDE_Y)])
    assert manager.total_in == 1
    assert manager.occupancy_operational == 1
