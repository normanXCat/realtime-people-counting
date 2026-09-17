"""Niveau 1 — Machine à états : **un test par ligne** de la table de transition.

Couvre la spec 4.1 (états), 4.2 (table complète, symétrie des demi-tours,
traitement explicite de ``EN_ZONE_MORTE``), 4.3 (disparition en cours de
franchissement) et 5.4 (suspension sur ancre non fiable).

Le test :func:`test_every_rule_has_a_test` échoue si une ligne de
:data:`fsm.TRANSITION_TABLE` est ajoutée sans contexte de test associé, ce qui
empêche une ligne de comportement de rester non vérifiée.
"""

from __future__ import annotations

import pytest

from fsm import (
    CONDITIONS,
    RULES_BY_NAME,
    TRANSITION_TABLE,
    Action,
    Approach,
    FsmContext,
    StateMachine,
    TrackState,
)
from geometry import Side, Zone

S, A = TrackState, Action

BASE = dict(
    observed=True,
    zone=Zone.INTERIEURE,
    side=Side.INTERIEURE,
    crossed=None,
    anchor_reliable=True,
    scale_available=True,
    lost_frames=0,
    grace_period_frames=10,
    interior_streak=0,
    confirmation_frames=2,
    warmup_done=True,
    warmup_stable=False,
    approach=Approach.INTERIEUR,
    pending_crossing=None,
    in_progress_policy="deferred_confirmation",
)


def ctx(**overrides) -> FsmContext:
    return FsmContext(**{**BASE, **overrides})


LOST = dict(observed=False, zone=None, side=Side.INDETERMINEE, interior_streak=0)
EXPIRED = dict(LOST, lost_frames=11)
RECOVERED = dict(lost_frames=3)

#: nom de règle -> (état source, contexte, état cible attendu, action attendue)
CASES: dict[str, tuple[TrackState, FsmContext, TrackState, Action]] = {
    # -- Sûreté (spec 5.4) -------------------------------------------------
    "suspend_unreliable_anchor": (
        S.PRESENTE,
        ctx(anchor_reliable=False),
        S.PRESENTE,
        A.SUSPEND_DECISION,
    ),
    # -- INITIALISATION (spec 5.8) ----------------------------------------
    "warmup_place_initial_inside": (
        S.INITIALISATION,
        ctx(warmup_stable=True, side=Side.INTERIEURE),
        S.PRESENTE,
        A.EMPLACE_INITIAL,
    ),
    "warmup_stable_outside": (
        S.INITIALISATION,
        ctx(warmup_stable=True, side=Side.EXTERIEURE, zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.AUCUNE,
    ),
    "warmup_exclude_unstable": (
        S.INITIALISATION,
        ctx(warmup_stable=False),
        S.EXTERIEUR,
        A.AUCUNE,
    ),
    # -- EXTERIEUR ---------------------------------------------------------
    "outside_cross_confirms_entry": (
        S.EXTERIEUR,
        ctx(crossed="in", zone=Zone.INTERIEURE),
        S.PRESENTE,
        A.COMPTE_ENTREE,
    ),
    "outside_cross_commits_entry": (
        S.EXTERIEUR,
        ctx(crossed="in", zone=Zone.MORTE),
        S.ENTREE_EN_COURS,
        A.AMORCE_ENTREE,
    ),
    "outside_enter_dead_zone": (
        S.EXTERIEUR,
        ctx(zone=Zone.MORTE, approach=Approach.INTERIEUR),
        S.EN_ZONE_MORTE,
        A.MARCHE_ZONE_MORTE_DEPUIS_EXTERIEUR,
    ),
    "outside_new_presence_confirmed": (
        S.EXTERIEUR,
        ctx(zone=Zone.INTERIEURE, interior_streak=2),
        S.PRESENTE,
        A.NOUVELLE_PRESENCE,
    ),
    "outside_lost": (S.EXTERIEUR, ctx(**LOST), S.OCCULTEE, A.MARQUE_OCCULTEE),
    # -- PRESENTE ----------------------------------------------------------
    "inside_cross_confirms_exit": (
        S.PRESENTE,
        ctx(crossed="out", zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.COMPTE_SORTIE,
    ),
    "inside_cross_commits_exit": (
        S.PRESENTE,
        ctx(crossed="out", zone=Zone.MORTE),
        S.SORTIE_EN_COURS,
        A.AMORCE_SORTIE,
    ),
    "inside_enter_dead_zone": (
        S.PRESENTE,
        ctx(zone=Zone.MORTE),
        S.EN_ZONE_MORTE,
        A.MARCHE_ZONE_MORTE_DEPUIS_INTERIEUR,
    ),
    "inside_drift_outside_without_crossing": (
        S.PRESENTE,
        ctx(zone=Zone.EXTERIEURE, crossed=None),
        S.SORTIE_EN_COURS,
        A.SIGNALE_DERIVE,
    ),
    "inside_lost": (S.PRESENTE, ctx(**LOST), S.OCCULTEE, A.MARQUE_OCCULTEE),
    # -- EN_ZONE_MORTE (spec 4.2) -----------------------------------------
    "dead_zone_cross_confirms_exit": (
        S.EN_ZONE_MORTE,
        ctx(crossed="out", zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.COMPTE_SORTIE,
    ),
    "dead_zone_cross_confirms_entry": (
        S.EN_ZONE_MORTE,
        ctx(crossed="in", zone=Zone.INTERIEURE),
        S.PRESENTE,
        A.COMPTE_ENTREE,
    ),
    "dead_zone_cross_commits_exit": (
        S.EN_ZONE_MORTE,
        ctx(crossed="out", zone=Zone.MORTE),
        S.SORTIE_EN_COURS,
        A.AMORCE_SORTIE,
    ),
    "dead_zone_cross_commits_entry": (
        S.EN_ZONE_MORTE,
        ctx(crossed="in", zone=Zone.MORTE),
        S.ENTREE_EN_COURS,
        A.AMORCE_ENTREE,
    ),
    "dead_zone_return_to_inside": (
        S.EN_ZONE_MORTE,
        ctx(zone=Zone.INTERIEURE, approach=Approach.INTERIEUR),
        S.PRESENTE,
        A.AUCUNE,
    ),
    "dead_zone_return_to_outside": (
        S.EN_ZONE_MORTE,
        ctx(zone=Zone.EXTERIEURE, approach=Approach.EXTERIEUR),
        S.EXTERIEUR,
        A.AUCUNE,
    ),
    "dead_zone_commit_exit": (
        S.EN_ZONE_MORTE,
        ctx(zone=Zone.EXTERIEURE, approach=Approach.INTERIEUR, crossed=None),
        S.SORTIE_EN_COURS,
        A.AMORCE_SORTIE,
    ),
    "dead_zone_commit_entry": (
        S.EN_ZONE_MORTE,
        ctx(zone=Zone.INTERIEURE, approach=Approach.EXTERIEUR, crossed=None),
        S.ENTREE_EN_COURS,
        A.AMORCE_ENTREE,
    ),
    "dead_zone_lost": (S.EN_ZONE_MORTE, ctx(**LOST), S.OCCULTEE, A.MARQUE_OCCULTEE),
    # -- ENTREE_EN_COURS ---------------------------------------------------
    "entry_confirmed": (S.ENTREE_EN_COURS, ctx(zone=Zone.INTERIEURE), S.PRESENTE, A.COMPTE_ENTREE),
    "entry_turnaround_outside": (
        S.ENTREE_EN_COURS,
        ctx(zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.ANNULE_FRANCHISSEMENT,
    ),
    "entry_turnaround_crossed_out": (
        S.ENTREE_EN_COURS,
        ctx(zone=Zone.MORTE, crossed="out"),
        S.EXTERIEUR,
        A.ANNULE_FRANCHISSEMENT,
    ),
    "entry_lost_immediate_policy": (
        S.ENTREE_EN_COURS,
        ctx(**LOST, pending_crossing="in", in_progress_policy="ambiguous_immediate"),
        S.ABSENTE,
        A.EXPIRE_AMBIGU,
    ),
    "entry_lost_in_progress": (
        S.ENTREE_EN_COURS,
        ctx(**LOST, pending_crossing="in"),
        S.OCCULTEE,
        A.MARQUE_OCCULTEE,
    ),
    # -- SORTIE_EN_COURS ---------------------------------------------------
    "exit_confirmed": (S.SORTIE_EN_COURS, ctx(zone=Zone.EXTERIEURE), S.EXTERIEUR, A.COMPTE_SORTIE),
    "exit_turnaround_inside": (
        S.SORTIE_EN_COURS,
        ctx(zone=Zone.INTERIEURE),
        S.PRESENTE,
        A.ANNULE_FRANCHISSEMENT,
    ),
    "exit_turnaround_crossed_in": (
        S.SORTIE_EN_COURS,
        ctx(zone=Zone.MORTE, crossed="in"),
        S.PRESENTE,
        A.ANNULE_FRANCHISSEMENT,
    ),
    "exit_lost_immediate_policy": (
        S.SORTIE_EN_COURS,
        ctx(**LOST, pending_crossing="out", in_progress_policy="ambiguous_immediate"),
        S.ABSENTE,
        A.EXPIRE_AMBIGU,
    ),
    "exit_lost_in_progress": (
        S.SORTIE_EN_COURS,
        ctx(**LOST, pending_crossing="out"),
        S.OCCULTEE,
        A.MARQUE_OCCULTEE,
    ),
    # -- OCCULTEE ----------------------------------------------------------
    "occluded_pending_in_confirmed": (
        S.OCCULTEE,
        ctx(**RECOVERED, pending_crossing="in", zone=Zone.INTERIEURE),
        S.PRESENTE,
        A.CONFIRME_ENTREE_DIFFEREE,
    ),
    "occluded_pending_in_cancelled": (
        S.OCCULTEE,
        ctx(**RECOVERED, pending_crossing="in", zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.ANNULE_FRANCHISSEMENT,
    ),
    "occluded_pending_out_confirmed": (
        S.OCCULTEE,
        ctx(**RECOVERED, pending_crossing="out", zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.CONFIRME_SORTIE_DIFFEREE,
    ),
    "occluded_pending_out_cancelled": (
        S.OCCULTEE,
        ctx(**RECOVERED, pending_crossing="out", zone=Zone.INTERIEURE),
        S.PRESENTE,
        A.ANNULE_FRANCHISSEMENT,
    ),
    "occluded_pending_in_dead_zone": (
        S.OCCULTEE,
        ctx(**RECOVERED, pending_crossing="in", zone=Zone.MORTE),
        S.EN_ZONE_MORTE,
        A.MARCHE_ZONE_MORTE_DEPUIS_EXTERIEUR,
    ),
    "occluded_pending_out_dead_zone": (
        S.OCCULTEE,
        ctx(**RECOVERED, pending_crossing="out", zone=Zone.MORTE),
        S.EN_ZONE_MORTE,
        A.MARCHE_ZONE_MORTE_DEPUIS_INTERIEUR,
    ),
    "occluded_reappears_in_dead_zone": (
        S.OCCULTEE,
        ctx(**RECOVERED, zone=Zone.MORTE),
        S.EN_ZONE_MORTE,
        A.AUCUNE,
    ),
    "occluded_reappears_inside": (
        S.OCCULTEE,
        ctx(**RECOVERED, zone=Zone.INTERIEURE),
        S.PRESENTE,
        A.RECUPERE_INTERIEUR,
    ),
    "occluded_reappears_outside": (
        S.OCCULTEE,
        ctx(**RECOVERED, zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.RECUPERE_EXTERIEUR,
    ),
    "occluded_grace_expired": (
        S.OCCULTEE,
        ctx(**EXPIRED),
        S.ABSENTE,
        A.EXPIRE_HORS_CHAMP,
    ),
    "occluded_grace_expired_in_progress": (
        S.OCCULTEE,
        ctx(**EXPIRED, pending_crossing="in"),
        S.ABSENTE,
        A.EXPIRE_AMBIGU,
    ),
    # -- ABSENTE (reviviscence depuis la galerie, spec 3.3) ---------------
    "absent_revives_inside": (
        S.ABSENTE,
        ctx(**RECOVERED, zone=Zone.INTERIEURE),
        S.PRESENTE,
        A.RECUPERE_INTERIEUR,
    ),
    "absent_revives_outside": (
        S.ABSENTE,
        ctx(**RECOVERED, zone=Zone.EXTERIEURE),
        S.EXTERIEUR,
        A.RECUPERE_EXTERIEUR,
    ),
    "absent_revives_dead_zone": (
        S.ABSENTE,
        ctx(**RECOVERED, zone=Zone.MORTE),
        S.EN_ZONE_MORTE,
        A.AUCUNE,
    ),
}

MACHINE = StateMachine()


def test_table_is_coherent():
    assert len(TRANSITION_TABLE) == len(RULES_BY_NAME)
    assert all(rule.condition in CONDITIONS for rule in TRANSITION_TABLE)
    assert len({rule.name for rule in TRANSITION_TABLE}) == len(TRANSITION_TABLE)


def test_every_rule_has_a_test():
    missing = sorted(set(RULES_BY_NAME) - set(CASES))
    obsolete = sorted(set(CASES) - set(RULES_BY_NAME))
    assert not missing, f"Règles sans test : {missing}"
    assert not obsolete, f"Tests sans règle : {obsolete}"


def test_absent_is_reachable_and_has_outgoing_rules():
    targets = {rule.state_to for rule in TRANSITION_TABLE}
    assert TrackState.ABSENTE in targets
    assert any(rule.state_from is TrackState.ABSENTE for rule in TRANSITION_TABLE)


@pytest.mark.parametrize("rule_name", sorted(CASES))
def test_transition_rule(rule_name):
    state_from, context, state_to, action = CASES[rule_name]
    decision = MACHINE.resolve(state_from, context)
    assert decision.rule == rule_name, (
        f"{rule_name} n'est pas la règle retenue (obtenu {decision.rule})"
    )
    assert decision.state_to is state_to
    assert decision.action is action


def test_no_rule_matched_leaves_state_unchanged():
    # Personne stable à l'intérieur, ni franchissement ni zone morte.
    context = ctx(zone=Zone.INTERIEURE, crossed=None, interior_streak=5)
    decision = MACHINE.resolve(TrackState.PRESENTE, context)
    assert decision.rule == "no_rule_matched"
    assert decision.state_to is TrackState.PRESENTE
    assert decision.action is Action.AUCUNE
    assert decision.state_changed is False


def test_suspension_short_circuits_even_grace_expiry():
    context = ctx(**EXPIRED, anchor_reliable=False)
    context = FsmContext(**{**context.__dict__, "observed": True})
    decision = MACHINE.resolve(TrackState.OCCULTEE, context)
    assert decision.action is Action.SUSPEND_DECISION
    assert decision.state_to is TrackState.OCCULTEE


def test_state_changed_flag_reports_transitions():
    decision = MACHINE.resolve(
        TrackState.PRESENTE, ctx(crossed="out", zone=Zone.EXTERIEURE)
    )
    assert decision.state_changed is True
    decision = MACHINE.resolve(
        TrackState.EN_ZONE_MORTE, ctx(crossed="out", zone=Zone.MORTE)
    )
    assert decision.state_changed is True
    decision = MACHINE.resolve(
        TrackState.PRESENTE, ctx(zone=Zone.INTERIEURE, crossed=None, interior_streak=5)
    )
    assert decision.state_changed is False
