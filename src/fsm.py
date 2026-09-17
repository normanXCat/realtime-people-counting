"""Machine à états de suivi d'une personne, sous forme de table explicite.

La table :data:`TRANSITION_TABLE` est **la** définition du comportement du
système : chaque ligne porte un nom, un état source, une condition nommée, un
état cible et une action. Il y a un test unitaire par ligne
(`tests/test_fsm.py` vérifie en plus qu'aucune ligne n'est sans test).

Corrections apportées par rapport à l'audit (§2.2 c, h, i et spec 4.2) :

- la logique n'est plus dispersée dans ``process_frame`` : elle est déclarative ;
- les états ``EN_ZONE_MORTE`` et ``ABSENTE`` existent réellement et sont
  atteints, contrairement à l'ancien ``EN_ZONE_LIGNE`` jamais assigné ;
- la symétrie des demi-tours est présente côté entrée **et** côté sortie ;
- un franchissement rapide (segment qui traverse la ligne et aboutit au-delà de
  la zone morte) est confirmé **dans la même frame** : pas de comptage perdu si
  la personne disparaît juste après ;
- une disparition **pendant** un franchissement est traitée par des lignes
  dédiées, avec la politique ``in_progress_disappearance_policy`` :
  ``deferred_confirmation`` (défaut) attend la réapparition puis, à l'expiration,
  émet ``AMBIGUOUS_CROSSING`` sans compter ; ``ambiguous_immediate`` tranche dès
  la disparition en déclarant l'ambiguïté ;
- une dérive de demi-plan sans intersection capturée (ligne non couvrante) émet
  ``INCONSISTENT_STATE`` au lieu d'être ignorée ou tranchée silencieusement ;
- ``NEW`` est strictement réservé à une personne **jamais observée à
  l'extérieur** : une piste vue dehors puis retrouvée loin à l'intérieur sans
  intersection capturée produit une entrée signalée (``ENTREE_EN_COURS``), pas
  une nouvelle présence — sinon une personne qui entre réellement serait
  comptée comme une apparition intérieure ;
- une apparition intérieure stable est **différée** tant qu'une personne déjà
  comptée est en occultation à proximité : le ``NEW`` peut être un doublon de
  cette personne, et l'ambiguïté est journalisée plutôt que tranchée.

Sémantique de résolution : la **première** ligne dont l'état source correspond
et dont la condition est vraie l'emporte. La ligne de suspension (spec 5.4) est
placée en tête et court-circuite toute décision de franchissement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Mapping

from geometry import Side, Zone


class TrackState(Enum):
    """États de la machine à états (spec 4.1)."""

    INITIALISATION = auto()   # phase de warm-up, aucun comptage
    EXTERIEUR = auto()        # côté sortie, hors de la zone morte
    PRESENTE = auto()         # à l'intérieur, présence établie
    EN_ZONE_MORTE = auto()    # dans la bande tampon, direction non engagée
    ENTREE_EN_COURS = auto()  # franchissement amorcé vers l'intérieur
    SORTIE_EN_COURS = auto()  # franchissement amorcé vers l'extérieur
    OCCULTEE = auto()         # piste perdue, dans la fenêtre de grâce
    ABSENTE = auto()          # grâce expirée, durablement hors champ


class Action(Enum):
    """Actions déclenchées par une transition. Aucune n'est facultative."""

    AUCUNE = auto()                  # transition d'état pure
    EMPLACE_INITIAL = auto()          # effectif initial du warm-up (spec 5.8)
    COMPTE_ENTREE = auto()            # IN confirmé
    COMPTE_SORTIE = auto()            # OUT confirmé
    NOUVELLE_PRESENCE = auto()        # NEW : apparition intérieure confirmée
    ANNULE_FRANCHISSEMENT = auto()    # demi-tour : franchissement amorcé annulé
    SIGNALE_DERIVE = auto()           # demi-plan changé sans intersection capturée
    AMORCE_ENTREE = auto()            # franchissement entrant amorcé (mémoire conservée)
    AMORCE_SORTIE = auto()            # franchissement sortant amorcé (mémoire conservée)
    MARQUE_OCCULTEE = auto()          # émission OCCLUDED, franchissement en l'état
    RECUPERE_INTERIEUR = auto()       # réapparition côté intérieur
    RECUPERE_EXTERIEUR = auto()       # réapparition côté extérieur
    CONFIRME_ENTREE_DIFFEREE = auto() # IN confirmé après réapparition (spec 4.3)
    CONFIRME_SORTIE_DIFFEREE = auto() # OUT confirmé après réapparition (spec 4.3)
    EXPIRE_HORS_CHAMP = auto()        # purge sans franchissement amorcé
    EXPIRE_AMBIGU = auto()            # purge avec AMBIGUOUS_CROSSING (spec 4.3)
    MARCHE_ZONE_MORTE_DEPUIS_INTERIEUR = auto()
    MARCHE_ZONE_MORTE_DEPUIS_EXTERIEUR = auto()
    SUSPEND_DECISION = auto()         # ancre non fiable : aucune décision (spec 5.4)
    SIGNALE_ENTREE_SANS_FRANCHISSEMENT = auto()  # changement de côté sans intersection
    DIFFERE_NOUVELLE_PRESENCE = auto()  # NEW retenu : personne comptée en occultation


class Approach(Enum):
    """D'où venait la personne en entrant dans la zone morte."""

    INTERIEUR = auto()
    EXTERIEUR = auto()


@dataclass(frozen=True)
class FsmContext:
    """Tout ce dont la table a besoin pour décider, pour une frame et une personne."""

    observed: bool = False
    """La piste technique est-elle visible sur cette frame ?"""

    zone: Zone | None = None
    """Zone hystérétique (``None`` si l'échelle locale est indisponible)."""

    side: Side = Side.INDETERMINEE
    """Demi-plan sans zone morte."""

    crossed: str | None = None
    """``"in"``, ``"out"`` ou ``None`` : la trajectoire coupe la ligne."""

    anchor_reliable: bool = True
    """Faux si la bbox touche un bord d'image (spec 5.4)."""

    scale_available: bool = True
    """Faux si la hauteur médiane des bbox n'est pas encore connue."""

    lost_frames: int = 0
    grace_period_frames: int = 150
    interior_streak: int = 0
    confirmation_frames: int = 15

    warmup_done: bool = True
    warmup_stable: bool = False
    """Stable en position et sans décision de franchissement claire (spec 5.8)."""

    exterior_history: bool = False
    """La piste a-t-elle déjà été observée dans la zone **extérieure** ?

    C'est la condition qui interdit ``NEW`` : une personne vue dehors ne peut
    pas « apparaître » à l'intérieur, elle y entre (franchissement capturé ou
    entrée signalée). Sans ce drapeau, une piste perdue puis réapparue de
    l'autre côté de la ligne produirait un faux ``NEW`` en plus de son ``IN``.
    """

    new_blocked_by_occlusion: bool = False
    """Une personne déjà comptée est en occultation à proximité de cette piste.

    Une apparition intérieure stable ne peut alors pas être confirmée comme
    nouvelle : le ``NEW`` serait possiblement un doublon de la personne occultée.
    """

    approach: Approach = Approach.INTERIEUR
    pending_crossing: str | None = None
    """``"in"`` ou ``"out"`` : franchissement amorcé avant la disparition."""

    in_progress_policy: str = "deferred_confirmation"

    @property
    def lost(self) -> bool:
        return not self.observed

    @property
    def grace_expired(self) -> bool:
        return self.lost_frames > self.grace_period_frames


@dataclass(frozen=True)
class TransitionRule:
    """Une ligne de la table de transition."""

    name: str
    state_from: TrackState | None   # ``None`` : s'applique à tous les états
    condition: str
    state_to: TrackState | None     # ``None`` : l'état est inchangé
    action: Action
    comment: str = ""


@dataclass(frozen=True)
class Decision:
    """Résultat de la résolution de la table pour une frame."""

    rule: str
    state_to: TrackState
    action: Action
    state_changed: bool


# ---------------------------------------------------------------------------
# Conditions (une fonction par nom, testables isolément)
# ---------------------------------------------------------------------------
def _far_inside(ctx: FsmContext) -> bool:
    return ctx.zone is Zone.INTERIEURE


def _far_outside(ctx: FsmContext) -> bool:
    return ctx.zone is Zone.EXTERIEURE


def _in_dead_zone(ctx: FsmContext) -> bool:
    return ctx.zone is Zone.MORTE


def _crossed(ctx: FsmContext, direction: str) -> bool:
    return ctx.crossed == direction


def _approach_inside(ctx: FsmContext) -> bool:
    return ctx.approach is Approach.INTERIEUR


def _immediate_policy(ctx: FsmContext) -> bool:
    return ctx.in_progress_policy == "ambiguous_immediate"


def _has_pending(ctx: FsmContext, direction: str) -> bool:
    return ctx.pending_crossing == direction


CONDITIONS: dict[str, Callable[[FsmContext], bool]] = {
    # -- Sûreté (spec 5.4) ------------------------------------------------
    "observed_and_unreliable_anchor": lambda ctx: ctx.observed
    and (not ctx.anchor_reliable or not ctx.scale_available),
    # -- Warm-up (spec 5.8) ----------------------------------------------
    "warmup_done_and_stable_inside": lambda ctx: ctx.warmup_done
    and ctx.warmup_stable
    and ctx.side is Side.INTERIEURE,
    "warmup_done_and_stable_outside": lambda ctx: ctx.warmup_done
    and ctx.warmup_stable
    and ctx.side is Side.EXTERIEURE,
    "warmup_done_and_unstable": lambda ctx: ctx.warmup_done and not ctx.warmup_stable,
    # -- Nouvelle présence (restreinte, spec 4 du prompt) ------------------
    "interior_stable_blocked_by_occlusion": lambda ctx: ctx.interior_streak
    >= max(1, ctx.confirmation_frames)
    and ctx.new_blocked_by_occlusion,
    "interior_stable_with_exterior_history": lambda ctx: ctx.interior_streak
    >= max(1, ctx.confirmation_frames)
    and ctx.exterior_history,
    "interior_stable_without_exterior_history": lambda ctx: ctx.interior_streak
    >= max(1, ctx.confirmation_frames)
    and not ctx.exterior_history,
    # -- Perte de piste ---------------------------------------------------
    "lost": lambda ctx: ctx.lost,
    "grace_expired_without_pending": lambda ctx: ctx.grace_expired
    and ctx.pending_crossing is None,
    "grace_expired_with_pending": lambda ctx: ctx.grace_expired
    and ctx.pending_crossing is not None,
    "lost_with_pending_and_immediate_policy": lambda ctx: ctx.lost
    and ctx.pending_crossing is not None
    and _immediate_policy(ctx),
    # -- Zones -------------------------------------------------------------
    "in_dead_zone": _in_dead_zone,
    "far_inside": _far_inside,
    "far_outside": _far_outside,
    # -- Franchissements ---------------------------------------------------
    "crossed_in_and_far_inside": lambda ctx: _crossed(ctx, "in") and _far_inside(ctx),
    "crossed_out_and_far_outside": lambda ctx: _crossed(ctx, "out") and _far_outside(ctx),
    "crossed_in_in_buffer": lambda ctx: _crossed(ctx, "in") and _in_dead_zone(ctx),
    "crossed_out_in_buffer": lambda ctx: _crossed(ctx, "out") and _in_dead_zone(ctx),
    "crossed_toward_inside": lambda ctx: _crossed(ctx, "in"),
    "crossed_toward_outside": lambda ctx: _crossed(ctx, "out"),
    # -- Zone morte --------------------------------------------------------
    "approach_inside_and_far_inside": lambda ctx: _approach_inside(ctx) and _far_inside(ctx),
    "approach_outside_and_far_outside": lambda ctx: not _approach_inside(ctx)
    and _far_outside(ctx),
    "approach_inside_and_far_outside": lambda ctx: _approach_inside(ctx) and _far_outside(ctx),
    "approach_outside_and_far_inside": lambda ctx: not _approach_inside(ctx)
    and _far_inside(ctx),

    # -- Réapparition ------------------------------------------------------
    "recovered": lambda ctx: ctx.observed and ctx.lost_frames > 0,
    "recovered_pending_in_confirmed": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _has_pending(ctx, "in")
    and (_far_inside(ctx) or _crossed(ctx, "in")),
    "recovered_pending_in_cancelled": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _has_pending(ctx, "in")
    and (_far_outside(ctx) or _crossed(ctx, "out")),
    "recovered_pending_out_confirmed": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _has_pending(ctx, "out")
    and (_far_outside(ctx) or _crossed(ctx, "out")),
    "recovered_pending_out_cancelled": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _has_pending(ctx, "out")
    and (_far_inside(ctx) or _crossed(ctx, "in")),
    "recovered_pending_in_dead_zone": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _has_pending(ctx, "in")
    and _in_dead_zone(ctx),
    "recovered_pending_out_dead_zone": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _has_pending(ctx, "out")
    and _in_dead_zone(ctx),
    "recovered_without_pending_in_dead_zone": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and ctx.pending_crossing is None
    and _in_dead_zone(ctx),
    "recovered_and_far_inside": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _far_inside(ctx),
    "recovered_and_far_outside": lambda ctx: ctx.observed
    and ctx.lost_frames > 0
    and _far_outside(ctx),
}


# ---------------------------------------------------------------------------
# Table de transition (spec 4.2)
# ---------------------------------------------------------------------------
S, A = TrackState, Action

TRANSITION_TABLE: tuple[TransitionRule, ...] = (
    # --- Sûreté : aucune décision sur une ancre non fiable (spec 5.4) -----
    # Ordre de priorité effectif de la table (spec 8 du prompt) :
    #   1. ancre non fiable / échelle indisponible (suspension, court-circuit) ;
    #   2. warm-up (état INITIALISATION) ;
    #   3. perte de piste ;
    #   4. franchissement confirmé ;
    #   5. franchissement amorcé ;
    #   6. demi-tour ;
    #   7. changement de zone / nouvelle présence ;
    #   8. aucune transition (état inchangé, aucune décision inventée).
    # ``tests/test_fsm.py::test_priority_order_matches_the_documented_intent``
    # verrouille cet ordre.
    TransitionRule(
        "suspend_unreliable_anchor", None,
        "observed_and_unreliable_anchor", None, A.SUSPEND_DECISION,
        "Ancre pieds au bord de l'image, chute anormale de hauteur ou échelle locale indisponible",
    ),
    # --- INITIALISATION (warm-up, spec 5.8) ------------------------------
    TransitionRule(
        "warmup_place_initial_inside", S.INITIALISATION,
        "warmup_done_and_stable_inside", S.PRESENTE, A.EMPLACE_INITIAL,
        "Piste stable à l'intérieur pendant tout le warm-up",
    ),
    TransitionRule(
        "warmup_stable_outside", S.INITIALISATION,
        "warmup_done_and_stable_outside", S.EXTERIEUR, A.AUCUNE,
        "Piste stable à l'extérieur : hors effectif initial",
    ),
    TransitionRule(
        "warmup_exclude_unstable", S.INITIALISATION,
        "warmup_done_and_unstable", S.EXTERIEUR, A.AUCUNE,
        "Piste non stable ou ayant amorcé un franchissement : exclue (spec 5.8)",
    ),
    # --- EXTERIEUR --------------------------------------------------------
    TransitionRule(
        "outside_cross_confirms_entry", S.EXTERIEUR,
        "crossed_in_and_far_inside", S.PRESENTE, A.COMPTE_ENTREE,
        "Franchissement rapide : confirmé dans la même frame",
    ),
    TransitionRule(
        "outside_cross_commits_entry", S.EXTERIEUR,
        "crossed_in_in_buffer", S.ENTREE_EN_COURS, A.AMORCE_ENTREE,
        "Ligne franchie mais encore dans la zone morte",
    ),
    TransitionRule(
        "outside_enter_dead_zone", S.EXTERIEUR,
        "in_dead_zone", S.EN_ZONE_MORTE, A.MARCHE_ZONE_MORTE_DEPUIS_EXTERIEUR,
    ),
    TransitionRule(
        "outside_new_presence_deferred_by_occlusion", S.EXTERIEUR,
        "interior_stable_blocked_by_occlusion", S.EXTERIEUR,
        A.DIFFERE_NOUVELLE_PRESENCE,
        "Apparition intérieure stable mais une personne comptée est occultée à "
        "proximité : NEW retenu et journalisé, jamais compté à l'aveugle",
    ),
    TransitionRule(
        "outside_entry_without_crossing_signalled", S.EXTERIEUR,
        "interior_stable_with_exterior_history", S.ENTREE_EN_COURS,
        A.SIGNALE_ENTREE_SANS_FRANCHISSEMENT,
        "Personne déjà observée dehors et retrouvée loin à l'intérieur sans "
        "intersection capturée : entrée signalée puis confirmée (jamais NEW)",
    ),
    TransitionRule(
        "outside_new_presence_confirmed", S.EXTERIEUR,
        "interior_stable_without_exterior_history", S.PRESENTE, A.NOUVELLE_PRESENCE,
        "Apparition directe à l'intérieur d'une personne jamais vue dehors",
    ),
    TransitionRule("outside_lost", S.EXTERIEUR, "lost", S.OCCULTEE, A.MARQUE_OCCULTEE),
    # --- PRESENTE ---------------------------------------------------------
    TransitionRule(
        "inside_cross_confirms_exit", S.PRESENTE,
        "crossed_out_and_far_outside", S.EXTERIEUR, A.COMPTE_SORTIE,
        "Franchissement rapide : confirmé dans la même frame",
    ),
    TransitionRule(
        "inside_cross_commits_exit", S.PRESENTE,
        "crossed_out_in_buffer", S.SORTIE_EN_COURS, A.AMORCE_SORTIE,
    ),
    TransitionRule(
        "inside_enter_dead_zone", S.PRESENTE,
        "in_dead_zone", S.EN_ZONE_MORTE, A.MARCHE_ZONE_MORTE_DEPUIS_INTERIEUR,
    ),
    TransitionRule(
        "inside_drift_outside_without_crossing", S.PRESENTE,
        "far_outside", S.SORTIE_EN_COURS, A.SIGNALE_DERIVE,
        "Demi-plan changé sans intersection capturée : signalé, confirmé ensuite",
    ),
    TransitionRule("inside_lost", S.PRESENTE, "lost", S.OCCULTEE, A.MARQUE_OCCULTEE),
    # --- EN_ZONE_MORTE (hystérésis, spec 4.2) ----------------------------
    TransitionRule(
        "dead_zone_cross_confirms_exit", S.EN_ZONE_MORTE,
        "crossed_out_and_far_outside", S.EXTERIEUR, A.COMPTE_SORTIE,
    ),
    TransitionRule(
        "dead_zone_cross_confirms_entry", S.EN_ZONE_MORTE,
        "crossed_in_and_far_inside", S.PRESENTE, A.COMPTE_ENTREE,
    ),
    TransitionRule(
        "dead_zone_cross_commits_exit", S.EN_ZONE_MORTE,
        "crossed_out_in_buffer", S.SORTIE_EN_COURS, A.AMORCE_SORTIE,
    ),
    TransitionRule(
        "dead_zone_cross_commits_entry", S.EN_ZONE_MORTE,
        "crossed_in_in_buffer", S.ENTREE_EN_COURS, A.AMORCE_ENTREE,
    ),
    # Ordre : franchissement amorcé (traversée complète du tampon) avant
    # demi-tour, conformément à la priorité annoncée en tête de table. Les
    # conditions sont mutuellement exclusives (approche + zone), l'ordre n'est
    # donc pas qu'une convention : il rend la priorité lisible et testable.
    TransitionRule(
        "dead_zone_commit_exit", S.EN_ZONE_MORTE,
        "approach_inside_and_far_outside", S.SORTIE_EN_COURS, A.AMORCE_SORTIE,
        "Traversée du tampon depuis l'intérieur sans signal de franchissement capturé",
    ),
    TransitionRule(
        "dead_zone_commit_entry", S.EN_ZONE_MORTE,
        "approach_outside_and_far_inside", S.ENTREE_EN_COURS, A.AMORCE_ENTREE,
        "Traversée du tampon depuis l'extérieur sans signal de franchissement capturé",
    ),
    TransitionRule(
        "dead_zone_return_to_inside", S.EN_ZONE_MORTE,
        "approach_inside_and_far_inside", S.PRESENTE, A.AUCUNE,
        "Demi-tour dans la zone morte, retour côté intérieur",
    ),
    TransitionRule(
        "dead_zone_return_to_outside", S.EN_ZONE_MORTE,
        "approach_outside_and_far_outside", S.EXTERIEUR, A.AUCUNE,
        "Demi-tour dans la zone morte, retour côté extérieur",
    ),
    TransitionRule(
        "dead_zone_lost", S.EN_ZONE_MORTE, "lost", S.OCCULTEE, A.MARQUE_OCCULTEE
    ),
    # --- ENTREE_EN_COURS -------------------------------------------------
    TransitionRule(
        "entry_confirmed", S.ENTREE_EN_COURS, "far_inside", S.PRESENTE, A.COMPTE_ENTREE
    ),
    TransitionRule(
        "entry_turnaround_outside", S.ENTREE_EN_COURS,
        "far_outside", S.EXTERIEUR, A.ANNULE_FRANCHISSEMENT,
        "Demi-tour côté entrée : le franchissement amorcé est annulé",
    ),
    TransitionRule(
        "entry_turnaround_crossed_out", S.ENTREE_EN_COURS,
        "crossed_toward_outside", S.EXTERIEUR, A.ANNULE_FRANCHISSEMENT,
    ),
    TransitionRule(
        "entry_lost_immediate_policy", S.ENTREE_EN_COURS,
        "lost_with_pending_and_immediate_policy", S.ABSENTE, A.EXPIRE_AMBIGU,
        "Politique ambiguous_immediate : ambiguïté déclarée dès la disparition",
    ),
    TransitionRule(
        "entry_lost_in_progress", S.ENTREE_EN_COURS,
        "lost", S.OCCULTEE, A.MARQUE_OCCULTEE,
        "Disparition pendant un franchissement entrant (spec 4.3)",
    ),
    # --- SORTIE_EN_COURS -------------------------------------------------
    TransitionRule(
        "exit_confirmed", S.SORTIE_EN_COURS, "far_outside", S.EXTERIEUR, A.COMPTE_SORTIE
    ),
    TransitionRule(
        "exit_turnaround_inside", S.SORTIE_EN_COURS,
        "far_inside", S.PRESENTE, A.ANNULE_FRANCHISSEMENT,
        "Demi-tour côté sortie : le franchissement amorcé est annulé",
    ),
    TransitionRule(
        "exit_turnaround_crossed_in", S.SORTIE_EN_COURS,
        "crossed_toward_inside", S.PRESENTE, A.ANNULE_FRANCHISSEMENT,
    ),
    TransitionRule(
        "exit_lost_immediate_policy", S.SORTIE_EN_COURS,
        "lost_with_pending_and_immediate_policy", S.ABSENTE, A.EXPIRE_AMBIGU,
    ),
    TransitionRule(
        "exit_lost_in_progress", S.SORTIE_EN_COURS,
        "lost", S.OCCULTEE, A.MARQUE_OCCULTEE,
        "Disparition pendant un franchissement sortant (spec 4.3)",
    ),
    # --- OCCULTEE ---------------------------------------------------------
    TransitionRule(
        "occluded_pending_in_confirmed", S.OCCULTEE,
        "recovered_pending_in_confirmed", S.PRESENTE, A.CONFIRME_ENTREE_DIFFEREE,
        "Réapparition cohérente : le franchissement entrant amorcé est confirmé",
    ),
    TransitionRule(
        "occluded_pending_in_cancelled", S.OCCULTEE,
        "recovered_pending_in_cancelled", S.EXTERIEUR, A.ANNULE_FRANCHISSEMENT,
    ),
    TransitionRule(
        "occluded_pending_out_confirmed", S.OCCULTEE,
        "recovered_pending_out_confirmed", S.EXTERIEUR, A.CONFIRME_SORTIE_DIFFEREE,
        "Réapparition cohérente : le franchissement sortant amorcé est confirmé",
    ),
    TransitionRule(
        "occluded_pending_out_cancelled", S.OCCULTEE,
        "recovered_pending_out_cancelled", S.PRESENTE, A.ANNULE_FRANCHISSEMENT,
    ),
    TransitionRule(
        "occluded_pending_in_dead_zone", S.OCCULTEE,
        "recovered_pending_in_dead_zone", S.EN_ZONE_MORTE,
        A.MARCHE_ZONE_MORTE_DEPUIS_EXTERIEUR,
    ),
    TransitionRule(
        "occluded_pending_out_dead_zone", S.OCCULTEE,
        "recovered_pending_out_dead_zone", S.EN_ZONE_MORTE,
        A.MARCHE_ZONE_MORTE_DEPUIS_INTERIEUR,
    ),
    TransitionRule(
        "occluded_reappears_in_dead_zone", S.OCCULTEE,
        "recovered_without_pending_in_dead_zone", S.EN_ZONE_MORTE, A.AUCUNE,
    ),
    TransitionRule(
        "occluded_reappears_inside", S.OCCULTEE,
        "recovered_and_far_inside", S.PRESENTE, A.RECUPERE_INTERIEUR,
    ),
    TransitionRule(
        "occluded_reappears_outside", S.OCCULTEE,
        "recovered_and_far_outside", S.EXTERIEUR, A.RECUPERE_EXTERIEUR,
    ),
    TransitionRule(
        "occluded_grace_expired", S.OCCULTEE,
        "grace_expired_without_pending", S.ABSENTE, A.EXPIRE_HORS_CHAMP,
        "Grâce expirée sans franchissement amorcé",
    ),
    TransitionRule(
        "occluded_grace_expired_in_progress", S.OCCULTEE,
        "grace_expired_with_pending", S.ABSENTE, A.EXPIRE_AMBIGU,
        "Grâce expirée avec franchissement amorcé : AMBIGUOUS_CROSSING (spec 4.3)",
    ),
    # --- ABSENTE (purge encaissée, galerie encore ouverte) ----------------
    TransitionRule(
        "absent_revives_inside", S.ABSENTE,
        "recovered_and_far_inside", S.PRESENTE, A.RECUPERE_INTERIEUR,
        "Réapparition d'une identité purgée encore présente en galerie (spec 3.3)",
    ),
    TransitionRule(
        "absent_revives_outside", S.ABSENTE,
        "recovered_and_far_outside", S.EXTERIEUR, A.RECUPERE_EXTERIEUR,
    ),
    TransitionRule(
        "absent_revives_dead_zone", S.ABSENTE,
        "recovered_without_pending_in_dead_zone", S.EN_ZONE_MORTE, A.AUCUNE,
    ),
)


class TransitionTableError(RuntimeError):
    """La table est incohérente : une condition ou une action est introuvable."""


def _check_table(table: tuple[TransitionRule, ...]) -> None:
    names = [rule.name for rule in table]
    if len(set(names)) != len(names):
        raise TransitionTableError(f"Noms de règles dupliqués : {names}")
    for rule in table:
        if rule.condition not in CONDITIONS:
            raise TransitionTableError(
                f"Règle {rule.name!r} : condition inconnue {rule.condition!r}"
            )
        if rule.state_to is None and rule.action is not A.SUSPEND_DECISION:
            raise TransitionTableError(
                f"Règle {rule.name!r} : state_to=None n'est autorisé que pour la "
                "suspension (spec 5.4)"
            )
    missing = set(TrackState) - {rule.state_from for rule in table}
    if missing:
        raise TransitionTableError(
            f"États sans règle sortante : {sorted(state.name for state in missing)}"
        )
    # Chaque état doit pouvoir être quitté **et** la suspension, premier filtre,
    # doit être la seule ligne applicable à tous les états.
    suspensions = [rule for rule in table if rule.state_from is None]
    if len(suspensions) != 1 or suspensions[0].action is not A.SUSPEND_DECISION:
        raise TransitionTableError(
            "La table doit contenir exactement une règle inter-états : la suspension"
        )


_check_table(TRANSITION_TABLE)

RULES_BY_NAME: Mapping[str, TransitionRule] = {rule.name: rule for rule in TRANSITION_TABLE}


class StateMachine:
    """Résout la transition applicable à une frame, à partir de la table."""

    def __init__(self, table: tuple[TransitionRule, ...] = TRANSITION_TABLE) -> None:
        self.table = table

    def resolve(self, state: TrackState, context: FsmContext) -> Decision:
        """Retourne la première transition applicable.

        Si aucune ligne ne s'applique, l'état est rendu inchangé : aucune
        transition n'est inventée par défaut, et aucune décision silencieuse
        n'est prise (règle 0.4).
        """
        for rule in self.table:
            if rule.state_from is not None and rule.state_from is not state:
                continue
            if not CONDITIONS[rule.condition](context):
                continue
            if rule.action is A.SUSPEND_DECISION:
                return Decision(
                    rule=rule.name, state_to=state, action=rule.action, state_changed=False
                )
            state_to = rule.state_to or state
            return Decision(
                rule=rule.name, state_to=state_to, action=rule.action,
                state_changed=state_to is not state,
            )
        return Decision(
            rule="no_rule_matched", state_to=state, action=A.AUCUNE, state_changed=False
        )


DEFAULT_MACHINE = StateMachine()


def apply(context: FsmContext, state: TrackState) -> Decision:
    """Raccourci fonctionnel utilisé par :class:`OccupancyManager`."""
    return DEFAULT_MACHINE.resolve(state, context)
