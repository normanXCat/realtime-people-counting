"""Gestionnaire d'occupation : cœur métier du pipeline (spec 4, 5.4, 5.8).

Ce module a été **corrigé** et non dupliqué : il garde son nom, son rôle et sa
place sur le chemin unique. La logique de transition a migré dans la table
explicite de :mod:`fsm`, la géométrie dans :mod:`geometry`, l'identité dans
:mod:`identity_manager`. Il ne reste ici que l'orchestration et la comptabilité.

Corrections apportées par rapport à l'audit (§2.2) :

- **plus de pixels absolus** : la zone morte vaut
  ``geometry.dead_zone_ratio × hauteur médiane des bbox`` ;
- **plus de durées en frames** : les fenêtres proviennent de
  :class:`config.FrameBudget`, dérivé du FPS **mesuré** ;
- **occupation encadrée** (spec 4.4) : ``occupancy_confirmed``,
  ``occupancy_uncertain`` et ``occupancy_range`` sont publiés à intervalle
  régulier, au lieu d'un entier unique qui ne pouvait que croître ;
- **garde-fous explicites** (spec 4.5) : OUT sans IN, double réassociation,
  occupation négative → événement ``INCONSISTENT_STATE``, jamais un
  ``max(0, ...)`` muet ;
- **purge jamais silencieuse** (spec 3.4) : chaque ``person_id`` purgé produit un
  événement ``PURGE`` avant sa suppression ;
- **disparition en cours de franchissement** (spec 4.3) : traitée par la
  politique ``deferred_confirmation`` avec ``AMBIGUOUS_CROSSING`` à
  l'expiration — jamais de IN/OUT confirmé sur simple disparition ;
- **ancre au bord** (spec 5.4) : décision de franchissement suspendue, avec
  événement ``ANCHOR_UNRELIABLE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from anchor_stabilizer import AnchorStabilizer
from bbox_height_locker import BBoxHeightLocker
from config import FrameBudget, PipelineConfig
from events import EventSink
from fsm import Action, Approach, Decision, FsmContext, StateMachine, TrackState
from geometry import (
    AnchorReliability,
    LocalScale,
    Side,
    VirtualLine,
    Zone,
    anchor_reliability,
    compute_anchor,
    side_of,
    zone_of,
)
from identity_manager import IdentityManager, Observation
from metrics import LatencyProfiler, Timer
from occupancy_types import OccupancySnapshot, OriginType, PersonTrack


@dataclass(frozen=True)
class Detection:
    """Détection brute d'une frame, avant toute identité logique."""

    technical_track_id: int
    bbox: np.ndarray | tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass
class _View:
    """Vue minimale utilisée par le rendu."""

    person_id: int
    technical_track_id: int
    bbox: np.ndarray | tuple[float, float, float, float]
    anchor: tuple[float, float]
    state: TrackState
    provisional: bool


STATE_COLORS: dict[TrackState, tuple[int, int, int]] = {
    TrackState.INITIALISATION: (200, 200, 200),
    TrackState.PRESENTE: (0, 255, 0),
    TrackState.EXTERIEUR: (255, 128, 0),
    TrackState.EN_ZONE_MORTE: (0, 255, 255),
    TrackState.ENTREE_EN_COURS: (0, 255, 160),
    TrackState.SORTIE_EN_COURS: (255, 255, 0),
    TrackState.OCCULTEE: (0, 165, 255),
    TrackState.ABSENTE: (0, 0, 255),
}


class OccupancyManager:
    """Assemble machine à états, identités, comptage et occupation encadrée."""

    def __init__(
        self,
        config: PipelineConfig,
        event_sink: EventSink | None = None,
        line: VirtualLine | None = None,
        identity_manager: IdentityManager | None = None,
        budget: FrameBudget | None = None,
        machine: StateMachine | None = None,
        bbox_locker: BBoxHeightLocker | None = None,
        anchor_stabilizer: AnchorStabilizer | None = None,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        self.config = config
        self.event_sink = event_sink
        self.budget = budget
        self.machine = machine or StateMachine()
        self.profiler = profiler

        self.line = line
        if self.line is None:
            # Aucune ligne par défaut : la ligne est obligatoirement celle validée
            # par l'opérateur au démarrage (voir :mod:`calibration`). Refuser ici
            # rend impossible toute divergence entre la ligne affichée et celle
            # utilisée pour la distance signée, les franchissements et le comptage.
            raise ValueError(
                "OccupancyManager exige la ligne validée par l'opérateur "
                "(paramètre `line`) : aucune ligne par défaut n'est admise."
            )
        self.scale = LocalScale(window=config.geometry.scale_window)
        self.identities = identity_manager or IdentityManager(
            long_term=config.reid.long_term,
            external_reid=config.reid.external_reid,
            grace_period_seconds=config.timing.grace_period_seconds,
            event_sink=event_sink,
        )
        self.bbox_locker = bbox_locker
        self.anchor_stabilizer = anchor_stabilizer

        self.tracks: dict[int, PersonTrack] = {}
        self.initial_occupancy = 0
        self.total_in = 0
        self.total_out = 0
        self.total_new = 0

        #: Identités purgées sans sortie observée (borne haute de l'occupation).
        self._uncertain: set[int] = set()
        self._warmup_done = False
        self._frames = 0
        self._bootstrapping_frames = 0
        self._last_snapshot_s: float | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._views: list[_View] = []

    # ------------------------------------------------------------------
    # Propriétés d'état
    # ------------------------------------------------------------------
    @property
    def phase(self) -> str:
        if self.budget is None:
            return "BOOTSTRAP"
        return "RUNNING" if self._warmup_done else "WARMUP"

    @property
    def occupancy_confirmed(self) -> int:
        """Entrées confirmées sans sortie confirmée (spec 4.4)."""
        return sum(
            1
            for track in self.tracks.values()
            if track.inside_occupancy and track.state is not TrackState.ABSENTE
        )

    @property
    def occupancy_uncertain(self) -> int:
        """Personnes purgées sans sortie observée, cumulées sur la session (spec 4.4).

        Le compte survit à la libération de la mémoire de galerie : sans cela,
        la borne haute de l'occupation s'effondrerait dès qu'une identité est
        oubliée, ce qui masquerait précisément les sorties non observées que
        l'encadrement doit rendre visibles.
        """
        return len(self._uncertain)

    @property
    def occupancy_range(self) -> tuple[int, int]:
        confirmed = self.occupancy_confirmed
        return (confirmed, confirmed + self.occupancy_uncertain)

    @property
    def occupancy_count(self) -> int:
        """Effectif confirmé (nom conservé pour le HUD et l'évaluation)."""
        return self.occupancy_confirmed

    @property
    def visible_count(self) -> int:
        return sum(1 for track in self.tracks.values() if track.is_visible)

    @property
    def occluded_count(self) -> int:
        return sum(1 for track in self.tracks.values() if track.state is TrackState.OCCULTEE)

    @property
    def bootstrap_frames(self) -> int:
        return self._bootstrapping_frames

    def snapshot(self) -> OccupancySnapshot:
        confirmed = self.occupancy_confirmed
        uncertain = self.occupancy_uncertain
        return OccupancySnapshot(
            confirmed=confirmed,
            uncertain=uncertain,
            range_low=confirmed,
            range_high=confirmed + uncertain,
            visible=self.visible_count,
            occluded=self.occluded_count,
            initial=self.initial_occupancy,
            total_in=self.total_in,
            total_out=self.total_out,
            total_new=self.total_new,
        )

    # ------------------------------------------------------------------
    # Budget temporel
    # ------------------------------------------------------------------
    def set_frame_budget(self, budget: FrameBudget) -> None:
        """Fixe les fenêtres en frames à partir du FPS **mesuré** (règle 0.2)."""
        self.budget = budget

    # ------------------------------------------------------------------
    # Traitement d'une frame
    # ------------------------------------------------------------------
    def process_frame(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        """Traite une frame complète : identités, FSM, comptage, occupation."""
        self._frames += 1
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        self.scale.observe_many([detection.bbox for detection in detections])

        if self.budget is None:
            # Aucun FPS mesuré : les durées ne peuvent pas être converties en
            # frames, donc aucune décision n'est prise (règle 0.2). Seule
            # l'échelle locale est accumulée.
            self._bootstrapping_frames += 1
            self._views = []
            return

        budget = self.budget
        self._update_warmup(frame_index, timestamp_s)
        dead_zone_px = self.scale.ratio_to_px(self.config.geometry.dead_zone_ratio)
        margin_ratio = self.config.geometry.anchor_edge_margin_ratio

        # Stabilisation optionnelle (spec 7) : la boîte brute reste disponible et
        # la correction est journalisée avec sa raison et son seuil.
        raw_boxes: dict[int, np.ndarray] = {}
        corrected: dict[int, tuple[np.ndarray, tuple[float, float]]] = {}
        for detection in detections:
            technical_id = int(detection.technical_track_id)
            raw = np.asarray(detection.bbox, dtype=float)
            bbox = raw
            if self.bbox_locker is not None:
                bbox = np.asarray(self.bbox_locker.process_bbox(technical_id, raw), dtype=float)
            anchor = compute_anchor(bbox)
            if self.anchor_stabilizer is not None:
                anchor = tuple(self.anchor_stabilizer.get_stabilized_anchor(technical_id, bbox))
            raw_boxes[technical_id] = raw
            corrected[technical_id] = (bbox, anchor)

        observations = [
            Observation(
                technical_track_id=int(detection.technical_track_id),
                bbox=corrected[int(detection.technical_track_id)][0],
                confidence=float(detection.confidence),
                anchor=corrected[int(detection.technical_track_id)][1],
                bbox_height=float(
                    corrected[int(detection.technical_track_id)][0][3]
                    - corrected[int(detection.technical_track_id)][0][1]
                ),
                anchor_reliable=(
                    anchor_reliability(detection.bbox, width, height, margin_ratio)
                    is AnchorReliability.FIABLE
                ),
            )
            for detection in detections
        ]

        with Timer() as reid_timer:
            assignments = self.identities.assign(
                observations, frame, timestamp_s, frame_index
            )
        if self.profiler is not None:
            self.profiler.record("reid", reid_timer.ms)

        with Timer() as fsm_timer:
            views: list[_View] = []
            seen: set[int] = set()
            for assignment in assignments:
                if not assignment.person_id:
                    # Politique « defer » : décision différée, aucun comptage,
                    # aucune création d'identité (spec 3.3 étape 5).
                    continue
                self._report_stabilization(assignment, raw_boxes, corrected, timestamp_s, frame_index)
                track = self._get_or_create_track(assignment, timestamp_s)
                seen.add(track.person_id)
                self._update_track(
                    track, assignment, width, height, dead_zone_px, budget,
                    timestamp_s, frame_index,
                )
                if track.is_visible:
                    views.append(
                        _View(
                            person_id=track.person_id,
                            technical_track_id=track.technical_track_id,
                            bbox=assignment.bbox,
                            anchor=track.anchor,
                            state=track.state,
                            provisional=track.provisional,
                        )
                    )
            self._views = views
            for person_id, track in list(self.tracks.items()):
                if person_id in seen:
                    continue
                self._handle_missing(track, budget, timestamp_s, frame_index)
        if self.profiler is not None:
            self.profiler.record("fsm", fsm_timer.ms)

        self._publish_snapshot_if_due(timestamp_s, frame_index)
        for person_id in self.identities.release_expired(timestamp_s):
            # La purge a déjà été journalisée lors du passage à ABSENTE : la
            # libération de mémoire n'est donc jamais silencieuse (spec 3.4).
            self.tracks.pop(person_id, None)

    def _report_stabilization(
        self,
        assignment,
        raw_boxes: dict[int, np.ndarray],
        corrected: dict[int, tuple[np.ndarray, tuple[float, float]]],
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        """Journalise une correction de stabilisation (spec 7)."""
        if self.bbox_locker is None and self.anchor_stabilizer is None:
            return
        technical_id = int(assignment.technical_track_id)
        raw = raw_boxes.get(technical_id)
        entry = corrected.get(technical_id)
        if raw is None or entry is None:
            return
        bbox, anchor = entry
        if not np.allclose(raw, bbox):
            self._emit(
                "STABILIZATION",
                timestamp_s,
                frame_index,
                person_id=assignment.person_id,
                raw_bbox=[round(float(v), 2) for v in raw],
                corrected_bbox=[round(float(v), 2) for v in bbox],
                reason="hauteur_corrigee_sous_seuil",
                threshold=float(self.bbox_locker.min_height_ratio),
                component="bbox_locker",
            )
            return
        if self.anchor_stabilizer is not None and anchor != compute_anchor(bbox):
            self._emit(
                "STABILIZATION",
                timestamp_s,
                frame_index,
                person_id=assignment.person_id,
                raw_bbox=[round(float(v), 2) for v in raw],
                corrected_bbox=[round(float(v), 2) for v in bbox],
                reason="ancrage_recalibre_sur_vitesse_tete",
                threshold=float(self.anchor_stabilizer.tolerance_px),
                component="anchor_stabilizer",
            )

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    def _update_warmup(self, frame_index: int, timestamp_s: float) -> None:
        if self._warmup_done or self.budget is None:
            return
        if frame_index >= self.budget.warmup_frames:
            self._warmup_done = True
            self._emit(
                "WARMUP_END",
                timestamp_s,
                frame_index,
                initial_occupancy=self.initial_occupancy,
                budget_frames=self.budget.warmup_frames,
                fps=round(self.budget.fps, 3),
            )

    # ------------------------------------------------------------------
    # Suivi d'une personne observée
    # ------------------------------------------------------------------
    def _get_or_create_track(self, assignment, timestamp_s: float) -> PersonTrack:
        track = self.tracks.get(assignment.person_id)
        if track is not None:
            return track
        track = PersonTrack(
            person_id=assignment.person_id,
            state=TrackState.INITIALISATION if not self._warmup_done else TrackState.EXTERIEUR,
            origin=(
                OriginType.REASSOCIATION if assignment.matched
                else OriginType.APPARITION_INTERIEURE
            ),
            technical_track_id=assignment.technical_track_id,
            anchor=assignment.anchor,
            bbox=np.asarray(assignment.bbox, dtype=float),
            bbox_height=assignment.bbox_height,
            last_seen_s=timestamp_s,
            provisional=assignment.provisional,
            aliases={assignment.technical_track_id},
        )
        self.tracks[assignment.person_id] = track
        return track

    def _update_track(
        self,
        track: PersonTrack,
        assignment,
        width: int,
        height: int,
        dead_zone_px: float | None,
        budget: FrameBudget,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        previous_state = track.state
        reliable = bool(assignment.anchor_reliable)
        distance = self.line.distance(assignment.anchor, width, height)
        side = side_of(distance, self.line.inside_side, self.line.on_line_policy)
        zone = zone_of(distance, dead_zone_px, self.line.inside_side)

        crossed: str | None = None
        if reliable and track.previous_anchor is not None:
            crossed = self.line.crossing(
                track.previous_anchor, assignment.anchor, width, height
            )
        if reliable:
            # Seule une ancre fiable alimente la référence de franchissement
            # (spec 5.4) : pendant une suspension, la référence est gelée.
            track.previous_anchor = assignment.anchor
        if crossed is not None and not self._warmup_done:
            track.crossed += 1

        track.interior_streak = track.interior_streak + 1 if zone is Zone.INTERIEURE else 0

        # Frames écoulées depuis la dernière observation : non nul dès qu'une
        # piste réapparaît, ce qui est la condition d'une récupération (spec 3.3).
        lost_frames = max(0, frame_index - track.last_seen_frame)

        context = FsmContext(
            observed=True,
            zone=zone,
            side=side,
            crossed=crossed,
            anchor_reliable=reliable,
            scale_available=dead_zone_px is not None,
            lost_frames=lost_frames,
            grace_period_frames=budget.grace_period_frames,
            interior_streak=track.interior_streak,
            confirmation_frames=budget.confirmation_frames,
            warmup_done=self._warmup_done,
            warmup_stable=(
                track.interior_streak >= budget.confirmation_frames and track.crossed == 0
            ),
            approach=track.approach,
            pending_crossing=track.pending_crossing,
            in_progress_policy=self.config.occupancy.in_progress_disappearance_policy,
        )
        decision = self.machine.resolve(track.state, context)
        self._apply_decision(
            track, decision, context, assignment, width, height, timestamp_s, frame_index
        )

        track.anchor = assignment.anchor
        track.bbox = np.asarray(assignment.bbox, dtype=float)
        track.bbox_height = assignment.bbox_height
        track.technical_track_id = assignment.technical_track_id
        track.confidence = assignment.confidence
        track.last_seen_frame = frame_index
        track.last_seen_s = timestamp_s
        track.last_zone = zone.name if zone is not None else "INDISPONIBLE"
        track.last_distance = distance
        track.aliases.add(assignment.technical_track_id)
        self._emit_transition_if_changed(
            track, previous_state, decision, zone, distance, timestamp_s, frame_index
        )

    # ------------------------------------------------------------------
    # Personne non observée
    # ------------------------------------------------------------------
    def _handle_missing(
        self,
        track: PersonTrack,
        budget: FrameBudget,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        if track.state is TrackState.ABSENTE:
            return
        previous_state = track.state
        context = FsmContext(
            observed=False,
            zone=None,
            side=Side.INDETERMINEE,
            crossed=None,
            anchor_reliable=True,
            scale_available=True,
            lost_frames=frame_index - track.last_seen_frame,
            grace_period_frames=budget.grace_period_frames,
            interior_streak=0,
            confirmation_frames=budget.confirmation_frames,
            warmup_done=self._warmup_done,
            warmup_stable=False,
            approach=track.approach,
            pending_crossing=track.pending_crossing,
            in_progress_policy=self.config.occupancy.in_progress_disappearance_policy,
        )
        decision = self.machine.resolve(track.state, context)
        self._apply_decision(
            track, decision, context, None, self._frame_size[0], self._frame_size[1],
            timestamp_s, frame_index,
        )
        self._emit_transition_if_changed(
            track, previous_state, decision, None, None, timestamp_s, frame_index
        )

    # ------------------------------------------------------------------
    # Application des actions de la table
    # ------------------------------------------------------------------
    def _apply_decision(
        self,
        track: PersonTrack,
        decision: Decision,
        context: FsmContext,
        assignment,
        width: int,
        height: int,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        action = decision.action
        track.last_rule = decision.rule
        technical_id = (
            track.technical_track_id if assignment is None else assignment.technical_track_id
        )

        if action is Action.SUSPEND_DECISION:
            track.anchor_suspend_streak += 1
            if track.anchor_suspend_streak == 1:
                reason = "edge_margin" if not context.anchor_reliable else "scale_unavailable"
                self._emit(
                    "ANCHOR_UNRELIABLE",
                    timestamp_s,
                    frame_index,
                    person_id=track.person_id,
                    reason=reason,
                    margin_px=round(
                        self.config.geometry.anchor_edge_margin_ratio * min(width, height), 2
                    ),
                    bbox=None if track.bbox is None else [round(float(v), 2) for v in track.bbox],
                    consecutive_frames=1,
                )
            return
        track.anchor_suspend_streak = 0

        if action is Action.EMPLACE_INITIAL:
            if not track.inside_occupancy:
                track.inside_occupancy = True
                self._uncertain.discard(track.person_id)
                self.initial_occupancy += 1
            track.origin = OriginType.INITIALISATION

        elif action in (Action.COMPTE_ENTREE, Action.CONFIRME_ENTREE_DIFFEREE):
            self._count_in(track, technical_id, action, timestamp_s, frame_index)

        elif action in (Action.COMPTE_SORTIE, Action.CONFIRME_SORTIE_DIFFEREE):
            self._count_out(track, technical_id, action, timestamp_s, frame_index)

        elif action is Action.NOUVELLE_PRESENCE:
            self._count_new(track, technical_id, timestamp_s, frame_index)

        elif action is Action.RECUPERE_INTERIEUR:
            self._uncertain.discard(track.person_id)
            track.origin = OriginType.REASSOCIATION
            track.pending_crossing = None

        elif action is Action.RECUPERE_EXTERIEUR:
            # Réapparition cohérente côté extérieur d'une personne qui était
            # comptée présente : la trajectoire se poursuit hors de la salle, la
            # sortie est donc confirmée à ce moment-là (spec 4.3, réapparition
            # cohérente). La raison est journalisée pour un audit séparé.
            if track.inside_occupancy:
                self._count_out(
                    track, technical_id, Action.RECUPERE_EXTERIEUR, timestamp_s, frame_index
                )
            else:
                self._uncertain.discard(track.person_id)
            track.pending_crossing = None

        elif action is Action.AMORCE_ENTREE:
            track.pending_crossing = "in"

        elif action is Action.AMORCE_SORTIE:
            track.pending_crossing = "out"

        elif action is Action.ANNULE_FRANCHISSEMENT:
            track.pending_crossing = None

        elif action is Action.SIGNALE_DERIVE:
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="side_change_without_crossing",
                person_id=track.person_id,
                technical_track_id=int(technical_id),
                details=(
                    "Demi-plan modifié sans intersection de la ligne capturée "
                    "(ligne non couvrante ou saut de suivi)"
                ),
                resolution="franchissement soumis à confirmation par la zone morte",
            )

        elif action is Action.MARQUE_OCCULTEE:
            self._mark_occluded(track, timestamp_s, frame_index)

        elif action is Action.EXPIRE_HORS_CHAMP:
            self._expire(track, timestamp_s, frame_index, ambiguous=False)

        elif action is Action.EXPIRE_AMBIGU:
            self._expire(track, timestamp_s, frame_index, ambiguous=True)

        elif action is Action.MARCHE_ZONE_MORTE_DEPUIS_INTERIEUR:
            track.approach = Approach.INTERIEUR

        elif action is Action.MARCHE_ZONE_MORTE_DEPUIS_EXTERIEUR:
            track.approach = Approach.EXTERIEUR

        track.state = decision.state_to

    # ------------------------------------------------------------------
    # Comptage avec garde-fous (spec 4.5)
    # ------------------------------------------------------------------
    def _count_in(
        self, track: PersonTrack, technical_id: int, action: Action,
        timestamp_s: float, frame_index: int,
    ) -> None:
        track.pending_crossing = None
        self._uncertain.discard(track.person_id)
        if track.inside_occupancy:
            return
        track.inside_occupancy = True
        if track.origin is not OriginType.INITIALISATION:
            track.origin = OriginType.ENTREE_LIGNE
        self.total_in += 1
        self._emit(
            "IN",
            timestamp_s,
            frame_index,
            person_id=track.person_id,
            technical_track_id=int(technical_id),
            direction="in",
            reason=(
                "deferred_confirmation"
                if action is Action.CONFIRME_ENTREE_DIFFEREE
                else "crossing_confirmed"
            ),
            occupancy_after=self.occupancy_confirmed,
        )

    def _count_out(
        self, track: PersonTrack, technical_id: int, action: Action,
        timestamp_s: float, frame_index: int,
    ) -> None:
        if not track.inside_occupancy:
            track.pending_crossing = None
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="out_without_in",
                person_id=track.person_id,
                technical_track_id=int(technical_id),
                details="Sortie confirmée alors que la personne n'était pas comptée présente",
                resolution="OUT non comptabilisé, état aligné sur la sortie",
            )
            return
        track.inside_occupancy = False
        self._uncertain.discard(track.person_id)
        track.pending_crossing = None
        self.total_out += 1
        if action is Action.CONFIRME_SORTIE_DIFFEREE:
            reason = "deferred_confirmation"
        elif action is Action.RECUPERE_EXTERIEUR:
            reason = "recovery_outside_coherent"
        else:
            reason = "crossing_confirmed"
        self._emit(
            "OUT",
            timestamp_s,
            frame_index,
            person_id=track.person_id,
            technical_track_id=int(technical_id),
            direction="out",
            reason=reason,
            occupancy_after=self.occupancy_confirmed,
        )

    def _count_new(
        self, track: PersonTrack, technical_id: int, timestamp_s: float, frame_index: int
    ) -> None:
        if track.inside_occupancy:
            return
        track.inside_occupancy = True
        self._uncertain.discard(track.person_id)
        track.origin = OriginType.APPARITION_INTERIEURE
        self.total_new += 1
        self._emit(
            "NEW",
            timestamp_s,
            frame_index,
            person_id=track.person_id,
            technical_track_id=int(technical_id),
            direction="in",
            reason="apparition_interieure_confirmee",
            origin=track.origin.name,
            occupancy_after=self.occupancy_confirmed,
            stable_frames=track.interior_streak,
        )

    def _mark_occluded(self, track: PersonTrack, timestamp_s: float, frame_index: int) -> None:
        track.occlusion_events += 1
        self._emit(
            "OCCLUDED",
            timestamp_s,
            frame_index,
            person_id=track.person_id,
            technical_track_id=track.technical_track_id,
            last_position=[round(track.anchor[0], 2), round(track.anchor[1], 2)],
            state_before=track.state.name,
        )

    def _expire(
        self, track: PersonTrack, timestamp_s: float, frame_index: int, ambiguous: bool
    ) -> None:
        if ambiguous:
            self._emit(
                "AMBIGUOUS_CROSSING",
                timestamp_s,
                frame_index,
                person_id=track.person_id,
                technical_track_id=track.technical_track_id,
                direction=track.pending_crossing or "indeterminate",
                pending_direction=track.pending_crossing,
                reason="franchissement_amorce_non_resolu",
                policy=self.config.occupancy.in_progress_disappearance_policy,
                last_position=[round(track.anchor[0], 2), round(track.anchor[1], 2)],
            )
            track.pending_crossing = None
        impact = "none"
        if track.inside_occupancy:
            # Purge sans sortie observée : la personne reste dans la borne haute
            # de l'occupation (spec 4.4), elle n'est jamais retirée en silence.
            self._uncertain.add(track.person_id)
            impact = "uncertain"
        self.identities.mark_purged(
            track.person_id,
            timestamp_s,
            frame_index,
            reason="grace_expired_in_progress" if ambiguous else "grace_expired",
            last_state=track.state.name,
            occupancy_impact=impact,
            grace_period_frames=None if self.budget is None else self.budget.grace_period_frames,
        )

    # ------------------------------------------------------------------
    # Occupation encadrée et cohérence (spec 4.4, 4.5)
    # ------------------------------------------------------------------
    def _publish_snapshot_if_due(self, timestamp_s: float, frame_index: int) -> None:
        interval = self.config.occupancy.snapshot_interval_seconds
        if (
            self._last_snapshot_s is not None
            # Tolérance de 1 µs : les horodatages mesurés ne sont jamais
            # exactement espacés, l'intervalle ne doit pas être manqué pour
            # un écart d'arrondi flottant.
            and timestamp_s - self._last_snapshot_s < interval - 1e-6
        ):
            return
        self._last_snapshot_s = timestamp_s
        self._check_consistency(timestamp_s, frame_index)
        self._emit("OCCUPANCY_SNAPSHOT", timestamp_s, frame_index, **self.snapshot().to_fields())

    def _check_consistency(self, timestamp_s: float, frame_index: int) -> None:
        ledger = self.initial_occupancy + self.total_in + self.total_new - self.total_out
        if ledger < 0:
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="negative_occupancy_before_clamp",
                details=f"Bilan d'occupation négatif : {ledger}",
                resolution="valeur publiée bornée, incohérence journalisée (jamais silencieuse)",
            )

    # ------------------------------------------------------------------
    # Émission
    # ------------------------------------------------------------------
    def _emit_transition_if_changed(
        self,
        track: PersonTrack,
        previous_state: TrackState,
        decision: Decision,
        zone: Zone | None,
        distance: float | None,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        if previous_state is track.state:
            return
        self._emit(
            "STATE_TRANSITION",
            timestamp_s,
            frame_index,
            person_id=track.person_id,
            from_state=previous_state.name,
            to_state=track.state.name,
            rule=decision.rule,
            action=decision.action.name,
            zone=None if zone is None else zone.name,
            distance_signed=None if distance is None else round(distance, 2),
        )

    def _emit(self, event_type: str, timestamp_s: float, frame_index: int, **fields: Any) -> None:
        if self.event_sink is None:
            return
        self.event_sink.emit(event_type, timestamp_s, frame_index, **fields)

    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray) -> None:
        """Dessine la ligne, la zone morte relative, les boîtes et le HUD."""
        height, width = frame.shape[:2]
        start, end = self.line.to_pixels(width, height)
        cv2.line(
            frame,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        self._draw_dead_zone(frame, start, end)

        for view in self._views:
            color = STATE_COLORS.get(view.state, (255, 255, 255))
            x1, y1, x2, y2 = (int(v) for v in view.bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(view.anchor[0]), int(view.anchor[1])), 4, (0, 0, 255), -1)
            label = f"P{view.person_id}"
            if view.provisional:
                label += " incertain"
            cv2.putText(
                frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        occluded_color = STATE_COLORS[TrackState.OCCULTEE]
        for track in self.tracks.values():
            if track.state is not TrackState.OCCULTEE:
                continue
            center = (int(track.anchor[0]), int(track.anchor[1]))
            cv2.circle(frame, center, 14, occluded_color, 2)
            cv2.putText(
                frame, f"?P{track.person_id}", (center[0] - 14, center[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, occluded_color, 1, cv2.LINE_AA,
            )

        self._draw_hud(frame)

    def _draw_dead_zone(
        self, frame: np.ndarray, start: tuple[float, float], end: tuple[float, float]
    ) -> None:
        dead_zone_px = self.scale.ratio_to_px(self.config.geometry.dead_zone_ratio)
        if not dead_zone_px:
            return
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(float(np.hypot(dx, dy)), 1.0)
        nx, ny = dy / length, -dx / length
        points = np.array(
            [
                [start[0] + nx * dead_zone_px, start[1] + ny * dead_zone_px],
                [end[0] + nx * dead_zone_px, end[1] + ny * dead_zone_px],
                [end[0] - nx * dead_zone_px, end[1] - ny * dead_zone_px],
                [start[0] - nx * dead_zone_px, start[1] - ny * dead_zone_px],
            ],
            dtype=np.int32,
        )
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 255))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    def _draw_hud(self, frame: np.ndarray) -> None:
        _height, width = frame.shape[:2]
        scale = max(0.5, min(1.0, width / 1280.0))
        snapshot = self.snapshot()
        lines = [
            f"OCCUPATION CONFIRMEE: {snapshot.confirmed}",
            f"INCERTAINE: {snapshot.uncertain}  RANGE: [{snapshot.range_low}, {snapshot.range_high}]",
            f"IN: {snapshot.total_in}  OUT: {snapshot.total_out}  NEW: {snapshot.total_new}",
            f"Visibles: {snapshot.visible}  Occultees: {snapshot.occluded}  Phase: {self.phase}",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 0.5 * scale
        thickness = max(1, int(1.5 * scale))
        line_height = int(22 * scale)
        pad = int(10 * scale)
        text_width = max(cv2.getTextSize(line, font, f_scale, thickness)[0][0] for line in lines)
        box_w = text_width + 2 * pad
        box_h = len(lines) * line_height + 2 * pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (120, 120, 120), 1)
        colors = [(0, 255, 255), (0, 200, 255), (0, 255, 0), (200, 200, 200)]
        for index, (text, color) in enumerate(zip(lines, colors)):
            y = 10 + pad + (index + 1) * line_height - int(4 * scale)
            cv2.putText(frame, text, (10 + pad, y), font, f_scale, color, thickness, cv2.LINE_AA)
