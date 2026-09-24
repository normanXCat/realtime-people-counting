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
  événement ``ANCHOR_UNRELIABLE`` ;
- **warm-up robuste** (spec 1 du prompt) : clôturé sur **deux** bornes
  (``warmup_min_frames`` frames observées **et** ``warmup_seconds`` écoulées),
  avec une frontière explicite (la frame qui atteint les deux bornes appartient
  au warm-up) et un ``WARMUP_END`` qui publie ``warmup_frames_observed``,
  ``warmup_elapsed_seconds`` et l'``initial_occupancy`` réellement retenu ;
- **``NEW`` strictement encadré** (spec 4 du prompt) : une apparition intérieure
  stable ne devient une nouvelle présence que si la personne n'a jamais été vue
  à l'extérieur et qu'aucune personne déjà comptée n'est occultée à proximité —
  sinon la décision est signalée et **différée**, jamais devinée ;
- **occupation opérationnelle / incertaine** (spec 6 du prompt) :
  ``occupancy_confirmed`` **est** l'effectif opérationnel
  (``initial + IN + NEW - OUT``) et il **ne diminue que sur une sortie
  confirmée par la FSM** — jamais sur une occultation, une expiration de grâce,
  une absence de détection, une purge technique ou un changement d'ID
  technique. ``occupancy_observed`` est la part actuellement observable, et
  ``occupancy_uncertain`` celle dont l'observation est perdue :
  ``observée + incertaine == opérationnelle`` est un invariant vérifié à chaque
  frame. Une purge est toujours ``purge_kind="technical"`` et ``is_exit=false`` :
  elle ne peut pas produire de sortie.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from anchor_stabilizer import AnchorStabilizer
from bbox_height_locker import BBoxHeightLocker
from calibration import draw_line_overlay
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
    check_anchor_height_drop,
    check_box_width_growth,
    heads_suggest_merge,
    check_detection_geometry,
    compute_anchor,
    side_of,
    zone_of,
)
from identity_manager import IdentityManager, Observation
from metrics import LatencyProfiler, Timer
from occupancy_types import OccupancySnapshot, OriginType, PersonTrack
from post_occlusion_recovery import PostOcclusionRecovery

_log = logging.getLogger("occupancy_manager")


@dataclass(frozen=True)
class Detection:
    """Détection brute d'une frame, avant toute identité logique."""

    technical_track_id: int
    bbox: np.ndarray | tuple[float, float, float, float]
    confidence: float = 1.0
    head_point: tuple[float, float] | None = None
    head_confidence: float | None = None
    #: Tous les points-clés de tête **confiants** de cette détection. Distinct de
    #: ``head_point`` (le point agrégé utilisé pour l'assistance de présence) :
    #: c'est leur **nombre** et leur **séparation** qui signalent une fusion de
    #: deux personnes dans une seule boîte, y compris dès la première frame.
    #: Vide tant que le modèle pose n'est pas actif (``presence.head_assist``).
    head_points: tuple[tuple[float, float], ...] = ()


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
        no_line: bool = False,
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
        #: Mode SANS ligne, toujours explicite (``--no-line`` ou touche « N ») :
        #: toute l'image est l'intérieur, aucun franchissement n'est possible,
        #: donc jamais de IN ni de OUT ; effectif = warm-up + NEW. La table de
        #: transition est inchangée : seules la zone et le côté sont imposés.
        self.no_line = bool(no_line) and line is None
        if self.line is None and not self.no_line:
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
            appearance_continuity=config.reid.appearance_continuity,
            grace_period_seconds=config.timing.grace_period_seconds,
            event_sink=event_sink,
        )
        self.bbox_locker = bbox_locker
        self.anchor_stabilizer = anchor_stabilizer
        #: Post-Occlusion Recovery, étape 1 (``None`` si désactivé : comportement
        #: strictement identique à la version sans récupération).
        self.recovery = (
            PostOcclusionRecovery(
                config.post_occlusion_recovery,
                describe=self.identities.describe,
                reference_feature=self._reference_feature,
                emit=self._emit,
            )
            if config.post_occlusion_recovery.enabled
            else None
        )

        self.tracks: dict[int, PersonTrack] = {}
        self.initial_occupancy = 0
        self.total_in = 0
        self.total_out = 0
        self.total_new = 0

        #: Identités purgées sans sortie observée (part incertaine de
        #: l'effectif opérationnel : la personne reste comptée).
        self._uncertain: set[int] = set()
        self._warmup_done = False
        self._frames = 0
        self._bootstrapping_frames = 0
        #: Warm-up : compteur de frames observées, horodatage de la première et
        #: durée écoulée constatée sur la dernière frame de warm-up.
        self._warmup_frames_observed = 0
        self._warmup_first_timestamp_s: float | None = None
        self._warmup_elapsed_seconds = 0.0
        self._warmup_report_pending = False
        self._last_snapshot_s: float | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._views: list[_View] = []
        #: Hauteur de boîte **brute** de la frame courante, par identifiant
        #: technique : c'est la mesure qui alimente la détection de chute.
        self._raw_heights: dict[int, float] = {}
        #: Clés de repli créées dans la frame courante (identifiant technique pas
        #: encore résolu par ``IdentityManager``) : elles ne doivent pas être
        #: conservées, contrairement aux profils indexés par ``person_id``.
        self._fallback_keys: set[int] = set()

    # ------------------------------------------------------------------
    # Clés de stabilisation : indexées par person_id, jamais par ID technique
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_key(technical_track_id: int) -> int:
        """Clé de repli pour un identifiant technique **pas encore résolu**.

        Négative, donc impossible à confondre avec un ``person_id`` (qui commence
        à 1) : un profil de personne ne peut pas hériter de la hauteur d'une piste
        technique homonyme.
        """
        return -1 - int(technical_track_id)

    def stabilization_key(self, technical_track_id: int) -> int:
        """Clé d'historique de hauteur/ancre pour un identifiant technique.

        BoT-SORT réattribue un identifiant technique après une occultation ;
        indexer l'historique par cet identifiant volatil le réinitialiserait
        exactement au moment où la correction de hauteur est la plus utile. Le
        ``person_id`` résolu par :mod:`identity_manager` est donc utilisé dès
        qu'il est connu, et la correspondance reste valable après un
        ``TECHNICAL_ID_CHANGED``.
        """
        resolved = self.identities.person_of(int(technical_track_id))
        if resolved is not None:
            return int(resolved)
        key = self._fallback_key(technical_track_id)
        self._fallback_keys.add(key)
        return key

    def stabilization_keys(self) -> set[int]:
        """Clés d'historique encore vivantes (état courante du suivi).

        Utilisée pour purger les profils : la purge doit se baser sur la
        **personne** suivie, pas sur les identifiants présents dans la frame, sinon
        une seule frame manquée détruirait l'historique de hauteur.
        """
        keys = {int(track.person_id) for track in self.tracks.values()}
        keys.update(self._fallback_keys)
        return keys

    # ------------------------------------------------------------------
    # Propriétés d'état
    # ------------------------------------------------------------------
    @property
    def phase(self) -> str:
        if self.budget is None:
            return "BOOTSTRAP"
        return "RUNNING" if self._warmup_done else "WARMUP"

    @property
    def occupancy_operational(self) -> int:
        """Effectif **opérationnel** : ``initial + IN + NEW - OUT``.

        C'est le bilan du pipeline, et il n'est **jamais** borné par un
        ``max(0, ...)`` : une valeur négative est une incohérence à journaliser
        (``INCONSISTENT_STATE``), pas à masquer.

        Il ne diminue que sur une **sortie confirmée** par la machine à états
        (ou une réapparition cohérente côté extérieur, qui vaut confirmation de
        sortie). Aucun autre chemin ne le touche : ni le passage
        ``PRESENTE -> OCCULTEE``, ni l'expiration de la grâce, ni une absence de
        détection, ni une purge technique, ni un changement d'identifiant
        technique, ni la libération de la mémoire de galerie.
        """
        return self.initial_occupancy + self.total_in + self.total_new - self.total_out

    @property
    def occupancy_confirmed(self) -> int:
        """Occupation **confirmée** = effectif opérationnel (figure publiée).

        Alias explicite de :attr:`occupancy_operational` : c'est la valeur
        affichée dans le HUD, publiée dans ``SESSION_END`` et utilisée par
        l'évaluation. Elle ne baisse donc **que** sur une sortie confirmée par
        la FSM — jamais sur une occultation ou une purge.
        """
        return self.occupancy_operational

    @property
    def occupancy_observed(self) -> int:
        """Part de l'effectif opérationnel **actuellement observable**.

        Une personne comptée mais occultée (ou purgée en attendant une
        réapparition) n'y figure pas : elle reste dans
        :attr:`occupancy_operational` et alimente :attr:`occupancy_uncertain`.
        ``occupancy_observed + occupancy_uncertain == occupancy_operational``
        est vérifié à chaque frame (invariant journalisé, jamais corrigé en
        silence).
        """
        return sum(
            1
            for track in self.tracks.values()
            if track.inside_occupancy and track.state is not TrackState.ABSENTE
        )

    @property
    def occupancy_uncertain(self) -> int:
        """Personnes comptées dont l'observation est perdue (spec 4.4, 6).

        Le compte survit à la libération de la mémoire de galerie : sans cela,
        l'information « quelqu'un est peut-être encore là » disparaîtrait dès
        qu'une identité est oubliée. ``occupancy_uncertain`` est un **sous-ensemble**
        de :attr:`occupancy_operational`, jamais une diminution de celui-ci.
        """
        return len(self._uncertain)

    @property
    def occupancy_range(self) -> tuple[int, int]:
        """Encadrement publié : ``[observée, opérationnelle]``."""
        return (self.occupancy_observed, self.occupancy_operational)

    @property
    def occupancy_count(self) -> int:
        """Effectif opérationnel confirmé (nom conservé pour le HUD et l'évaluation)."""
        return self.occupancy_operational

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
        operational = self.occupancy_operational
        observed = self.occupancy_observed
        uncertain = self.occupancy_uncertain
        return OccupancySnapshot(
            confirmed=operational,
            uncertain=uncertain,
            range_low=observed,
            range_high=operational,
            visible=self.visible_count,
            occluded=self.occluded_count,
            initial=self.initial_occupancy,
            total_in=self.total_in,
            total_out=self.total_out,
            total_new=self.total_new,
            observed=observed,
            operational=operational,
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
        raw_detections: np.ndarray | None = None,
    ) -> None:
        """Traite une frame complète : identités, FSM, comptage, occupation.

        ``raw_detections`` : détections YOLO brutes, avant tracker
        (``x1, y1, x2, y2, conf, ...``), utilisées seulement par la
        récupération post-occultation.
        """
        self._frames += 1
        height, width = frame.shape[:2]
        self._frame_size = (width, height)

        # Filtrage géométrique (spec correction 1) : écarter les artefacts extrêmes
        # tout en acceptant les postures assises (ratio W/H jusqu'à max_aspect_ratio_wh).
        valid_detections: list[Detection] = []
        for det in detections:
            valid, reason, ratio = check_detection_geometry(
                det.bbox,
                confidence=det.confidence,
                min_aspect_ratio=self.config.geometry.min_aspect_ratio_wh,
                max_aspect_ratio=self.config.geometry.max_aspect_ratio_wh,
                min_height_px=self.config.geometry.min_box_height_px,
                min_width_px=self.config.geometry.min_box_width_px,
            )
            if not valid:
                _log.debug(
                    "Frame %d: detection %d rejected by geometric filter: reason=%s, w/h=%.2f, conf=%.3f",
                    frame_index,
                    det.technical_track_id,
                    reason,
                    ratio,
                    det.confidence,
                )
                self._emit(
                    "DETECTION_REJECTED_GEOMETRY",
                    timestamp_s,
                    frame_index,
                    reason=reason or "aspect_ratio_out_of_range",
                    bbox_wh_ratio=round(float(ratio), 4),
                    confidence=round(float(det.confidence), 4),
                    frame=frame_index,
                    bbox=[round(float(v), 2) for v in det.bbox],
                    min_ratio=self.config.geometry.min_aspect_ratio_wh,
                    max_ratio=self.config.geometry.max_aspect_ratio_wh,
                    technical_track_id=int(det.technical_track_id),
                )
                continue
            valid_detections.append(det)
        detections = valid_detections

        if self.budget is None:
            # Aucun FPS mesuré : les durées ne peuvent pas être converties en
            # frames, donc aucune décision n'est prise (règle 0.2). Seule
            # l'échelle locale est accumulée — sur les boîtes brutes, faute de
            # pouvoir appliquer la stabilisation à ce stade.
            self.scale.observe_many([detection.bbox for detection in detections])
            self._bootstrapping_frames += 1
            self._views = []
            return

        budget = self.budget
        self._advance_warmup(timestamp_s)
        margin_ratio = self.config.geometry.anchor_edge_margin_ratio

        # Stabilisation optionnelle (spec 7) : la boîte brute reste disponible et
        # la correction est journalisée avec sa raison et son seuil.
        raw_boxes: dict[int, np.ndarray] = {}
        corrected: dict[int, tuple[np.ndarray, tuple[float, float]]] = {}
        #: Par identifiant technique : (largeur courante, largeur stabilisée,
        #: ratio, seuil, signal, séparation de têtes en pixels).
        multi_person_detected: dict[
            int, tuple[float, float, float, float, str, float | None]
        ] = {}
        self._raw_heights.clear()
        self._fallback_keys.clear()
        for detection in detections:
            technical_id = int(detection.technical_track_id)
            raw = np.asarray(detection.bbox, dtype=float)
            stab_key = self.stabilization_key(technical_id)
            current_w = max(0.0, float(raw[2] - raw[0]))
            current_h = max(0.0, float(raw[3] - raw[1]))

            # -- Boîte multi-personnes : DEUX signaux indépendants ------------
            # (1) croissance anormale de la largeur par rapport à l'historique
            #     stabilisé de la même personne ;
            # (2) deux têtes séparées dans la boîte, signal **sans historique**.
            # Le signal (2) lève l'angle mort de (1) : deux personnes assises
            # côte à côte dès la première frame ne produisent jamais (1), leur
            # largeur fusionnée étant la référence. Il exige le modèle pose
            # (``presence.head_assist.enabled``) ; sinon (1) reste seul actif et
            # le comportement est inchangé.
            stable_w = 0.0
            ratio = 1.0
            thresh = float(self.config.geometry.multi_person_box.max_growth_ratio)
            width_signal = False
            multi_person_enabled = self.config.geometry.multi_person_box.enabled
            if multi_person_enabled and self.bbox_locker is not None:
                measured_w = self.bbox_locker.stable_width(stab_key)
                stable_h = self.bbox_locker.stable_height(stab_key)
                if measured_w is not None and measured_w > 0:
                    stable_w = float(measured_w)
                    width_signal, ratio, thresh = check_box_width_growth(
                        current_w,
                        [stable_w],
                        self.config.geometry.multi_person_box.max_growth_ratio,
                        current_height=current_h,
                        height_history=[stable_h] if stable_h else None,
                    )

            head_signal = False
            head_separation: float | None = None
            if multi_person_enabled and self.config.presence.head_assist.enabled:
                head_signal, head_separation = heads_suggest_merge(
                    detection.head_points or (),
                    current_w,
                    self.config.geometry.multi_person_box.head_separation_ratio,
                )
                if not head_signal:
                    head_separation = None

            if width_signal or head_signal:
                multi_person_detected[technical_id] = (
                    current_w,
                    stable_w,
                    ratio,
                    thresh,
                    "width_growth" if width_signal else "multiple_heads",
                    head_separation,
                )

            bbox = raw
            if self.bbox_locker is not None:
                # Historique indexé par ``person_id`` : une occultation ou un
                # changement d'identifiant technique ne doit pas réinitialiser la
                # hauteur de référence, sinon le verrouillage disparaîtrait
                # précisément sur la frame où la hauteur chute.
                bbox = np.asarray(
                    self.bbox_locker.process_bbox(
                        stab_key, raw
                    ),
                    dtype=float,
                )
            anchor = compute_anchor(bbox)
            if self.anchor_stabilizer is not None:
                anchor = tuple(
                    self.anchor_stabilizer.get_stabilized_anchor(
                        stab_key, bbox
                    )
                )
            raw_boxes[technical_id] = raw
            self._raw_heights[technical_id] = current_h
            corrected[technical_id] = (bbox, anchor)

        # Échelle locale (zone morte, rayon d'ambiguïté) : alimentée par la
        # hauteur **stabilisée** de chaque personne (BBoxHeightLocker, clé
        # ``person_id``), jamais par la hauteur brute instantanée. Alimentée par
        # la boîte brute, la zone morte grossissait et rétrécissait au rythme du
        # bruit de détection au lieu de suivre la personne. La hauteur stabilisée
        # est déjà calculée pour l'ancre : aucune seconde logique de lissage
        # n'est introduite, et le calcul reste causal (la frame courante compte).
        self._observe_local_scale(detections, corrected)
        dead_zone_px = self.scale.ratio_to_px(self.config.geometry.dead_zone_ratio)

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
                possible_multi_person=(
                    int(detection.technical_track_id) in multi_person_detected
                ),
                head_point=detection.head_point,
                head_confidence=detection.head_confidence,
            )
            for detection in detections
        ]

        with Timer() as reid_timer:
            assignments = self.identities.assign(
                observations, frame, timestamp_s, frame_index
            )
        if self.profiler is not None:
            self.profiler.record("reid", reid_timer.ms)

        ambiguity_radius_px = self.scale.ratio_to_px(
            self.config.geometry.occlusion_ambiguity_ratio
        )
        observed_person_ids = {
            int(assignment.person_id) for assignment in assignments if assignment.person_id
        }

        # Signalement explicite des boîtes suspectées de contenir plusieurs
        # personnes. Une personne **absorbée** dans une boîte fusionnée n'a plus
        # de détection propre : elle ne doit donc pas être purgée comme si elle
        # était sortie. Elle est déjà maintenue en borne haute par le chemin
        # normal (``OCCULTEE`` -> grâce -> purge incertaine, ``occupancy_impact="uncertain"``,
        # spec 4.4) ; l'arbitrage ci-dessous le rend **explicite et traçable** en
        # nommant les personnes déjà comptées et perdues dans le rayon
        # d'ambiguïté, sans jamais réduire ``occupancy_operational``.
        for assignment in assignments:
            tech_id = int(assignment.technical_track_id)
            if tech_id not in multi_person_detected or not assignment.person_id:
                continue
            curr_w, stab_w, r, th, signal, head_sep = multi_person_detected[tech_id]
            track = self.tracks.get(int(assignment.person_id))
            lost_neighbours = (
                []
                if track is None
                else self._lost_counted_neighbours_within(
                    track, assignment.anchor, ambiguity_radius_px, observed_person_ids
                )
            )
            self._emit(
                "POSSIBLE_MULTI_PERSON_BOX",
                timestamp_s,
                frame_index,
                person_id=int(assignment.person_id),
                frame=int(frame_index),
                current_width=round(float(curr_w), 2),
                stable_width=round(float(stab_w), 2),
                ratio=round(float(r), 4),
                threshold=round(float(th), 4),
                technical_track_id=tech_id,
                bbox=[round(float(v), 2) for v in assignment.bbox],
                signal=signal,
                head_separation_px=(
                    None if head_sep is None else round(float(head_sep), 2)
                ),
                ambiguity_radius_px=(
                    None if ambiguity_radius_px is None else round(float(ambiguity_radius_px), 2)
                ),
                lost_counted_neighbours=lost_neighbours,
            )
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
                    timestamp_s, frame_index, ambiguity_radius_px,
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
            recovered = self._recover_post_occlusion(
                detections, raw_detections, seen, frame, timestamp_s, frame_index, views
            )
            self._views = views
            for person_id, track in list(self.tracks.items()):
                if person_id in seen or person_id in recovered:
                    continue
                self._handle_missing(track, budget, timestamp_s, frame_index)
        if self.profiler is not None:
            self.profiler.record("fsm", fsm_timer.ms)

        self._report_warmup_end_if_pending(timestamp_s, frame_index, budget)
        self._publish_snapshot_if_due(timestamp_s, frame_index)
        for person_id in self.identities.release_expired(timestamp_s):
            # La purge a déjà été journalisée lors du passage à ABSENTE : la
            # libération de mémoire n'est donc jamais silencieuse (spec 3.4).
            self.tracks.pop(person_id, None)

    def _observe_local_scale(
        self,
        detections: Sequence[Detection],
        corrected: dict[int, tuple[np.ndarray, tuple[float, float]]],
    ) -> None:
        """Alimente l'échelle locale avec la hauteur **stabilisée** par personne.

        Sans verrouillage de hauteur, la boîte corrigée est la boîte brute : le
        comportement reste alors celui d'avant (aucun lissage à dupliquer).
        Lorsque le verrouillage est actif, ``h_stable`` (moyenne glissante par
        ``person_id``) est préféré à la hauteur corrigée : c'est la même
        grandeur que celle qui a servi à fixer l'ancre de la frame, et elle est
        déjà bornée par ``anchor.height_history_window_frames``.
        """
        for detection in detections:
            technical_id = int(detection.technical_track_id)
            bbox = corrected[technical_id][0]
            stabilized = (
                None
                if self.bbox_locker is None
                else self.bbox_locker.stable_height(self.stabilization_key(technical_id))
            )
            if stabilized is None:
                stabilized = float(bbox[3] - bbox[1])
            self.scale.observe(stabilized)

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
    def _advance_warmup(self, timestamp_s: float) -> None:
        """Décide si la frame courante appartient encore au warm-up.

        Frontière **explicite** : le warm-up couvre les frames ``1..K``, où
        ``K`` est la première frame pour laquelle ``K >= warmup_min_frames``
        **et** ``t_K - t_1 >= warmup_seconds``. La frame ``K`` appartient donc au
        warm-up ; ``_warmup_done`` ne passe à vrai qu'en arrivant sur la frame
        ``K+1``, qui est la première frame comptée. Les deux bornes sont exigées :
        la borne de frames protège des FPS élevés (15 frames en 0,15 s) et la
        borne de durée protège des FPS faibles (0,5 s pour 3 frames vues).
        """
        if self._warmup_done or self.budget is None:
            return
        if self._warmup_first_timestamp_s is None:
            self._warmup_first_timestamp_s = timestamp_s
            self._warmup_frames_observed = 1
            return
        elapsed = timestamp_s - self._warmup_first_timestamp_s
        if (
            self._warmup_frames_observed >= self.budget.warmup_min_frames
            and elapsed >= self.config.timing.warmup_seconds
        ):
            # ``warmup_frames_observed`` et ``warmup_elapsed_seconds`` décrivent
            # la **dernière frame de warm-up** (la frame K ci-dessus) : c'est bien
            # la durée du warm-up, pas la durée écoulée sur la première frame
            # comptée. Les deux bornes du budget (``warmup_min_frames`` et
            # ``warmup_frames``) sont publiées avec, pour que l'écart entre le
            # critère déclaré et le critère appliqué soit vérifiable après coup.
            self._warmup_done = True
            self._warmup_report_pending = True
            return
        self._warmup_frames_observed += 1
        self._warmup_elapsed_seconds = elapsed

    def _report_warmup_end_if_pending(
        self, timestamp_s: float, frame_index: int, budget: FrameBudget
    ) -> None:
        """Publie ``WARMUP_END`` après la première frame comptée.

        L'événement est émis **après** le traitement de la frame ``K+1`` : c'est
        sur cette frame que les règles de la table emploient l'effectif initial
        (``EMPLACE_INITIAL``). L'émettre plus tôt — à la fin de la dernière frame
        de warm-up — publierait systématiquement ``initial_occupancy = 0``, ce qui
        rendrait l'événement inexploitable pour l'audit.
        """
        if not self._warmup_report_pending:
            return
        self._warmup_report_pending = False
        self._emit(
            "WARMUP_END",
            timestamp_s,
            frame_index,
            initial_occupancy=self.initial_occupancy,
            warmup_frames_observed=self._warmup_frames_observed,
            warmup_elapsed_seconds=round(self._warmup_elapsed_seconds, 4),
            warmup_min_frames=budget.warmup_min_frames,
            budget_frames=budget.warmup_frames,
            fps=round(budget.fps, 3),
        )

    def _reference_feature(self, person_id: int) -> np.ndarray | None:
        record = self.identities.record(person_id)
        return None if record is None else record.feature

    def _recover_post_occlusion(
        self,
        detections: Sequence[Detection],
        raw_detections: np.ndarray | None,
        seen: set[int],
        frame: np.ndarray,
        timestamp_s: float,
        frame_index: int,
        views: list[_View],
    ) -> set[int]:
        """Post-Occlusion Recovery (étape 1) : visibilité seule.

        L'identité récupérée reste ``OCCULTEE`` pour la FSM et n'est pas passée
        à :meth:`_handle_missing` sur cette frame : aucune transition, aucun
        comptage. Sa boîte est celle de la détection brute ; son ancre n'est
        pas recalculée.
        """
        if self.recovery is None:
            return set()
        recovered = self.recovery.step(
            self.tracks, raw_detections, [d.bbox for d in detections], seen, frame,
            timestamp_s, frame_index, self.config.timing.grace_period_seconds,
        )
        for person_id, track in self.tracks.items():
            track.recovery_active = person_id in recovered
            if not track.recovery_active:
                continue
            box, confidence = recovered[person_id]
            track.bbox = np.asarray(box, dtype=float)
            track.confidence = float(confidence)
            track.last_seen_frame = frame_index
            track.last_seen_s = timestamp_s
            views.append(
                _View(
                    person_id=track.person_id,
                    technical_track_id=track.technical_track_id,
                    bbox=track.bbox,
                    anchor=track.anchor,
                    state=track.state,
                    provisional=track.provisional,
                )
            )
        return set(recovered)

    def _occlusion_ambiguity_radius_px(self) -> float | None:
        """Rayon (relatif à l'échelle locale) d'ambiguïté avec une personne occultée.

        Au-delà, une apparition intérieure ne peut plus être confondue avec une
        personne déjà comptée dont l'observation est perdue.
        """
        return self.scale.ratio_to_px(self.config.geometry.occlusion_ambiguity_ratio)

    def occluded_counted_neighbours(
        self, track: PersonTrack, anchor: tuple[float, float], radius_px: float | None
    ) -> list[int]:
        """``person_id`` déjà comptés, en occultation, à moins de ``radius_px``.

        C'est la condition qui interdit de confirmer un ``NEW`` : compter une
        nouvelle personne à l'endroit exact où une personne comptée vient d'être
        perdue produirait un doublon d'occupation. La proximité est exprimée en
        fraction de l'échelle locale (hauteur médiane des bbox), jamais en pixels
        absolus (règle 0.3).
        """
        if radius_px is None:
            return []
        neighbours: list[int] = []
        for person_id, other in self.tracks.items():
            if person_id == track.person_id or not other.inside_occupancy:
                continue
            if other.state not in (TrackState.OCCULTEE, TrackState.ABSENTE):
                continue
            distance = float(
                np.hypot(anchor[0] - other.anchor[0], anchor[1] - other.anchor[1])
            )
            if distance <= radius_px:
                neighbours.append(person_id)
        return sorted(neighbours)

    # ------------------------------------------------------------------
    # Suivi d'une personne observée
    # ------------------------------------------------------------------
    def _lost_counted_neighbours_within(
        self,
        track: PersonTrack,
        anchor: tuple[float, float],
        radius_px: float | None,
        observed: set[int] | None = None,
    ) -> list[int]:
        """Personnes déjà **comptées** et *perdues* dans le rayon d'ambiguïté.

        Réutilise :meth:`occluded_counted_neighbours` (même rayon,
        ``geometry.occlusion_ambiguity_ratio``, mêmes personnes comptées), en y
        ajoutant celles qui, sur la frame courante, n'ont **aucune détection** :
        une personne absorbée dans une boîte fusionnée perd sa détection propre
        au moment même de la fusion, donc son état n'est pas encore ``OCCULTEE``
        quand la boîte est signalée. Sans cette inclusion, la personne absorbée
        ne serait identifiée qu'après coup.
        """
        if radius_px is None:
            return []
        neighbours = set(self.occluded_counted_neighbours(track, anchor, radius_px))
        if observed:
            for person_id, other in self.tracks.items():
                if person_id in observed or person_id == track.person_id:
                    continue
                if not other.inside_occupancy:
                    continue
                distance = float(
                    np.hypot(anchor[0] - other.anchor[0], anchor[1] - other.anchor[1])
                )
                if distance <= radius_px:
                    neighbours.add(int(person_id))
        return sorted(neighbours)

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
            last_reliable_anchor=assignment.anchor if assignment.anchor_reliable else None,
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
        ambiguity_radius_px: float | None = None,
    ) -> None:
        previous_state = track.state
        anchor_cfg = self.config.anchor
        # La surveillance de chute doit voir la hauteur **mesurée** (boîte brute).
        # Alimentée par la hauteur corrigée, elle serait structurellement muette :
        # le verrouillage empêche par construction toute chute, donc le signal
        # « ancre non fiable » que la FSM consomme en priorité (règle
        # ``suspend_unreliable_anchor``) ne pourrait plus jamais être émis.
        raw_height = float(
            self._raw_heights.get(
                int(assignment.technical_track_id), assignment.bbox_height
            )
        )
        is_drop, drop_ratio, ref_h = check_anchor_height_drop(
            raw_height,
            track.height_history,
            anchor_cfg.max_height_drop_ratio,
        )

        effective_anchor = assignment.anchor
        if is_drop:
            track.height_drop_streak += 1
            track.drop_previous_height = ref_h
            track.drop_current_height = float(raw_height)
            track.drop_ratio = drop_ratio
            if track.height_drop_streak >= anchor_cfg.height_drop_confirm_frames:
                # La chute s'est stabilisée (personne assise et restée assise) :
                # réadaptation de l'historique de hauteur et réactivation de la fiabilité.
                #
                # La condition `not presence.head_assist.enabled` a été RETIRÉE :
                # elle neutralisait cette branche dès que l'assistance tête était
                # activée, si bien qu'une personne assise restait marquée
                # `anchor_unreliable` indéfiniment, sa coordonnée verticale gelée
                # sur `last_reliable_anchor`, et basculait en OCCULTEE à
                # l'expiration de `max_presence_extension_seconds`. La
                # fonctionnalité censée gérer les personnes assises dégradait
                # donc exactement le cas « personne assise ». Les deux
                # mécanismes sont complémentaires : réadaptation de hauteur ET
                # maintien par la tête.
                track.height_history = [float(raw_height)]
                track.height_drop_streak = 0
                track.anchor_unreliable_reason = None
                reliable = bool(assignment.anchor_reliable)
                effective_anchor = assignment.anchor
            else:
                # Chute brutale non confirmée : ancre non fiable, gel de la coordonnée verticale
                reliable = False
                track.anchor_unreliable_reason = "height_drop"
                if track.last_reliable_anchor is not None:
                    effective_anchor = (assignment.anchor[0], track.last_reliable_anchor[1])
        else:
            track.height_drop_streak = 0
            track.anchor_unreliable_reason = None
            track.height_history.append(float(raw_height))
            if len(track.height_history) > anchor_cfg.height_history_window_frames:
                track.height_history.pop(0)
            reliable = bool(assignment.anchor_reliable)
        if getattr(assignment, "feet_anchor_status", None) == "no_detection":
            reliable = False
            track.anchor_unreliable_reason = "no_detection"

        track.current_head_point = getattr(assignment, "head_point", None)
        track.current_head_confidence = getattr(assignment, "head_confidence", None)
        # La fraîcheur du point tête est indexée sur la frame d'**observation** :
        # le point est issu de la même détection que la boîte.
        track.head_point_frame_index = frame_index

        if reliable:
            track.last_reliable_anchor = effective_anchor
            track.head_presence_first_s = None
            track.head_presence_expired = False
        elif self.config.presence.head_assist.enabled and track.state is TrackState.PRESENTE:
            ha_cfg = self.config.presence.head_assist
            head_pt = assignment.head_point
            head_conf = assignment.head_confidence
            head_ok = (
                head_pt is not None
                and head_conf is not None
                and head_conf >= ha_cfg.min_keypoint_confidence
            )
            if head_ok:
                if track.head_presence_first_s is None:
                    track.head_presence_first_s = timestamp_s
                elapsed = timestamp_s - track.head_presence_first_s
                if elapsed <= ha_cfg.max_presence_extension_seconds:
                    status = track.anchor_unreliable_reason or "edge"
                    self._emit(
                        "PRESENCE_MAINTAINED_BY_HEAD",
                        timestamp_s,
                        frame_index,
                        person_id=track.person_id,
                        reason="feet_anchor_unreliable_head_confident",
                        head_confidence=round(float(head_conf), 4),
                        feet_anchor_status=status,
                        frame=frame_index,
                        head_point=[round(float(head_pt[0]), 2), round(float(head_pt[1]), 2)],
                    )
                else:
                    if not track.head_presence_expired:
                        track.head_presence_expired = True
                        self._emit(
                            "PRESENCE_EXTENSION_EXPIRED",
                            timestamp_s,
                            frame_index,
                            person_id=track.person_id,
                            duration_s=round(elapsed, 3),
                            max_extension_seconds=float(ha_cfg.max_presence_extension_seconds),
                            frame=frame_index,
                            reason="max_extension_exceeded",
                        )
                    self._mark_occluded(track, timestamp_s, frame_index)
                    track.state = TrackState.OCCULTEE
                    track.anchor = effective_anchor
                    track.bbox = np.asarray(assignment.bbox, dtype=float)
                    track.bbox_height = assignment.bbox_height
                    track.technical_track_id = assignment.technical_track_id
                    track.confidence = assignment.confidence
                    track.last_seen_frame = frame_index
                    track.last_seen_s = timestamp_s
                    track.aliases.add(assignment.technical_track_id)
                    self._emit_transition_if_changed(
                        track,
                        previous_state,
                        Decision(
                            rule="head_assist_extension_expired",
                            state_to=TrackState.OCCULTEE,
                            action=Action.MARQUE_OCCULTEE,
                            state_changed=True,
                        ),
                        None,
                        None,
                        timestamp_s,
                        frame_index,
                    )
                    return
            else:
                self._mark_occluded(track, timestamp_s, frame_index)
                track.state = TrackState.OCCULTEE
                track.anchor = effective_anchor
                track.bbox = np.asarray(assignment.bbox, dtype=float)
                track.bbox_height = assignment.bbox_height
                track.technical_track_id = assignment.technical_track_id
                track.confidence = assignment.confidence
                track.last_seen_frame = frame_index
                track.last_seen_s = timestamp_s
                track.aliases.add(assignment.technical_track_id)
                self._emit_transition_if_changed(
                    track,
                    previous_state,
                    Decision(
                        rule="head_assist_unreliable_head",
                        state_to=TrackState.OCCULTEE,
                        action=Action.MARQUE_OCCULTEE,
                        state_changed=True,
                    ),
                    None,
                    None,
                    timestamp_s,
                    frame_index,
                )
                return

        if self.no_line:
            distance = 0.0
            side = Side.INTERIEURE
            # Même condition d'échelle que le mode avec ligne (zone_of rend
            # None sans échelle locale), sans zone morte ni extérieur.
            zone = Zone.INTERIEURE if dead_zone_px is not None else None
        else:
            distance = self.line.distance(effective_anchor, width, height)
            side = side_of(distance, self.line.inside_side, self.line.on_line_policy)
            zone = zone_of(distance, dead_zone_px, self.line.inside_side)

        if reliable and zone is Zone.EXTERIEURE:
            # Historique extérieur : dès qu'il existe, ``NEW`` est interdit par
            # la table — la personne entre (franchissement ou entrée signalée),
            # elle n'apparaît pas.
            track.observed_outside = True
        blocked_by_occlusion = bool(
            self.occluded_counted_neighbours(
                track, effective_anchor, ambiguity_radius_px
            )
        )
        if not blocked_by_occlusion:
            # La situation ambiguë a cessé : un futur report sera re-journalisé.
            track.new_deferred_reported = False

        crossed: str | None = None
        if reliable and track.previous_anchor is not None and not self.no_line:
            crossed = self.line.crossing(
                track.previous_anchor, effective_anchor, width, height
            )
        if reliable:
            # Seule une ancre fiable alimente la référence de franchissement
            # (spec 5.4) : pendant une suspension, la référence est gelée.
            track.previous_anchor = effective_anchor
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
            exterior_history=track.observed_outside,
            new_blocked_by_occlusion=blocked_by_occlusion,
        )
        decision = self.machine.resolve(track.state, context)
        self._apply_decision(
            track, decision, context, assignment, width, height, timestamp_s, frame_index
        )

        track.anchor = effective_anchor
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
    def _head_point_is_fresh(self, track: PersonTrack, frame_index: int) -> bool:
        """Le point tête est-il issu de la frame courante ou de la précédente ?

        ``_handle_missing`` relisait ``track.current_head_point`` figé lors de la
        **dernière frame où la personne était observée**. Les keypoints sont
        extraits de ``result.keypoints[i]``, indexés sur la même détection que la
        boîte : pas de boîte, pas de point tête. Le code pouvait donc émettre
        ``PRESENCE_MAINTAINED_BY_HEAD`` avec ``feet_anchor_status="no_detection"``
        sur la base d'une donnée sans lien avec l'image courante, et maintenir
        jusqu'à ``max_presence_extension_seconds`` une personne réellement sortie
        du champ. Au-delà de la tolérance, le chemin normal s'applique :
        ``OCCULTEE``, grâce, purge.
        """
        if track.head_point_frame_index is None:
            return False
        tolerance = self.config.presence.head_assist.max_head_staleness_frames
        return (int(frame_index) - int(track.head_point_frame_index)) <= int(tolerance)

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
        if (
            self.config.presence.head_assist.enabled
            and track.state is TrackState.PRESENTE
            and track.current_head_point is not None
            and track.current_head_confidence is not None
            and track.current_head_confidence >= self.config.presence.head_assist.min_keypoint_confidence
            and self._head_point_is_fresh(track, frame_index)
        ):
            ha_cfg = self.config.presence.head_assist
            if track.head_presence_first_s is None:
                track.head_presence_first_s = timestamp_s
            elapsed = timestamp_s - track.head_presence_first_s
            if elapsed <= ha_cfg.max_presence_extension_seconds:
                self._emit(
                    "PRESENCE_MAINTAINED_BY_HEAD",
                    timestamp_s,
                    frame_index,
                    person_id=track.person_id,
                    reason="feet_anchor_unreliable_head_confident",
                    head_confidence=round(float(track.current_head_confidence), 4),
                    feet_anchor_status="no_detection",
                    frame=frame_index,
                    head_point=[round(float(track.current_head_point[0]), 2), round(float(track.current_head_point[1]), 2)],
                )
                return
            else:
                if not track.head_presence_expired:
                    track.head_presence_expired = True
                    self._emit(
                        "PRESENCE_EXTENSION_EXPIRED",
                        timestamp_s,
                        frame_index,
                        person_id=track.person_id,
                        duration_s=round(elapsed, 3),
                        max_extension_seconds=float(ha_cfg.max_presence_extension_seconds),
                        frame=frame_index,
                        reason="max_extension_exceeded",
                    )
                self._mark_occluded(track, timestamp_s, frame_index)
                track.state = TrackState.OCCULTEE
                self._emit_transition_if_changed(
                    track,
                    previous_state,
                    Decision(
                        rule="head_assist_extension_expired",
                        state_to=TrackState.OCCULTEE,
                        action=Action.MARQUE_OCCULTEE,
                        state_changed=True,
                    ),
                    None,
                    None,
                    timestamp_s,
                    frame_index,
                )
                return
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
            exterior_history=track.observed_outside,
            new_blocked_by_occlusion=False,
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
                bbox_val = (
                    [round(float(v), 2) for v in assignment.bbox]
                    if assignment is not None and getattr(assignment, "bbox", None) is not None
                    else [round(float(v), 2) for v in track.bbox]
                    if track.bbox is not None
                    else None
                )
                if track.anchor_unreliable_reason == "height_drop":
                    self._emit(
                        "ANCHOR_UNRELIABLE",
                        timestamp_s,
                        frame_index,
                        person_id=track.person_id,
                        reason="height_drop",
                        previous_height=round(float(track.drop_previous_height), 2),
                        current_height=round(float(track.drop_current_height), 2),
                        drop_ratio=round(float(track.drop_ratio), 4),
                        frame=frame_index,
                        bbox=bbox_val,
                        consecutive_frames=1,
                    )
                else:
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
                        bbox=bbox_val,
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

        elif action is Action.SIGNALE_ENTREE_SANS_FRANCHISSEMENT:
            # Symétrique de `inside_drift_outside_without_crossing` : le demi-plan
            # a changé vers l'intérieur sans intersection capturée (ligne non
            # couvrante ou saut de suivi). L'entrée est **signalée** puis soumise
            # à confirmation par la zone morte ; elle n'est ni ignorée ni
            # comptée comme une nouvelle présence.
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="entry_side_change_without_crossing",
                person_id=track.person_id,
                technical_track_id=int(technical_id),
                details=(
                    "Demi-plan passé à l'intérieur sans intersection de la ligne "
                    "capturée, alors que la personne a déjà été observée à "
                    "l'extérieur"
                ),
                resolution="entrée soumise à confirmation (jamais comptée comme NEW)",
            )
            track.pending_crossing = "in"

        elif action is Action.DIFFERE_NOUVELLE_PRESENCE:
            # Une personne déjà comptée est occultée à proximité : confirmer un
            # NEW ici risquerait de compter deux fois la même personne. La
            # décision est différée, et journalisée une fois par épisode.
            if not track.new_deferred_reported:
                track.new_deferred_reported = True
                self._emit(
                    "INCONSISTENT_STATE",
                    timestamp_s,
                    frame_index,
                    kind="new_deferred_occluded_person_nearby",
                    person_id=track.person_id,
                    technical_track_id=int(technical_id),
                    details=(
                        "Apparition intérieure stable alors que des personnes "
                        "déjà comptées sont en occultation à proximité : NEW "
                        "n'est pas confirmé, le doublon est possible"
                    ),
                    resolution=(
                        "NEW différé ; il sera confirmé si la situation se "
                        "clarifie (sortie confirmée ou réapparition cohérente)"
                    ),
                )

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
        """Confirme une nouvelle présence — sous conditions strictes (spec 4).

        La table de :mod:`fsm` n'y mène déjà que pour une personne jamais vue à
        l'extérieur et sans personne comptée occultée à proximité ; ce contrôle
        est répété ici en défense en profondeur (même principe que ``_count_out``
        et son ``out_without_in``) : une règle mal configurée ne doit jamais
        pouvoir incrémenter l'occupation en silence.
        """
        if track.inside_occupancy:
            return
        if track.pending_crossing is not None:
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="new_with_pending_crossing",
                person_id=track.person_id,
                technical_track_id=int(technical_id),
                details=(
                    "Apparition intérieure alors qu'un franchissement "
                    f"({track.pending_crossing}) est amorcé : NEW refusé"
                ),
                resolution="NEW non compté ; le franchissement amorcé reste la décision",
            )
            return
        if track.observed_outside:
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="new_after_exterior_observation",
                person_id=track.person_id,
                technical_track_id=int(technical_id),
                details=(
                    "La personne a déjà été observée à l'extérieur : son "
                    "apparition intérieure est une entrée, pas une nouvelle présence"
                ),
                resolution="NEW non compté",
            )
            return
        neighbours = self.occluded_counted_neighbours(
            track, track.anchor, self._occlusion_ambiguity_radius_px()
        )
        if neighbours:
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="new_with_occluded_candidates",
                person_id=track.person_id,
                technical_track_id=int(technical_id),
                details=f"Personnes déjà comptées et occultées à proximité : {neighbours}",
                resolution="NEW non compté, décision différée",
            )
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
        """Vérifie l'invariant ``opérationnel == observable + incertain``.

        Aucune valeur n'est bornée ni corrigée : l'incohérence est publiée telle
        quelle et journalisée (règle 0.4). Un ``max(0, ...)`` masquerait
        exactement le défaut qu'il faut voir.
        """
        ledger = self.occupancy_operational
        if ledger < 0:
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="negative_occupancy_before_clamp",
                details=f"Bilan d'occupation négatif : {ledger}",
                resolution="valeur publiée bornée, incohérence journalisée (jamais silencieuse)",
            )
            return
        observable = self.occupancy_observed + self.occupancy_uncertain
        if ledger != observable:
            self._emit(
                "INCONSISTENT_STATE",
                timestamp_s,
                frame_index,
                kind="occupancy_ledger_mismatch",
                details=(
                    f"Occupation opérationnelle {ledger} != observable {observable} "
                    f"(observée {self.occupancy_observed} + incertaine "
                    f"{self.occupancy_uncertain})"
                ),
                resolution=(
                    "incohérence publiée et journalisée : elle signale une "
                    "identité comptée dont l'enregistrement logique a été perdu"
                ),
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
        self.draw_line_overlay(frame)

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

    def dead_zone_px(self) -> float | None:
        """Demi-largeur courante de la zone morte, en pixels (``None`` sans échelle)."""
        return self.scale.ratio_to_px(self.config.geometry.dead_zone_ratio)

    def draw_line_overlay(self, frame: np.ndarray) -> None:
        """Ligne, zone morte et flèche seules : même dessin que la vue web.

        Délègue à :func:`calibration.draw_line_overlay`, dessin unique de ces
        trois éléments (fenêtre OpenCV, vue web, relecture).
        """
        if self.line is not None:
            draw_line_overlay(frame, self.line, self.dead_zone_px())

    def _draw_hud(self, frame: np.ndarray) -> None:
        _height, width = frame.shape[:2]
        scale = max(0.5, min(1.0, width / 1280.0))
        snapshot = self.snapshot()
        lines = [
            f"OCCUPATION CONFIRMEE: {snapshot.confirmed}",
            f"OBSERVEE: {snapshot.observed}  INCERTAINE: {snapshot.uncertain}  "
            f"RANGE: [{snapshot.range_low}, {snapshot.range_high}]",
            f"IN: {snapshot.total_in}  OUT: {snapshot.total_out}  NEW: {snapshot.total_new}",
            f"Visibles: {snapshot.visible}  Occultees: {snapshot.occluded}  Phase: {self.phase}",
        ]
        if self.no_line:
            lines.append("MODE SANS LIGNE : effectif = warm-up + NEW")
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
