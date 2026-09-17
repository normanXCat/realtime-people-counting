"""Gestionnaire d'identités à deux niveaux (spec section 3).

Ce module est la **fusion** des trois implémentations concurrentes relevées à
l'audit (§2.1 a) : ``IDManager`` (dans `main.py`), ``ReIDGallery`` (`reid.py`) et
``HybridAssociation`` (`hybrid_tracker.py`). Il ne reste qu'une galerie, un
seuil et une politique.

Les trois identifiants de la spec 3.1 sont explicites :

- ``technical_track_id`` : identifiant interne BoT-SORT, instable, **jamais**
  utilisé directement pour le comptage ;
- ``person_id`` : identifiant logique stable, qui porte le comptage ;
- ``event_id`` : porté par le journal (`events.py`), pas par ce module.

Niveau 1 (court terme) : les IDs techniques restent attribués par BoT-SORT et
sa persistance native (réassociation intra-buffer via ``with_reid``).

Niveau 2 (long terme) : galerie de personnes ré-associables selon l'algorithme
de la spec 3.3 — filtre spatio-temporel, seuil de similarité, marge de sécurité
vis-à-vis du second candidat, journalisation systématique de
``REID_MATCH`` / ``REID_NEW`` / ``REID_AMBIGUOUS``.

Ce qui a été repris de ``HybridAssociation`` : la contrainte spatio-temporelle
(vitesse maximale × hauteur de bbox × Δt) et la règle de qualité de
descripteur avant écriture en galerie. Ce qui a été abandonné : le
déclenchement « à l'occlusion imminente », l'assignation hongroise, l'état
``raw_to_stable`` parallèle et toutes les références ByteTrack/DeepSORT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from config import ExternalReidConfig, LongTermReidConfig
from events import EventSink


# ---------------------------------------------------------------------------
# Extraction de descripteurs d'apparence
# ---------------------------------------------------------------------------
class AppearanceExtractor(Protocol):
    """Calcule un descripteur L2-normalisé à partir d'un crop de personne."""

    name: str

    def describe(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        ...  # pragma: no cover - protocole


class HistogramAppearanceExtractor:
    """Descripteur léger couleur (HSV), sans dépendance à torch.

    C'est l'extracteur par défaut : rapide, déterministe, sans téléchargement de
    poids. Il reprend la signature de ``IDManager.appearance`` (audit §2.1 a).
    """

    name = "histogram_hsv"

    def __init__(
        self,
        bins_hue: int = 24,
        bins_saturation: int = 8,
        min_width: int = 8,
        min_height: int = 16,
    ) -> None:
        self.bins_hue = int(bins_hue)
        self.bins_saturation = int(bins_saturation)
        self.min_width = int(min_width)
        self.min_height = int(min_height)

    def describe(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < self.min_width or y2 - y1 < self.min_height:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.bins_hue, self.bins_saturation], [0, 180, 0, 256],
        )
        flat = histogram.flatten().astype(np.float32)
        norm = float(np.linalg.norm(flat))
        return None if norm < 1e-8 else flat / norm


class DeepAppearanceExtractor:
    """Descripteur d'apparence profond (ex-``reid.py``), ablation de la section 9.

    Chargement paresseux : aucun poids n'est téléchargé à l'import. OSNet
    (``torchreid``) est utilisé si disponible, sinon le repli MobileNetV3 Small
    tronqué — exactement le comportement consolidé des deux modules supprimés.
    """

    name = "deep_mobilenetv3"

    def __init__(
        self,
        config: ExternalReidConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or ExternalReidConfig()
        self._device_override = device
        self._model: Any = None
        self._transform: Any = None
        self._device: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch  # import local : le chemin par défaut n'a pas besoin de torch
        import torch.nn as nn
        from torchvision import models, transforms

        device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)
        height, width = self.config.image_size
        try:
            import torchreid  # type: ignore

            self._model = (
                torchreid.models.build_model(
                    name=self.config.weights, num_classes=1000, pretrained=True
                )
                .eval()
                .to(self._device)
            )
            self.name = f"deep_{self.config.weights}"
        except Exception:
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            backbone.classifier = nn.Identity()
            self._model = backbone.eval().to(self._device)
            self.name = "deep_mobilenetv3"
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def describe(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            # Crop inexploitable : on refuse **avant** de charger un modèle, ce
            # qui évite tout téléchargement de poids pour rien.
            return None
        import torch

        self._ensure_model()
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            vector = (
                self._model(self._transform(crop).unsqueeze(0).to(self._device))
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        norm = float(np.linalg.norm(vector))
        return None if norm < 1e-6 else vector / norm


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """Une piste technique observée sur la frame courante."""

    technical_track_id: int
    bbox: np.ndarray
    confidence: float
    anchor: tuple[float, float]
    bbox_height: float
    feature: np.ndarray | None = None
    anchor_reliable: bool = True


@dataclass
class Assignment:
    """Résultat d'affectation d'une observation à une identité logique."""

    technical_track_id: int
    person_id: int
    bbox: np.ndarray
    confidence: float
    anchor: tuple[float, float]
    bbox_height: float
    provisional: bool = False
    matched: bool = False
    similarity: float | None = None
    runner_up_similarity: float | None = None
    candidates_evaluated: int = 0
    allowed_distance: float | None = None
    distance: float | None = None
    reason: str = ""
    anchor_reliable: bool = True


@dataclass
class IdentityRecord:
    """État d'identité détenu par le gestionnaire (niveau long terme)."""

    person_id: int
    technical_track_id: int
    feature: np.ndarray | None
    anchor: tuple[float, float]
    bbox_height: float
    bbox: np.ndarray | None
    first_seen_s: float
    last_seen_s: float
    live: bool = True
    provisional: bool = False
    purged_at_s: float | None = None
    purge_reason: str | None = None
    observations: int = 0
    aliases: set[int] = field(default_factory=set)
    #: Dernière raison de rejet d'apparence (`REID_DESCRIPTOR_REJECTED`). Sert à
    #: ne journaliser qu'un changement de situation, jamais une ligne par frame.
    last_rejection: str | None = None

    @property
    def age_s(self) -> float:
        return self.last_seen_s - self.first_seen_s


# ---------------------------------------------------------------------------
# Gestionnaire
# ---------------------------------------------------------------------------
class IdentityManager:
    """Convertit les pistes techniques en ``person_id`` stables (spec 3.2, 3.3).

    La galerie n'accepte que des apparences **exploitables** : les seuils
    (confiance minimale, taille minimale de crop, netteté, ancre fiable) vivent
    dans :class:`config.LongTermReidConfig`, jamais en dur ici (règle 0.3). Un
    descripteur refusé ne remplace **jamais** l'apparence connue d'une identité
    et le refus est journalisé une fois par changement de motif.
    """

    def __init__(
        self,
        long_term: LongTermReidConfig | None = None,
        external_reid: ExternalReidConfig | None = None,
        grace_period_seconds: float = 5.0,
        appearance: AppearanceExtractor | None = None,
        event_sink: EventSink | None = None,
        feature_momentum: float = 0.85,
    ) -> None:
        self.long_term = long_term or LongTermReidConfig()
        self.external_reid = external_reid or ExternalReidConfig()
        self.grace_period_seconds = float(grace_period_seconds)
        #: Durée de rétention d'une identité **purgée** dans la galerie, pour
        #: permettre une réassociation « récemment purgée » (spec 3.3), puis
        #: libération silencieuse de la mémoire : c'est une mémoire cache, pas
        #: une identité comptée — la purge du ``person_id`` a déjà été
        #: journalisée au moment de son passage à ``ABSENTE`` (spec 3.4).
        self.purge_retention_seconds = self.grace_period_seconds
        self.appearance: AppearanceExtractor = appearance or (
            DeepAppearanceExtractor(self.external_reid)
            if self.external_reid.enabled
            else HistogramAppearanceExtractor()
        )
        self.event_sink = event_sink
        self.feature_momentum = float(feature_momentum)

        self.records: dict[int, IdentityRecord] = {}
        self.technical_to_person: dict[int, int] = {}
        self._next_person_id = 1
        self._claimed: set[int] = set()
        #: Identités provisoires issues d'une décision ambiguë (spec 3.3 étape 5).
        self.provisional_ids: set[int] = set()
        #: Diagnostics (compteurs publiés dans le résumé de session).
        self.descriptor_rejections = 0
        self.technical_id_changes = 0

    # -- API de lecture ----------------------------------------------------
    @property
    def live_count(self) -> int:
        return sum(1 for record in self.records.values() if record.live)

    @property
    def gallery_size(self) -> int:
        return sum(1 for record in self.records.values() if not record.live)

    def record(self, person_id: int) -> IdentityRecord | None:
        return self.records.get(person_id)

    def person_of(self, technical_track_id: int) -> int | None:
        return self.technical_to_person.get(int(technical_track_id))

    # -- Descripteurs ------------------------------------------------------
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator > 1e-8 else 0.0

    def describe(self, frame: np.ndarray | None, bbox: Sequence[float]) -> np.ndarray | None:
        if frame is None:
            return None
        return self.appearance.describe(frame, bbox)

    # -- Qualité du descripteur avant écriture en galerie -------------------
    @staticmethod
    def sharpness(frame: np.ndarray, bbox: Sequence[float]) -> float | None:
        """Netteté d'un crop : variance du Laplacien (0 = image uniforme).

        Une personne légèrement floue produit un descripteur d'apparence peu
        informatif : l'écrire en galerie dégraderait durablement la
        réassociation. Le contrôle n'est actif que si
        ``reid.long_term.gallery_min_sharpness`` > 0.
        """
        if frame is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _gallery_rejection(
        self,
        observation: Observation,
        feature: np.ndarray | None,
        frame: np.ndarray | None,
    ) -> tuple[str | None, float | None, float | None]:
        """Motif de rejet d'une apparence pour la galerie, ou ``None``.

        Returns:
            ``(raison, seuil, valeur_mesurée)``. ``raison`` est l'un de
            ``low_confidence``, ``tiny_crop``, ``edge_truncated``,
            ``blurred_or_unreliable`` — ou ``None`` si l'apparence est
            exploitable.
        """
        limits = self.long_term
        if observation.confidence < limits.gallery_min_confidence:
            return "low_confidence", float(limits.gallery_min_confidence), float(
                observation.confidence
            )
        bbox = np.asarray(observation.bbox, dtype=float)
        box_height = float(bbox[3] - bbox[1])
        box_width = float(bbox[2] - bbox[0])
        if (
            box_height < limits.gallery_min_crop_height_px
            or box_width < limits.gallery_min_crop_width_px
        ):
            return "tiny_crop", float(limits.gallery_min_crop_height_px), box_height
        if not observation.anchor_reliable:
            return "edge_truncated", None, None
        if feature is None:
            return "blurred_or_unreliable", None, None
        if limits.gallery_min_sharpness > 0 and frame is not None:
            measured = self.sharpness(frame, bbox)
            if measured is not None and measured < limits.gallery_min_sharpness:
                return "blurred_or_unreliable", float(limits.gallery_min_sharpness), measured
        return None, None, None

    # -- Affectation -------------------------------------------------------
    def assign(
        self,
        observations: Sequence[Observation],
        frame: np.ndarray | None,
        timestamp_s: float,
        frame_index: int,
    ) -> list[Assignment]:
        """Affecte chaque observation à un ``person_id`` (création ou réassociation).

        Toutes les identités sont d'abord marquées non observées : une identité
        ne redevient vivante que si elle est effectivement affectée sur cette
        frame. C'est cette bascule qui définit la galerie (identités perdues),
        et qui rend le filtre spatio-temporel utilisable dès la frame suivante.
        """
        self._claimed = set()
        for record in self.records.values():
            record.live = False
        assignments: list[Assignment] = []
        for observation in observations:
            assignment = self._assign_one(observation, frame, timestamp_s, frame_index)
            assignments.append(assignment)
            if assignment.person_id:
                self._claimed.add(assignment.person_id)
        return assignments

    def _assign_one(
        self,
        observation: Observation,
        frame: np.ndarray | None,
        timestamp_s: float,
        frame_index: int,
    ) -> Assignment:
        technical_id = int(observation.technical_track_id)
        feature = observation.feature
        if feature is None:
            feature = self.describe(frame, observation.bbox)

        known_person = self.technical_to_person.get(technical_id)
        if known_person is not None and known_person in self.records:
            record = self.records[known_person]
            if record.person_id in self._claimed:
                # Spec 4.5 : deux pistes ne peuvent pas viser la même identité.
                self._inconsistent(
                    "duplicate_person_claim",
                    timestamp_s,
                    frame_index,
                    person_id=record.person_id,
                    technical_track_id=technical_id,
                    details="Deux pistes techniques visent le même person_id sur la même frame",
                    resolution="nouveau person_id attribué à la seconde piste",
                )
                return self._create_new(
                    observation, feature, timestamp_s, frame_index,
                    reason="duplicate_claim_new_identity",
                    frame=frame,
                )
            self._touch(record, observation, feature, timestamp_s, frame_index, frame)
            return Assignment(
                technical_track_id=technical_id,
                person_id=record.person_id,
                bbox=observation.bbox,
                confidence=observation.confidence,
                anchor=observation.anchor,
                bbox_height=observation.bbox_height,
                provisional=record.provisional,
                reason="technical_track_locked",
                anchor_reliable=observation.anchor_reliable,
            )

        if not self.long_term.enabled:
            return self._create_new(
                observation, feature, timestamp_s, frame_index,
                reason="long_term_reid_disabled",
                frame=frame,
            )

        candidates = self._plausible_candidates(observation, timestamp_s)
        if not candidates:
            return self._create_new(
                observation, feature, timestamp_s, frame_index,
                reason="no_plausible_candidate",
                candidates_evaluated=0,
                frame=frame,
            )

        if feature is None:
            return self._create_new(
                observation, feature, timestamp_s, frame_index,
                reason="no_descriptor",
                candidates_evaluated=len(candidates),
                frame=frame,
            )

        scored = sorted(
            (
                (self.cosine_similarity(feature, record.feature), person_id, allowed, distance)
                for person_id, record, allowed, distance in candidates
                if record.feature is not None
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not scored:
            return self._create_new(
                observation, feature, timestamp_s, frame_index,
                reason="candidates_without_descriptor",
                candidates_evaluated=len(candidates),
                frame=frame,
            )

        best_similarity, best_person, allowed, distance = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else None
        threshold = self.long_term.similarity_threshold
        margin = self.long_term.safety_margin

        if best_similarity < threshold:
            return self._create_new(
                observation, feature, timestamp_s, frame_index,
                reason="below_similarity_threshold",
                candidates_evaluated=len(scored),
                best_similarity=best_similarity,
                frame=frame,
            )

        if runner_up is not None and (best_similarity - runner_up) < margin:
            # Spec 3.3 étape 5 : jamais de fusion silencieuse.
            return self._resolve_ambiguous(
                observation, feature, best_person, best_similarity, runner_up,
                [person for _score, person, _a, _d in scored],
                timestamp_s, frame_index, allowed, distance, frame,
            )

        return self._attach_existing(
            observation, feature, best_person, timestamp_s,
            similarity=best_similarity,
            runner_up=runner_up,
            candidates_evaluated=len(scored),
            allowed=allowed,
            distance=distance,
            reason="reid_match",
            frame_index=frame_index,
            frame=frame,
        )

    # -- Mise en relation --------------------------------------------------
    def _attach_existing(
        self,
        observation: Observation,
        feature: np.ndarray | None,
        person_id: int,
        timestamp_s: float,
        similarity: float | None,
        runner_up: float | None,
        candidates_evaluated: int,
        allowed: float | None,
        distance: float | None,
        reason: str,
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> Assignment:
        record = self.records[person_id]
        new_technical_id = int(observation.technical_track_id)
        previous_technical_id = record.technical_track_id
        self.technical_to_person[new_technical_id] = person_id
        record.aliases.add(new_technical_id)
        if previous_technical_id is not None and previous_technical_id != new_technical_id:
            # Un nouvel identifiant technique est rattaché à la même personne :
            # c'est la trace explicite de la perte/reprise de piste (flou,
            # occultation). Sans cet événement, le changement d'identifiant
            # technique serait invisible dans le journal.
            self.technical_id_changes += 1
            self._emit(
                "TECHNICAL_ID_CHANGED", timestamp_s, frame_index,
                person_id=person_id,
                technical_track_id=new_technical_id,
                previous_technical_track_id=previous_technical_id,
                decision="reassigned_to_existing_person",
                reason=reason,
                similarity=None if similarity is None else round(similarity, 4),
                distance=None if distance is None else round(distance, 2),
                allowed_distance=None if allowed is None else round(allowed, 2),
                descriptor_reliable=self._gallery_rejection(observation, feature, frame)[0]
                is None,
            )
        self._touch(record, observation, feature, timestamp_s, frame_index, frame)
        if similarity is not None:
            self._emit(
                "REID_MATCH", timestamp_s, frame_index,
                person_id=person_id,
                technical_track_id=int(observation.technical_track_id),
                similarity=round(similarity, 4),
                runner_up_similarity=None if runner_up is None else round(runner_up, 4),
                candidates_evaluated=candidates_evaluated,
                distance=None if distance is None else round(distance, 2),
                allowed_distance=None if allowed is None else round(allowed, 2),
                winner_margin=None if runner_up is None else round(similarity - runner_up, 4),
            )
        return Assignment(
            technical_track_id=int(observation.technical_track_id),
            person_id=person_id,
            bbox=observation.bbox,
            confidence=observation.confidence,
            anchor=observation.anchor,
            bbox_height=observation.bbox_height,
            provisional=record.provisional,
            matched=True,
            similarity=similarity,
            runner_up_similarity=runner_up,
            candidates_evaluated=candidates_evaluated,
            allowed_distance=allowed,
            distance=distance,
            reason=reason,
            anchor_reliable=observation.anchor_reliable,
        )

    def _resolve_ambiguous(
        self,
        observation: Observation,
        feature: np.ndarray,
        best_person: int,
        best_similarity: float,
        runner_up: float,
        candidate_ids: list[int],
        timestamp_s: float,
        frame_index: int,
        allowed: float | None,
        distance: float | None,
        frame: np.ndarray | None = None,
    ) -> Assignment:
        """Décision ambiguë : jamais de fusion arbitraire (spec 3.3 étape 5)."""
        policy = self.long_term.ambiguous_policy
        self._emit(
            "REID_AMBIGUOUS", timestamp_s, frame_index,
            technical_track_id=int(observation.technical_track_id),
            best_similarity=round(best_similarity, 4),
            runner_up_similarity=round(runner_up, 4),
            safety_margin=float(self.long_term.safety_margin),
            candidates=candidate_ids,
            resolution=policy,
            distance=None if distance is None else round(distance, 2),
            allowed_distance=None if allowed is None else round(allowed, 2),
        )
        if policy == "defer":
            # Aucun person_id n'est attribué : la piste reste en attente et
            # l'appelant ne prend aucune décision de comptage cette frame.
            return Assignment(
                technical_track_id=int(observation.technical_track_id),
                person_id=0,
                bbox=observation.bbox,
                confidence=observation.confidence,
                anchor=observation.anchor,
                bbox_height=observation.bbox_height,
                similarity=best_similarity,
                runner_up_similarity=runner_up,
                candidates_evaluated=len(candidate_ids),
                allowed_distance=allowed,
                distance=distance,
                reason="reid_ambiguous_deferred",
                anchor_reliable=observation.anchor_reliable,
            )
        assignment = self._create_new(
            observation, feature, timestamp_s, frame_index,
            reason="reid_ambiguous_provisional",
            candidates_evaluated=len(candidate_ids),
            best_similarity=best_similarity,
            frame=frame,
        )
        # Identité provisoire : elle compte comme une nouvelle personne, mais
        # reste marquée incertaine pour l'audit des décisions ambiguës (spec 3.3).
        assignment.provisional = True
        assignment.similarity = best_similarity
        assignment.runner_up_similarity = runner_up
        record = self.records.get(assignment.person_id)
        if record is not None:
            record.provisional = True
        self.provisional_ids.add(assignment.person_id)
        return assignment

    def _create_new(
        self,
        observation: Observation,
        feature: np.ndarray | None,
        timestamp_s: float,
        frame_index: int,
        reason: str,
        candidates_evaluated: int = 0,
        best_similarity: float | None = None,
        deferred: bool = False,
        frame: np.ndarray | None = None,
    ) -> Assignment:
        person_id = self._next_person_id
        self._next_person_id += 1
        # L'apparence initiale passe le même contrôle de qualité que les mises
        # à jour : un descripteur non exploitable n'entre pas dans la galerie
        # (il ne servirait qu'à produire de fausses réassociations), et le refus
        # est journalisé plutôt que silencieux.
        rejection, threshold, measured = self._gallery_rejection(observation, feature, frame)
        self.records[person_id] = IdentityRecord(
            person_id=person_id,
            technical_track_id=int(observation.technical_track_id),
            feature=feature if rejection is None else None,
            anchor=observation.anchor,
            bbox_height=observation.bbox_height,
            bbox=np.asarray(observation.bbox, dtype=float),
            first_seen_s=timestamp_s,
            last_seen_s=timestamp_s,
            live=True,
            observations=1,
            aliases={int(observation.technical_track_id)},
        )
        self.technical_to_person[int(observation.technical_track_id)] = person_id
        if rejection is not None:
            self._report_descriptor_rejection(
                self.records[person_id], observation, rejection, threshold, measured,
                timestamp_s, frame_index,
            )
        self._emit(
            "REID_NEW", timestamp_s, frame_index,
            person_id=person_id,
            technical_track_id=int(observation.technical_track_id),
            reason=reason,
            candidates_evaluated=candidates_evaluated,
            best_similarity=None if best_similarity is None else round(best_similarity, 4),
            deferred=deferred,
        )
        return Assignment(
            technical_track_id=int(observation.technical_track_id),
            person_id=person_id,
            bbox=observation.bbox,
            confidence=observation.confidence,
            anchor=observation.anchor,
            bbox_height=observation.bbox_height,
            matched=False,
            candidates_evaluated=candidates_evaluated,
            reason=reason,
            anchor_reliable=observation.anchor_reliable,
        )

    # -- Filtre spatio-temporel (spec 3.3 étape 1) -------------------------
    def _plausible_candidates(
        self, observation: Observation, timestamp_s: float
    ) -> list[tuple[int, IdentityRecord, float, float]]:
        """Candidats satisfaisant la contrainte de déplacement maximale.

        Un candidat est retenu si

            distance <= v_max_ratio × hauteur_bbox × min(Δt, grâce)
                        + spatial_margin_ratio × hauteur_bbox

        où ``Δt`` est le temps écoulé **depuis la dernière observation** de
        l'identité. Les identités purgées restent interrogeables pendant une
        fenêtre de rétention supplémentaire (spec 3.3, « récemment purgés »),
        mesurée depuis l'instant de purge et non depuis la dernière observation.
        """
        candidates: list[tuple[int, IdentityRecord, float, float]] = []
        for person_id, record in self.records.items():
            if record.live or person_id in self._claimed:
                continue
            if not self.within_gallery_window(record, timestamp_s):
                continue
            reference_height = record.bbox_height
            if reference_height <= 0:
                continue
            elapsed = timestamp_s - record.last_seen_s
            allowed = self.long_term.v_max_ratio * reference_height * min(
                elapsed, self.grace_period_seconds
            ) + self.long_term.spatial_margin_ratio * reference_height
            distance = _distance(observation.anchor, record.anchor)
            if distance <= allowed:
                candidates.append((person_id, record, allowed, distance))
        return candidates

    def within_gallery_window(self, record: IdentityRecord, timestamp_s: float) -> bool:
        """L'identité est-elle encore interrogeable par la galerie ?

        Deux fenêtres distinctes :

        - avant purge (piste perdue mais encore dans la grâce), la fenêtre est
          la **grâce FSM augmentée de la rétention**. La galerie n'est jamais
          plus stricte que la machine à états : sinon elle libérerait une
          identité avant que le FSM ait pu la purger, ce qui supprimerait
          l'événement ``PURGE`` et le passage en occupation incertaine ;
        - après purge, la rétention ouvre une fenêtre supplémentaire
          (« récemment purgés » de la spec 3.3) mesurée depuis la purge.
        """
        if record.purged_at_s is not None:
            return (timestamp_s - record.purged_at_s) <= self.purge_retention_seconds
        return (timestamp_s - record.last_seen_s) <= (
            self.grace_period_seconds + self.purge_retention_seconds
        )

    # -- Cycle de vie ------------------------------------------------------
    def _touch(
        self,
        record: IdentityRecord,
        observation: Observation,
        feature: np.ndarray | None,
        timestamp_s: float,
        frame_index: int = 0,
        frame: np.ndarray | None = None,
    ) -> None:
        record.live = True
        record.technical_track_id = int(observation.technical_track_id)
        record.anchor = observation.anchor
        record.bbox_height = observation.bbox_height
        record.bbox = np.asarray(observation.bbox, dtype=float)
        record.last_seen_s = timestamp_s
        record.observations += 1
        # Une identité observée redevient vivante : elle cesse d'être « purgée »
        # et retrouve la fenêtre de grâce normale. L'événement PURGE déjà
        # journalisé n'est pas effacé : la réassociation est tracée par
        # REID_MATCH et par la transition d'état depuis ABSENTE.
        record.purged_at_s = None
        record.purge_reason = None
        rejection, threshold, measured = self._gallery_rejection(observation, feature, frame)
        if rejection is None:
            record.feature = _blend(record.feature, feature, self.feature_momentum)
            record.last_rejection = None
            return
        # Apparence refusée : l'apparence connue est conservée telle quelle.
        self._report_descriptor_rejection(
            record, observation, rejection, threshold, measured, timestamp_s, frame_index
        )

    def _report_descriptor_rejection(
        self,
        record: IdentityRecord,
        observation: Observation,
        reason: str,
        threshold: float | None,
        measured: float | None,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        """Journalise un refus d'apparence, une fois par changement de motif."""
        if record.last_rejection == reason:
            return
        record.last_rejection = reason
        self.descriptor_rejections += 1
        self._emit(
            "REID_DESCRIPTOR_REJECTED", timestamp_s, frame_index,
            person_id=record.person_id,
            technical_track_id=int(observation.technical_track_id),
            reason=reason,
            threshold=None if threshold is None else round(threshold, 4),
            confidence=round(float(observation.confidence), 4),
            bbox_height=round(float(observation.bbox_height), 2),
            bbox_width=round(
                float(observation.bbox[2] - observation.bbox[0]) if len(observation.bbox) >= 4
                else 0.0,
                2,
            ),
            **({} if measured is None else {"sharpness": round(float(measured), 4)}),
        )

    def mark_purged(
        self,
        person_id: int,
        timestamp_s: float,
        frame_index: int,
        reason: str,
        last_state: str,
        occupancy_impact: str,
        grace_period_frames: int | None = None,
    ) -> None:
        """Journalise la purge d'un ``person_id`` (spec 3.4) et ouvre la galerie.

        L'identité reste ré-associable pendant une seconde fenêtre de grâce
        (``purge_retention_seconds``) ; au-delà, :meth:`release_expired` libère la
        mémoire. La purge du ``person_id`` elle-même est **toujours** explicite :
        raison, dernière position et dernier statut sont journalisés.

        ``purge_kind="technical"`` et ``is_exit=False`` : cette méthode ne peut
        pas produire une sortie. Seul ``COMPTE_SORTIE`` diminue l'occupation
        (spec 6 du prompt) — la libération de mémoire d'une piste perdue n'a
        aucune signification métier de sortie.
        """
        record = self.records.get(person_id)
        if record is None:
            return
        self._emit(
            "PURGE", timestamp_s, frame_index,
            person_id=person_id,
            reason=reason,
            last_position=list(record.anchor),
            last_state=last_state,
            occupancy_impact=occupancy_impact,
            grace_period_frames=grace_period_frames,
            lifetime_s=round(record.age_s, 4),
            purge_kind="technical",
            is_exit=False,
        )
        record.purged_at_s = timestamp_s
        record.purge_reason = reason
        record.live = False

    def release_expired(self, timestamp_s: float) -> list[int]:
        """Libère les identités dont la fenêtre de galerie est terminée."""
        released: list[int] = []
        for person_id, record in list(self.records.items()):
            if record.live:
                continue
            if not self.within_gallery_window(record, timestamp_s):
                released.append(person_id)
                del self.records[person_id]
                for technical_id in list(self.technical_to_person):
                    if self.technical_to_person[technical_id] == person_id:
                        del self.technical_to_person[technical_id]
                self.provisional_ids.discard(person_id)
        return released

    def forget(self, person_id: int) -> None:
        """Retire immédiatement une identité de la galerie."""
        self.records.pop(person_id, None)
        for technical_id in list(self.technical_to_person):
            if self.technical_to_person[technical_id] == person_id:
                del self.technical_to_person[technical_id]
        self.provisional_ids.discard(person_id)

    # -- Émission ----------------------------------------------------------
    def _inconsistent(self, kind: str, timestamp_s: float, frame_index: int, **fields: Any) -> None:
        self._emit("INCONSISTENT_STATE", timestamp_s, frame_index, kind=kind, **fields)

    def _emit(self, event_type: str, timestamp_s: float, frame_index: int, **fields: Any) -> None:
        if self.event_sink is None:
            return
        self.event_sink.emit(event_type, timestamp_s, frame_index, **fields)


def _blend(previous: np.ndarray | None, current: np.ndarray, momentum: float) -> np.ndarray:
    """Met à jour un descripteur en moyenne glissante, puis le renormalise."""
    if previous is None or previous.shape != current.shape:
        blended = current.astype(np.float32)
    else:
        blended = momentum * previous + (1.0 - momentum) * current
    norm = float(np.linalg.norm(blended))
    return blended if norm < 1e-8 else blended / norm


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
