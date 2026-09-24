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

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

#: Affectation optimale (coût = 1 - similarité) pour la résolution globale d'un
#: groupe de trois pistes ou plus. Sans ``scipy``, repli sur la correction par
#: paires (comportement historique).
try:  # pragma: no cover - dépend de l'environnement d'exécution
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:  # pragma: no cover
    _linear_sum_assignment = None  # type: ignore[assignment]

from config import (
    AppearanceContinuityConfig,
    ExternalReidConfig,
    LongTermReidConfig,
    SwapCorrectionConfig,
)
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
    """Descripteur léger couleur (HSV) **par bandes horizontales**, sans torch.

    C'est l'extracteur par défaut : rapide, déterministe, sans téléchargement de
    poids. Il reprend la signature de ``IDManager.appearance`` (audit §2.1 a).

    Le descripteur historique calculait un histogramme teinte × saturation
    (24 × 8) sur le **crop complet**. En salle de cours, il encodait donc
    principalement la couleur des chaises, des tables et du mur derrière : deux
    personnes différentes de la même rangée donnaient couramment une similarité
    cosinus > 0,9 alors que ``similarity_threshold`` vaut 0,40 — l'écart
    inter-personnes était inférieur au bruit du descripteur, ce qui est la cause
    structurelle des réassociations croisées.

    Trois changements, dans cet ordre :

    1. le crop est **érodé** horizontalement (``horizontal_crop_ratio`` de chaque
       côté) : c'est le bord de la boîte qui avale le fond et les voisins ;
    2. le crop rogné est découpé en ``bands`` bandes horizontales (tête / torse /
       bas), chacune décrite par son propre histogramme H×S, **normalisé
       séparément** ;
    3. chaque bande est pondérée (``band_weights``), la bande basse plus
       faiblement puisqu'elle est la plus souvent occultée par une table, puis
       les bandes sont concaténées et l'ensemble renormalisé (contrat
       « descripteur L2-normalisé » du protocole : le poids relatif des bandes
       est conservé).

    La dimension du descripteur devient ``bands × bins_hue × bins_saturation``.
    ``_blend`` gère déjà un changement de dimension (il repart du descripteur
    courant) et aucun descripteur n'est persisté entre deux sessions.
    """

    name = "histogram_hsv_bands"

    def __init__(
        self,
        bins_hue: int = 24,
        bins_saturation: int = 8,
        min_width: int = 8,
        min_height: int = 16,
        horizontal_crop_ratio: float = 0.15,
        bands: int = 3,
        band_weights: Sequence[float] | None = None,
    ) -> None:
        self.bins_hue = int(bins_hue)
        self.bins_saturation = int(bins_saturation)
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.horizontal_crop_ratio = float(horizontal_crop_ratio)
        self.bands = max(1, int(bands))
        weights = tuple(float(weight) for weight in (band_weights or ()))
        if len(weights) < self.bands:
            # Pondération incomplète : complétée par le poids neutre, plutôt que
            # de laisser une bande silencieusement sans poids.
            weights = weights + (1.0,) * (self.bands - len(weights))
        self.band_weights = weights[: self.bands]

    @classmethod
    def from_config(cls, descriptor) -> "HistogramAppearanceExtractor":
        """Construit l'extracteur depuis ``reid.long_term.descriptor``."""
        return cls(
            bins_hue=descriptor.bins_hue,
            bins_saturation=descriptor.bins_saturation,
            horizontal_crop_ratio=descriptor.horizontal_crop_ratio,
            bands=descriptor.bands,
            band_weights=descriptor.band_weights,
        )

    def describe(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < self.min_width or y2 - y1 < self.min_height:
            return None

        margin = int(round((x2 - x1) * self.horizontal_crop_ratio))
        if margin > 0:
            x1, x2 = x1 + margin, x2 - margin
        if x2 - x1 < self.min_width or y2 - y1 < self.min_height:
            return None

        crop = frame[y1:y2, x1:x2]
        band_height = crop.shape[0] // self.bands
        if band_height < 1:
            return None

        parts: list[np.ndarray] = []
        for index in range(self.bands):
            top = index * band_height
            # La dernière bande absorbe les lignes restantes : aucune ligne du
            # crop rogné n'est perdue par un arrondi de division.
            bottom = crop.shape[0] if index == self.bands - 1 else top + band_height
            band = crop[top:bottom]
            if band.size == 0:
                return None
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist(
                [hsv], [0, 1], None,
                [self.bins_hue, self.bins_saturation], [0, 180, 0, 256],
            )
            flat = histogram.flatten().astype(np.float32)
            norm = float(np.linalg.norm(flat))
            if norm < 1e-8:
                return None
            parts.append(flat / norm * float(self.band_weights[index]))

        concatenated = np.concatenate(parts)
        norm = float(np.linalg.norm(concatenated))
        return None if norm < 1e-8 else concatenated / norm


class DeepAppearanceUnavailable(RuntimeError):
    """Le descripteur profond ne peut pas être chargé (modèle absent, empreinte
    différente, moteur indisponible). Déclenche le repli sur l'histogramme."""


#: Sessions ONNX ouvertes, par chemin de modèle : l'ouverture (plusieurs
#: secondes) n'est payée qu'une fois par processus, et peut l'être pendant la
#: calibration web (préchargement). L'inférence d'une session est déterministe.
_ONNX_SESSIONS: dict[str, Any] = {}


class DeepAppearanceExtractor:
    """Descripteur d'apparence profond (lot 3.a).

    Deux moteurs (``reid.external_reid.backend``) :

    - ``onnxruntime`` : OSNet exporté en ONNX, lu depuis ``model_path`` (chemin
      relatif à la racine du dépôt). Crop **corps entier**, redimensionné à
      ``image_size`` (h, w), normalisation ImageNet, vecteur L2-normalisé ;
    - ``torchreid`` : ancien chemin, conservé tel quel (poids **ImageNet** de
      torchreid, repli MobileNetV3) — il n'utilise pas ``model_path``.

    Le chargement est explicite (:meth:`load`) : un modèle absent, illisible ou
    d'empreinte différente lève :class:`DeepAppearanceUnavailable`, que
    :class:`IdentityManager` transforme en repli annoncé.
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
        self._session: Any = None
        self._input_name: str | None = None

    def load(self) -> None:
        """Charge le moteur ; lève :class:`DeepAppearanceUnavailable` en cas d'échec."""
        if self.config.backend == "onnxruntime":
            self._load_onnx()
        else:
            self._ensure_model()

    def _load_onnx(self) -> None:
        if self._session is not None:
            return
        import hashlib

        from config import REPO_ROOT

        path = Path(self.config.model_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise DeepAppearanceUnavailable(f"modèle ReID absent : {path}")
        expected = self.config.model_expected_sha256
        if expected:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest.lower() != str(expected).lower():
                raise DeepAppearanceUnavailable(
                    f"empreinte du modèle ReID différente : {digest} (attendue {expected})"
                )
        cached = _ONNX_SESSIONS.get(str(path))
        if cached is not None:
            # Session déjà ouverte (préchargement pendant la calibration web) :
            # même fichier, même empreinte vérifiée ci-dessus, même moteur.
            self._session = cached
        else:
            try:
                import onnxruntime as ort  # import local : dépendance du seul chemin profond
            except ImportError as error:  # pragma: no cover - dépend de l'environnement
                raise DeepAppearanceUnavailable(f"onnxruntime indisponible : {error}") from error
            try:
                self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            except Exception as error:  # modèle illisible
                raise DeepAppearanceUnavailable(f"modèle ReID illisible : {error}") from error
            _ONNX_SESSIONS[str(path)] = self._session
        self._input_name = self._session.get_inputs()[0].name
        self.name = f"onnx_{path.stem}"

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
        if self.config.backend == "onnxruntime":
            self._load_onnx()
            out_h, out_w = self.config.image_size
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)
            tensor = (crop.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)
            vector = self._session.run(None, {self._input_name: tensor})[0][0].astype(np.float32)
        else:
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


#: Normalisation ImageNet, celle de l'entraînement d'OSNet (torchreid).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    possible_multi_person: bool = False
    head_point: tuple[float, float] | None = None
    head_confidence: float | None = None


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
    head_point: tuple[float, float] | None = None
    head_confidence: float | None = None


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
    #: Frame jusqu'à laquelle la mise à jour du descripteur d'apparence est
    #: **gelée** sur cette identité. Armé par ``_check_appearance_continuity``
    #: quand une discontinuité d'apparence est détectée sur la piste technique
    #: qui porte cette identité ; honoré par ``_touch``. Sans ce gel, la
    #: référence dérive vers l'autre personne d'un swap non détecté et la
    #: correction croisée devient aveugle exactement quand elle servirait.
    appearance_suspect_until_frame: int | None = None
    #: Dernière raison de rejet d'apparence (`REID_DESCRIPTOR_REJECTED`). Sert à
    #: ne journaliser qu'un changement de situation, jamais une ligne par frame.
    last_rejection: str | None = None
    #: Instant (secondes de scène) de la dernière écriture du descripteur de
    #: galerie. En mode ``on_demand`` (lot 3.a), le descripteur d'une identité
    #: suivie n'est recalculé qu'au bout de ``gallery_refresh_seconds``.
    feature_updated_s: float | None = None

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
        appearance_continuity: AppearanceContinuityConfig | None = None,
        swap_correction: SwapCorrectionConfig | None = None,
        grace_period_seconds: float = 5.0,
        appearance: AppearanceExtractor | None = None,
        event_sink: EventSink | None = None,
        feature_momentum: float | None = None,
    ) -> None:
        self.long_term = long_term or LongTermReidConfig()
        self.external_reid = external_reid or ExternalReidConfig()
        #: Garde-fou « inversement d'identifiant » (swap interne au tracker) :
        #: diagnostic seul, aucune réaffectation automatique.
        self.appearance_continuity = (
            appearance_continuity or AppearanceContinuityConfig()
        )
        #: Correction active des inversions d'identité (swap)
        self.swap_correction = swap_correction or SwapCorrectionConfig()
        self.grace_period_seconds = float(grace_period_seconds)
        #: Durée de rétention d'une identité **purgée** dans la galerie, pour
        #: permettre une réassociation « récemment purgée » (spec 3.3), puis
        #: libération silencieuse de la mémoire : c'est une mémoire cache, pas
        #: une identité comptée — la purge du ``person_id`` a déjà été
        #: journalisée au moment de son passage à ``ABSENTE`` (spec 3.4).
        #:
        #: Par défaut, elle est **alignée** sur la grâce d'occupation : la
        #: mémoire d'apparence n'est jamais libérée avant que la personne n'ait
        #: eu une chance réaliste de réapparaître (fenêtre totale = grâce +
        #: rétention). ``reid.long_term.gallery_retention_seconds`` permet de
        #: découpler explicitement les deux durées, au prix d'une justification
        #: dans la configuration.
        configured_retention = self.long_term.gallery_retention_seconds
        self.purge_retention_seconds = (
            self.grace_period_seconds
            if configured_retention is None
            else float(configured_retention)
        )
        self.event_sink = event_sink
        #: Raison du repli sur l'histogramme quand le descripteur profond est
        #: demandé mais indisponible (lot 3.a) ; ``None`` sinon.
        self.appearance_fallback_reason: str | None = None
        self.appearance: AppearanceExtractor = appearance or self._build_appearance()
        #: Politique « à la demande » (lot 3.a) : active seulement si le
        #: descripteur profond est réellement en service. Un extracteur injecté
        #: (tests, ablations) garde le calcul à chaque frame.
        self.on_demand_descriptors = bool(
            appearance is None
            and self.external_reid.enabled
            and self.appearance_fallback_reason is None
            and self.external_reid.descriptor_policy == "on_demand"
        )
        #: Coût du descripteur (publié dans le résumé de session).
        self.descriptors_computed = 0
        self.descriptor_seconds = 0.0
        #: Dont les descripteurs calculés pour la correction des inversions,
        #: déclenchés par la proximité de deux pistes (mode ``on_demand``).
        self.proximity_descriptors_computed = 0
        #: Descripteurs de la frame courante réservés à la correction des
        #: inversions : jamais écrits en galerie (découpes prises pendant un
        #: croisement, donc potentiellement mêlées).
        self._swap_features: dict[int, np.ndarray] = {}
        # Le momentum vit dans la configuration (``reid.long_term.feature_momentum``)
        # et non plus en dur ici : il était impossible de le régler sans toucher au
        # code, alors qu'il gouverne la vitesse de dérive du descripteur de galerie
        # après un swap. Le paramètre du constructeur reste une surcharge explicite
        # (tests, ablations) qui prime sur la configuration.
        self.feature_momentum = float(
            self.long_term.feature_momentum if feature_momentum is None else feature_momentum
        )
        #: Durée (frames) pendant laquelle le descripteur de galerie est gelé après
        #: une discontinuité d'apparence sur la piste.
        self.freeze_gallery_frames = int(self.appearance_continuity.freeze_gallery_frames)

        self.records: dict[int, IdentityRecord] = {}
        self.technical_to_person: dict[int, int] = {}
        self._next_person_id = 1
        self._claimed: set[int] = set()
        #: Identités provisoires issues d'une décision ambiguë (spec 3.3 étape 5).
        self.provisional_ids: set[int] = set()
        #: Diagnostics (compteurs publiés dans le résumé de session).
        self.descriptor_rejections = 0
        self.technical_id_changes = 0
        self.appearance_discontinuities = 0
        self.swap_corrections_applied = 0
        #: Dernier descripteur observé **par identifiant technique** — et non par
        #: ``person_id`` : c'est la seule vue qui permette de détecter un swap
        #: interne au tracker, où la personne suit un identifiant technique qui
        #: garde sa cible « logique » et change pourtant d'apparence.
        #: ``(descripteur, frame_index, timestamp_s)``.
        self._track_appearance: dict[int, tuple[np.ndarray, int, float]] = {}
        #: Pistes actuellement signalées : évite une ligne par frame tant que la
        #: discontinuité n'est pas refermée par un retour à la continuité.
        self._discontinuity_active: set[int] = set()
        self._last_continuity_emit_s: dict[int, float] = {}
        #: Échantillon borné des similarités d'apparence **mesurées entre deux
        #: frames consécutives d'une même piste continue**. C'est ce qui permet de
        #: régler ``similarity_threshold`` sur des valeurs observées plutôt que
        #: sur une intuition : un seuil calé sur le p1 des mesures ne déclenche
        #: aucun faux diagnostic, un seuil plus haut en déclenche.
        self.continuity_similarities: deque[float] = deque(maxlen=4000)

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
        start = time.perf_counter()
        feature = self.appearance.describe(frame, bbox)
        self.descriptor_seconds += time.perf_counter() - start
        self.descriptors_computed += 1
        return feature

    def _build_appearance(self) -> AppearanceExtractor:
        """Descripteur profond si demandé et disponible, sinon histogramme.

        Le repli n'est jamais silencieux : avertissement sur la journalisation
        Python et sur stderr, événement ``CONFIG_WARNING``, et raison conservée
        pour le résumé. ``fallback_to_histogram: false`` fait échouer le
        démarrage au lieu de replier.
        """
        histogram = HistogramAppearanceExtractor.from_config(self.long_term.descriptor)
        if not self.external_reid.enabled:
            return histogram
        deep = DeepAppearanceExtractor(self.external_reid)
        try:
            deep.load()
        except DeepAppearanceUnavailable as error:
            if not self.external_reid.fallback_to_histogram:
                raise
            self.appearance_fallback_reason = str(error)
            logging.getLogger(__name__).warning(
                "ReID profond indisponible (%s) : repli sur l'histogramme", error
            )
            print(f"[REID] AVERTISSEMENT : {error} -- repli sur l'histogramme", file=sys.stderr)
            self._emit(
                "CONFIG_WARNING", 0.0, 0,
                code="reid_deep_descriptor_unavailable",
                details=f"{error} ; repli sur l'histogramme",
                source="reid",
            )
            return histogram
        return deep

    def _insertable(self, observation: Observation) -> bool:
        """Critères de galerie vérifiables **sans** descripteur (lot 3.a).

        Mêmes seuils que :meth:`_gallery_rejection`, avant l'étape qui exige le
        descripteur : en mode ``on_demand``, on ne paie pas un réseau pour un
        échantillon qui serait refusé de toute façon.
        """
        limits = self.long_term
        bbox = np.asarray(observation.bbox, dtype=float)
        return (
            observation.confidence >= limits.gallery_min_confidence
            and float(bbox[3] - bbox[1]) >= limits.gallery_min_crop_height_px
            and float(bbox[2] - bbox[0]) >= limits.gallery_min_crop_width_px
            and not getattr(observation, "possible_multi_person", False)
            and observation.anchor_reliable
        )

    def _gallery_refresh_due(
        self, record: IdentityRecord, observation: Observation, timestamp_s: float, frame_index: int
    ) -> bool:
        """En mode ``on_demand`` : faut-il recalculer le descripteur de galerie ?"""
        if record.feature is not None and record.feature_updated_s is not None and (
            float(timestamp_s) - record.feature_updated_s
        ) < float(self.external_reid.gallery_refresh_seconds):
            return False
        if self._appearance_is_frozen(record, frame_index):
            return False
        return self._insertable(observation)

    def descriptor_summary(self) -> dict[str, Any]:
        """Descripteur en service et coût mesuré (résumé de session)."""
        count = int(self.descriptors_computed)
        return {
            "name": str(getattr(self.appearance, "name", type(self.appearance).__name__)),
            "policy": "on_demand" if self.on_demand_descriptors else "every_frame",
            "fallback_reason": self.appearance_fallback_reason,
            "computed": count,
            "computed_for_swap_correction": int(self.proximity_descriptors_computed),
            "total_ms": round(self.descriptor_seconds * 1000.0, 1),
            "mean_ms": round(self.descriptor_seconds * 1000.0 / count, 3) if count else None,
        }

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
        if getattr(observation, "possible_multi_person", False):
            return "possible_multi_person_crop", None, None
        if not observation.anchor_reliable:
            return "edge_truncated", None, None
        if feature is None:
            return "blurred_or_unreliable", None, None
        if limits.gallery_min_sharpness > 0 and frame is not None:
            measured = self.sharpness(frame, bbox)
            if measured is not None and measured < limits.gallery_min_sharpness:
                return "blurred_or_unreliable", float(limits.gallery_min_sharpness), measured
        return None, None, None

    def _extract_valid_feature(
        self,
        observation: Observation,
        frame: np.ndarray | None,
    ) -> np.ndarray | None:
        """Extrait le descripteur d'une observation s'il respecte les critères de qualité."""
        feature = observation.feature
        if feature is None:
            # Mode ``on_demand`` : descripteur calculé par le déclencheur de
            # proximité (:meth:`_describe_close_tracks`), réservé à la correction.
            feature = self._swap_features.get(int(observation.technical_track_id))
        if feature is None and frame is not None and not self.on_demand_descriptors:
            feature = self.describe(frame, observation.bbox)
            if feature is not None:
                observation.feature = feature
        if feature is None:
            return None
        rejection, _, _ = self._gallery_rejection(observation, feature, frame)
        if rejection is not None:
            return None
        return feature

    def _correct_cross_swaps(
        self,
        observations: Sequence[Observation],
        frame_index: int,
        timestamp_s: float,
        frame: np.ndarray | None = None,
    ) -> None:
        """Corrige activement les inversions d'identité (swap) entre paires de pistes connues.

        Pour chaque paire de pistes techniques présentes sur la frame courante et déjà
        mappées à deux identités distinctes et vivantes/non purgées, compare la similarité
        directe (A↔A, B↔B) contre la similarité croisée (A↔B, B↔A). Si le croisement
        l'emporte d'au moins la marge configurée, le mapping technical_to_person est échangé
        et un événement IDENTITY_SWAP_CORRECTED est émis.

        Les groupes de trois pistes ou plus dont les boîtes se touchent sont d'abord
        résolus globalement (:meth:`_correct_group_swaps`) : une permutation
        circulaire A→B, B→C, C→A n'est pas décomposable en décisions par paires.
        Les paires internes à un groupe ainsi résolu ne sont plus réexaminées ;
        toutes les autres suivent la comparaison par paires ci-dessous, inchangée.
        """
        if not self.swap_correction.enabled or len(observations) < 2:
            return

        corrected_tracks: set[int] = set()
        group_of = self._correct_group_swaps(
            observations, frame, frame_index, timestamp_s, corrected_tracks
        )

        for i in range(len(observations)):
            obs_a = observations[i]
            tech_a = int(obs_a.technical_track_id)
            if tech_a in corrected_tracks:
                continue

            person_a = self.technical_to_person.get(tech_a)
            if person_a is None or person_a not in self.records:
                continue
            rec_a = self.records[person_a]
            if rec_a.purged_at_s is not None or rec_a.feature is None:
                continue

            for j in range(i + 1, len(observations)):
                obs_b = observations[j]
                tech_b = int(obs_b.technical_track_id)
                if tech_b in corrected_tracks:
                    continue
                if tech_a in group_of and group_of.get(tech_b) == group_of[tech_a]:
                    continue  # paire déjà tranchée par la résolution de groupe

                person_b = self.technical_to_person.get(tech_b)
                if person_b is None or person_b not in self.records or person_b == person_a:
                    continue
                rec_b = self.records[person_b]
                if rec_b.purged_at_s is not None or rec_b.feature is None:
                    continue

                feat_a = self._extract_valid_feature(obs_a, frame)
                feat_b = self._extract_valid_feature(obs_b, frame)
                if feat_a is None or feat_b is None:
                    continue

                ref_a = rec_a.feature
                ref_b = rec_b.feature
                direct = self.cosine_similarity(feat_a, ref_a) + self.cosine_similarity(feat_b, ref_b)
                crossed = self.cosine_similarity(feat_a, ref_b) + self.cosine_similarity(feat_b, ref_a)

                if crossed > (direct + self.swap_correction.margin):
                    # Inversion confirmée avec marge nette
                    self.technical_to_person[tech_a] = person_b
                    self.technical_to_person[tech_b] = person_a
                    corrected_tracks.add(tech_a)
                    corrected_tracks.add(tech_b)
                    self.swap_corrections_applied += 1

                    self._emit(
                        "IDENTITY_SWAP_CORRECTED",
                        timestamp_s,
                        frame_index,
                        frame=int(frame_index),
                        technical_track_id_a=tech_a,
                        technical_track_id_b=tech_b,
                        person_id_a_before=person_a,
                        person_id_a_after=person_b,
                        person_id_b_before=person_b,
                        person_id_b_after=person_a,
                        similarity_direct=round(float(direct), 4),
                        similarity_crossed=round(float(crossed), 4),
                        margin_applied=round(float(self.swap_correction.margin), 4),
                        reason="cross_appearance_correction",
                    )
                    break

    def _correct_group_swaps(
        self,
        observations: Sequence[Observation],
        frame: np.ndarray | None,
        frame_index: int,
        timestamp_s: float,
        corrected_tracks: set[int],
    ) -> dict[int, int]:
        """Résolution globale des groupes de trois pistes ou plus (algorithme hongrois).

        Éligible : piste mappée à une identité non purgée dont l'empreinte de
        galerie existe, et dont le descripteur **courant** est déjà disponible.
        En mode ``on_demand`` aucun descripteur n'est calculé ici : seules les
        pistes décrites sur cette frame (rafraîchissement de galerie, déclencheur
        de proximité) participent, comme pour la comparaison par paires.

        Groupes : composantes connexes de boîtes qui se touchent
        (``external_reid.proximity_margin_ratio``, le critère du déclencheur de
        proximité). Matrice N×N (courant × référence), coût ``1 - similarité`` ;
        appliquée si elle déplace au moins deux pistes et si le gain total dépasse
        ``swap_correction.margin``. Sans ``scipy`` : rien n'est fait ici, la
        comparaison par paires traite tout.

        Returns:
            ``{technical_track_id: indice de groupe}`` des groupes résolus.
        """
        if _linear_sum_assignment is None or len(observations) < 3:
            return {}
        entries: list[tuple[Observation, int, np.ndarray, np.ndarray]] = []
        for obs in observations:
            person_id = self.technical_to_person.get(int(obs.technical_track_id))
            record = self.records.get(person_id) if person_id is not None else None
            if record is None or record.purged_at_s is not None or record.feature is None:
                continue
            feature = self._extract_valid_feature(obs, frame)
            if feature is None:
                continue
            entries.append((obs, person_id, feature, record.feature))
        if len(entries) < 3:
            return {}

        parent = list(range(len(entries)))

        def find(k: int) -> int:
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        margin_ratio = float(self.external_reid.proximity_margin_ratio)
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if self._boxes_touch(entries[i][0].bbox, entries[j][0].bbox, margin_ratio):
                    parent[find(j)] = find(i)
        groups: dict[int, list[int]] = {}
        for k in range(len(entries)):
            groups.setdefault(find(k), []).append(k)

        group_of: dict[int, int] = {}
        for index, members in enumerate(g for g in groups.values() if len(g) >= 3):
            group = [entries[k] for k in members]
            if len({person_id for _o, person_id, _f, _r in group}) < len(group):
                continue  # deux pistes sur une même identité : pas une permutation
            for obs, _p, _f, _r in group:
                group_of[int(obs.technical_track_id)] = index
            count = len(group)
            similarity = np.array(
                [[self.cosine_similarity(f, ref) for _o, _p, _f, ref in group] for _o, _p, f, _r in group],
                dtype=float,
            )
            rows, columns = _linear_sum_assignment(1.0 - similarity)
            assignment = {int(r): int(c) for r, c in zip(rows, columns)}
            changed = [r for r in range(count) if assignment[r] != r]
            current_total = float(np.trace(similarity))
            optimal_total = float(sum(similarity[r, assignment[r]] for r in range(count)))
            margin = float(self.swap_correction.margin)
            if len(changed) < 2 or (optimal_total - current_total) <= margin:
                continue
            group_person_ids = [group[assignment[r]][1] for r in range(count)]
            for r in changed:
                obs, person_before = group[r][0], group[r][1]
                tech = int(obs.technical_track_id)
                person_after = group[assignment[r]][1]
                self.technical_to_person[tech] = person_after
                corrected_tracks.add(tech)
                self.swap_corrections_applied += 1
                self._emit(
                    "IDENTITY_SWAP_CORRECTED",
                    timestamp_s,
                    frame_index,
                    frame=int(frame_index),
                    technical_track_id=tech,
                    person_id_before=person_before,
                    person_id_after=person_after,
                    group_size=count,
                    group_person_ids=group_person_ids,
                    similarity_direct=round(current_total, 4),
                    similarity_crossed=round(optimal_total, 4),
                    margin_applied=round(margin, 4),
                    reason="group_appearance_correction",
                )
        return group_of

    @staticmethod
    def _boxes_touch(a: Sequence[float], b: Sequence[float], margin_ratio: float) -> bool:
        """Deux boîtes se touchent-elles ou se recouvrent-elles ?

        Chaque boîte est élargie de ``margin_ratio`` × sa propre hauteur ; 0
        signifie contact ou recouvrement strict.
        """
        ma = margin_ratio * float(a[3] - a[1])
        mb = margin_ratio * float(b[3] - b[1])
        return not (
            float(a[2]) + ma < float(b[0]) - mb
            or float(b[2]) + mb < float(a[0]) - ma
            or float(a[3]) + ma < float(b[1]) - mb
            or float(b[3]) + mb < float(a[1]) - ma
        )

    def _describe_close_tracks(
        self, observations: Sequence[Observation], frame: np.ndarray | None
    ) -> None:
        """Mode ``on_demand`` : descripteur des pistes proches, pour la correction.

        Les inversions se produisent quand deux personnes se croisent : c'est là,
        et seulement là, que la correction a besoin du descripteur courant. Une
        piste n'est décrite que si sa boîte touche celle d'une autre piste et que
        **les deux** portent une identité vivante dotée d'un descripteur de
        galerie (sans quoi la correction ne peut rien comparer), et si
        l'échantillon passe les critères de galerie vérifiables sans descripteur
        (même règle que :meth:`_extract_valid_feature`). Coût borné par le nombre
        de pistes en contact.
        """
        if frame is None:
            return

        def eligible(obs: Observation) -> bool:
            person = self.technical_to_person.get(int(obs.technical_track_id))
            record = self.records.get(person) if person is not None else None
            return (
                record is not None
                and record.purged_at_s is None
                and record.feature is not None
                and self._insertable(obs)
            )

        candidates = [obs for obs in observations if eligible(obs)]
        margin = float(self.external_reid.proximity_margin_ratio)
        close: set[int] = set()
        for i, a in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                if self._boxes_touch(a.bbox, candidates[j].bbox, margin):
                    close.update((i, j))
        for index in sorted(close):
            obs = candidates[index]
            feature = self.describe(frame, obs.bbox)
            self.proximity_descriptors_computed += 1
            if feature is not None:
                self._swap_features[int(obs.technical_track_id)] = feature

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
        self._prune_track_appearance(frame_index)

        # Correction active des inversions d'identité (swap)
        self._swap_features = {}
        if self.swap_correction.enabled:
            if self.on_demand_descriptors and self.external_reid.describe_on_proximity:
                self._describe_close_tracks(observations, frame)
            self._correct_cross_swaps(observations, frame_index, timestamp_s, frame=frame)

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
            if not self.on_demand_descriptors:
                feature = self.describe(frame, observation.bbox)
            else:
                # Lot 3.a : descripteur calculé seulement pour une ré-identification
                # (piste inconnue) ou un rafraîchissement de galerie dû.
                locked = self.technical_to_person.get(technical_id)
                locked_record = self.records.get(locked) if locked is not None else None
                if locked_record is None or self._gallery_refresh_due(
                    locked_record, observation, timestamp_s, frame_index
                ):
                    feature = self.describe(frame, observation.bbox)

        # Indépendant du chemin d'affectation : la continuité d'apparence d'une
        # piste technique est vérifiée même quand elle est verrouillée sur une
        # identité (cas même le plus fréquent d'un swap, où rien ne « disparaît »).
        self._check_appearance_continuity(technical_id, feature, timestamp_s, frame_index)

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
                head_point=observation.head_point,
                head_confidence=observation.head_confidence,
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
        # Le seuil dépend de la durée d'absence du meilleur candidat : une longue
        # occultation change légèrement l'apparence, un seuil rigoureusement
        # constant rejetterait alors une réapparition légitime. Il n'est jamais
        # assoupli sans plancher explicite (voir ``effective_similarity_threshold``).
        absence_s = float(timestamp_s - self.records[best_person].last_seen_s)
        threshold = self.effective_similarity_threshold(absence_s)
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
            threshold_applied=threshold,
            absence_s=absence_s,
        )

    # -- Garde-fou de swap (continuité d'apparence par piste technique) ----
    def _prune_track_appearance(self, frame_index: int) -> None:
        """Oublie les descripteurs des pistes absentes au-delà de la tolérance.

        Un écart supérieur à ``max_gap_frames`` signifie que la piste a été
        perdue : la comparaison de continuité n'a plus de sens (le chemin de
        réassociation normale s'applique) et le descripteur ne sert plus. La
        mémoire reste donc bornée par la tolérance, pas par l'historique complet
        de la session.
        """
        if not self.appearance_continuity.enabled:
            return
        horizon = int(frame_index) - int(self.appearance_continuity.max_gap_frames)
        stale = [
            technical_id
            for technical_id, (_feature, seen_frame, _seen_s) in self._track_appearance.items()
            if seen_frame < horizon
        ]
        for technical_id in stale:
            del self._track_appearance[technical_id]
            self._discontinuity_active.discard(technical_id)

    def _check_appearance_continuity(
        self,
        technical_id: int,
        feature: np.ndarray | None,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        """Journalise un saut d'apparence sur une piste **jamais perdue**.

        Le swap est détecté ici uniquement si la piste technique était présente
        à une frame suffisamment proche (``max_gap_frames``) : une piste absente
        entre-temps a suivi le chemin de réassociation normal, où la marge de
        sécurité et le seuil de galerie s'appliquent déjà — ce n'est plus un
        swap interne au tracker.

        Aucune identité n'est réaffectée : l'événement sert à mesurer l'ampleur
        réelle du phénomène avant d'envisager une correction automatique plus
        risquée.
        """
        config = self.appearance_continuity
        if not config.enabled or feature is None:
            return
        current = np.asarray(feature, dtype=np.float32)
        previous = self._track_appearance.get(int(technical_id))
        self._track_appearance[int(technical_id)] = (current, int(frame_index), float(timestamp_s))
        if previous is None:
            return
        previous_feature, previous_frame, previous_s = previous
        gap = int(frame_index) - int(previous_frame)
        if gap <= 0 or gap > int(config.max_gap_frames):
            self._discontinuity_active.discard(int(technical_id))
            return
        if previous_feature.shape != current.shape:
            return
        similarity = self.cosine_similarity(previous_feature, current)
        self.continuity_similarities.append(similarity)
        if similarity >= float(config.similarity_threshold):
            # Continuité rétablie : la piste peut être signalée à nouveau plus tard.
            self._discontinuity_active.discard(int(technical_id))
            return
        # Apparence suspecte : la référence de galerie de cette identité est gelée
        # (ré-armé à chaque frame où la discontinuité persiste, y compris quand la
        # limitation de débit supprime l'événement — le gel suit l'état réel de la
        # piste, pas la journalisation).
        self._mark_appearance_suspect(int(technical_id), frame_index)
        if int(technical_id) in self._discontinuity_active:
            return
        last_emit = self._last_continuity_emit_s.get(int(technical_id))
        if last_emit is not None and (
            float(timestamp_s) - last_emit
        ) < float(config.min_interval_seconds):
            self._discontinuity_active.add(int(technical_id))
            return
        self._discontinuity_active.add(int(technical_id))
        self._last_continuity_emit_s[int(technical_id)] = float(timestamp_s)
        self.appearance_discontinuities += 1
        self._emit(
            "TRACK_ID_APPEARANCE_DISCONTINUITY",
            timestamp_s,
            frame_index,
            technical_track_id=int(technical_id),
            person_id_before=self.technical_to_person.get(int(technical_id)),
            similarity_to_previous_descriptor=round(similarity, 4),
            previous_descriptor_frame_index=int(previous_frame),
            descriptor_age_s=round(float(timestamp_s) - previous_s, 4),
            gap_frames=gap,
            threshold=round(float(config.similarity_threshold), 4),
            reason="possible_internal_swap",
        )

    def continuity_similarity_summary(self) -> dict[str, float | int | None]:
        """Distribution des similarités de continuité observées (calibration).

        Returns:
            ``samples``, ``min``, ``p1``, ``p5``, ``median`` et
            ``below_threshold`` (nombre d'échantillons sous le seuil configuré).
            ``samples = 0`` signifie qu'aucune comparaison n'a été possible
            (aucune piste n'a été vue deux frames de suite).
        """
        values = list(self.continuity_similarities)
        if not values:
            return {
                "samples": 0, "min": None, "p1": None, "p5": None,
                "median": None, "below_threshold": 0,
            }
        array = np.asarray(values, dtype=np.float64)
        threshold = float(self.appearance_continuity.similarity_threshold)
        return {
            "samples": int(array.size),
            "min": float(array.min()),
            "p1": float(np.percentile(array, 1)),
            "p5": float(np.percentile(array, 5)),
            "median": float(np.median(array)),
            "below_threshold": int((array < threshold).sum()),
        }

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
        threshold_applied: float | None = None,
        absence_s: float | None = None,
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
                # Seuil réellement appliqué et durée d'absence : sans eux, un
                # assouplissement progressif serait invisible dans le journal.
                threshold_applied=(
                    None if threshold_applied is None else round(threshold_applied, 4)
                ),
                absence_s=None if absence_s is None else round(absence_s, 3),
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
            head_point=observation.head_point,
            head_confidence=observation.head_confidence,
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
                head_point=observation.head_point,
                head_confidence=observation.head_confidence,
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
        if feature is None and self.on_demand_descriptors and frame is not None:
            # Lot 3.a : la création d'une identité est une insertion en galerie.
            feature = self.describe(frame, observation.bbox)
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
            feature_updated_s=float(timestamp_s) if rejection is None else None,
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
            head_point=observation.head_point,
            head_confidence=observation.head_confidence,
        )

    # -- Filtre spatio-temporel (spec 3.3 étape 1) -------------------------
    def effective_similarity_threshold(self, absence_s: float) -> float:
        """Seuil d'apparence applicable pour une durée d'absence donnée.

        Par défaut (``long_absence_similarity_floor`` non déclaré), le seuil est
        **constant** : ``similarity_threshold``. Aucun assouplissement n'est
        subi par défaut, précisément parce que baisser un seuil de similarité
        augmente le risque de réassocier deux personnes différentes.

        Si un plancher est déclaré, le seuil descend **linéairement** de
        ``similarity_threshold`` vers ce plancher au bout de
        ``long_absence_seconds`` d'absence, et ne descend **jamais** plus bas.
        Le plancher borne donc l'assouplissement, et la marge de sécurité
        vis-à-vis du second candidat reste appliquée inchangée.
        """
        base = float(self.long_term.similarity_threshold)
        floor = self.long_term.long_absence_similarity_floor
        if floor is None:
            return base
        floor = float(floor)
        if floor >= base:
            return base
        span = max(float(self.long_term.long_absence_seconds), 1e-9)
        progress = min(1.0, max(0.0, float(absence_s)) / span)
        return base + (floor - base) * progress

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
        if feature is None and self.on_demand_descriptors:
            # Descripteur volontairement non calculé sur cette frame (lot 3.a) :
            # ce n'est pas un rejet de qualité, l'apparence connue est conservée.
            return
        rejection, threshold, measured = self._gallery_rejection(observation, feature, frame)
        if rejection is None:
            record.last_rejection = None
            if self._appearance_is_frozen(record, frame_index):
                # Descripteur gelé le temps de la discontinuité : seul le
                # descripteur est concerné. La position, la hauteur, `live` et
                # `last_seen_s` ont déjà été mis à jour ci-dessus.
                return
            record.feature = _blend(record.feature, feature, self.feature_momentum)
            record.feature_updated_s = float(timestamp_s)
            return
        # Apparence refusée : l'apparence connue est conservée telle quelle.
        self._report_descriptor_rejection(
            record, observation, rejection, threshold, measured, timestamp_s, frame_index
        )

    def _appearance_is_frozen(self, record: IdentityRecord, frame_index: int) -> bool:
        """La mise à jour du descripteur de cette identité est-elle gelée ?"""
        deadline = record.appearance_suspect_until_frame
        return deadline is not None and int(frame_index) <= int(deadline)

    def _mark_appearance_suspect(self, technical_id: int, frame_index: int) -> None:
        """Gèle le descripteur de l'identité portée par ``technical_id``.

        Une discontinuité d'apparence sur une piste signifie que la référence
        d'apparence n'est plus fiable pour cette piste : continuer à la mélanger
        ferait basculer ``record.feature`` vers l'apparence de l'autre personne,
        ce qui rendrait la correction croisée (``_correct_cross_swaps``, qui
        compare l'apparence courante à ``record.feature``) aveugle au moment
        précis où le swap s'installe.
        """
        person_id = self.technical_to_person.get(int(technical_id))
        if person_id is None:
            return
        record = self.records.get(person_id)
        if record is None:
            return
        record.appearance_suspect_until_frame = int(frame_index) + self.freeze_gallery_frames

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
