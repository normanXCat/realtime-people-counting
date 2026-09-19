"""Chargement, validation et résolution de la configuration unique du pipeline.

Toute la logique métier lit ses paramètres ici : aucun seuil, aucune durée et
aucune marge géométrique n'est codé en dur ailleurs (spec section 2 et règle 0.3).

Règles appliquées à ce niveau :

- **Validation stricte au chargement** (spec 5.6) : toute valeur négative,
  incohérente ou hors bornes fait échouer le démarrage avec un message
  explicite, jamais un ajustement silencieux.
- **Ligne virtuelle obligatoirement manuelle** : la configuration ne contient
  plus ses deux points (``line.p1``/``line.p2`` sont refusés explicitement).
  Seul ``line.min_length_ratio`` est lu ici ; la validation « deux points
  distincts et longueur suffisante » (spec 5.7) s'applique à la sélection de
  l'opérateur, dans :mod:`calibration`.
- **Durées en secondes, jamais en frames** (règle 0.2) : les durées sont
  converties en frames au runtime par :meth:`TimingConfig.frame_budget` avec le
  FPS réellement mesuré.
- **Marges relatives** (règle 0.3) : les marges géométriques sont des fractions
  d'une échelle locale (hauteur médiane des bbox), jamais des pixels absolus.

Aucune dépendance à OpenCV ou Ultralytics : ce module est importable en test pur.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


# Racine du dépôt : src/config.py -> src -> racine.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"

#: Nom de fenêtre par défaut. Les noms de fenêtres OpenCV doivent rester en ASCII
#: (voir la validation de ``display.window_name``).
DEFAULT_WINDOW_NAME = "People counting"

ALLOWED_INSIDE_SIDES = ("positive", "negative")
ALLOWED_ON_LINE_POLICIES = ("indeterminate", "inside", "outside")
ALLOWED_DISAPPEARANCE_POLICIES = ("deferred_confirmation", "ambiguous_immediate")
ALLOWED_AMBIGUOUS_POLICIES = ("provisional_person", "defer")
ALLOWED_FPS_SOURCES = ("measured",)
#: Méthodes de compensation de mouvement caméra (GMC) acceptées par BoT-SORT.
#: ``none`` est la seule correcte pour une caméra **fixe** : les autres
#: réintroduiraient une estimation de mouvement sur une image immobile, donc une
#: source d'instabilité gratuite des boîtes et des identifiants techniques.
ALLOWED_GMC_METHODS = ("none", "sparseOptFlow", "orb", "ecc", "sof")


class ConfigError(ValueError):
    """Configuration invalide : le démarrage doit échouer, pas être corrigé."""


@dataclass(frozen=True)
class SourceConfig:
    default: str = "0"
    reconnect_attempts: int = 0
    #: Délai sans aucune détection au-delà duquel l'absence est signalée
    #: explicitement (spec 5.5), distinct d'une fin de flux et d'une erreur.
    no_detection_seconds: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    """Inférence YOLO.

    ``confidence`` est le **seuil global** de l'inférence : c'est lui qui décide
    quelles détections existent avant BoT-SORT. Il doit rester suffisamment bas
    pour laisser passer les détections faibles qu'une personne floue produit
    encore, sinon l'association de BoT-SORT (et donc la persistance de piste)
    perd une information qui existait. Le garde-fou contre les faux positifs
    n'est pas ce seuil : il est porté par les seuils distincts de
    :class:`TrackerConfig` (``new_track_thresh``) et par le comptage lui-même
    (confirmation, zone morte, ReID).

    La cohérence ``confidence <= TrackerConfig.track_low_thresh`` est vérifiée
    par :func:`coherence_warnings` : au-delà, la gamme
    ``[track_low_thresh, confidence[`` n'atteint jamais le tracker.
    """

    path: str = "models/yolo11n.pt"
    expected_sha256: str = ""
    confidence: float = 0.10
    nms_iou: float = 0.55
    image_size: int = 960
    classes: tuple[int, ...] = (0,)
    device: str = "cpu"


@dataclass(frozen=True)
class TrackerConfig:
    """BoT-SORT : persistance de piste et association à deux étages.

    Les trois seuils sont distincts du seuil YOLO (règle 0.3 appliquée au
    tracking) :

    - ``track_high_thresh`` : plancher des détections utilisées pour
      l'association principale ;
    - ``track_low_thresh`` : plancher des détections conservées pour le second
      étage d'association (celles qu'une personne floue produit encore) ;
    - ``new_track_thresh`` : plancher exigé pour créer une **nouvelle** piste,
      c'est-à-dire le garde-fou explicite contre les faux positifs du tracker.

    Ces valeurs doivent rester synchronisées avec le fichier YAML lu par
    Ultralytics (``config_path``) ; un test le vérifie, car une divergence
    serait silencieuse au runtime.

    Persistance de piste et ré-identification **native** du tracker :

    - ``track_buffer`` : nombre de frames qu'une piste perdue reste vivante
      avant suppression définitive côté tracker. Il doit couvrir la durée
      typique des occlusions mesurée sur les vidéos de test (médiane ≈ 2,5 s,
      p90 ≈ 5 s) : ``150`` frames valent ≈ 5 s à la cadence source de 30 ips,
      soit le p90 observé. Compromis documenté : un buffer plus long réduit les
      changements d'identifiant technique mais augmente la mémoire/le calcul et
      le risque de réassociation erronée en forte densité ;
    - ``with_reid`` : active le module d'apparence **de BoT-SORT**, qui tente une
      réassociation avant de créer un nouvel identifiant technique. Il complète
      (sans le remplacer) le ReID long terme de :mod:`identity_manager`, qui
      opère, lui, après expiration de la piste technique ;
    - ``match_thresh`` / ``proximity_thresh`` / ``appearance_thresh`` : seuils
      d'association (IoU, proximité, apparence) du tracker ;
    - ``gmc_method`` : compensation de mouvement caméra. ``none`` pour une
      caméra fixe (toute autre valeur estimerait un mouvement inexistant).
    """

    config_path: str = "src/configs/custom_botsort.yaml"
    persist: bool = True
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    #: Persistance de piste : durée d'occlusion couverte par le tracker.
    track_buffer: int = 150
    #: Ré-identification native BoT-SORT (association par apparence).
    with_reid: bool = True
    match_thresh: float = 0.9
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    #: Compensation de mouvement caméra : ``none`` (caméra fixe).
    gmc_method: str = "none"


@dataclass(frozen=True)
class LongTermReidConfig:
    """Galerie long terme et **qualité des descripteurs écrits en galerie**.

    Les quatre garde-fous ``gallery_*`` décrivent quand une apparence est
    exploitable. Ils existent parce qu'un descripteur calculé sur une détection
    faible, un crop minuscule, une boîte tronquée par un bord ou une image floue
    dégrade la galerie : réassocier une identité connue à partir d'une apparence
    non fiable est exactement ce qui produit les fausses identités et les faux
    ``NEW``. Une apparence rejetée n'écrase **jamais** une apparence connue ;
    le rejet est journalisé (``REID_DESCRIPTOR_REJECTED``), jamais silencieux.
    """

    enabled: bool = True
    similarity_threshold: float = 0.40
    safety_margin: float = 0.10
    v_max_ratio: float = 1.5
    spatial_margin_ratio: float = 0.3
    ambiguous_policy: str = "provisional_person"
    gallery_min_confidence: float = 0.70
    gallery_min_crop_height_px: int = 32
    gallery_min_crop_width_px: int = 16
    #: Variance minimale du Laplacien du crop (netteté). 0 = contrôle désactivé.
    gallery_min_sharpness: float = 0.0
    #: Durée de rétention d'une identité **purgée** dans la galerie d'apparence.
    #: ``None`` = alignée sur ``timing.grace_period_seconds`` (valeur par défaut) :
    #: la mémoire d'apparence n'est alors jamais libérée avant que la personne
    #: n'ait eu une chance réaliste de réapparaître (fenêtre totale = grâce +
    #: rétention). Une valeur explicite découple la mémoire du ReID de la grâce
    #: d'occupation, ce qui doit être justifié dans la configuration.
    gallery_retention_seconds: float | None = None
    #: Tolérance **progressive** du seuil d'apparence selon la durée d'absence
    #: (une longue occultation change légèrement l'apparence : angle, lumière).
    #: ``long_absence_similarity_floor`` = seuil plancher atteint au bout de
    #: ``long_absence_seconds`` d'absence. ``None`` (défaut) = pas d'assouplissement
    #: (le seuil reste ``similarity_threshold``) : le compromis est choisi
    #: explicitement, jamais subi, pour éviter les fausses réassociations.
    long_absence_seconds: float = 2.0
    long_absence_similarity_floor: float | None = None
    #: Momentum de la moyenne glissante qui met à jour le descripteur d'apparence
    #: de la galerie (``_blend`` dans :mod:`identity_manager`). Était codé en dur
    #: à 0,85 dans le constructeur : il vit désormais dans la configuration.
    #: Une valeur élevée garde une référence stable (peu de dérive vers un autre
    #: individu, mais lente adaptation à un vrai changement d'apparence) ; une
    #: valeur basse suit l'apparence courante et **dérive vite** vers la mauvaise
    #: personne après un swap non détecté. Bornes ``0,0 <= x < 1,0`` : 1,0
    #: figerait le descripteur pour toujours.
    feature_momentum: float = 0.85
    #: Paramètres du descripteur d'apparence léger (HSV). Voir
    #: :class:`DescriptorConfig`.
    descriptor: "DescriptorConfig" = field(default_factory=lambda: DescriptorConfig())


@dataclass(frozen=True)
class DescriptorConfig:
    """Paramètres de ``HistogramAppearanceExtractor`` (descripteur HSV par bandes).

    L'ancien descripteur calculait un histogramme teinte × saturation sur le
    **crop complet**. En salle de cours, il encodait donc principalement la
    couleur des chaises, des tables et du mur derrière, et accessoirement le haut
    du vêtement : deux personnes différentes de la même rangée donnaient
    couramment une similarité > 0,9 alors que ``similarity_threshold`` vaut 0,40.
    Le descripteur rogne désormais le crop, le découpe en bandes horizontales et
    normalise chaque bande séparément, ce qui rend l'écart inter-personnes
    exploitable.

    - ``horizontal_crop_ratio`` : fraction rognée de chaque côté en largeur
      (0,15 = 15 %), pour écarter le fond et les voisins absorbés par le crop.
    - ``bands`` : nombre de bandes horizontales (tête / torse / bas).
    - ``band_weights`` : poids relatif de chaque bande, de la plus haute à la
      plus basse. Une bande basse pondérée plus faiblement est moins sensible aux
      tables qui l'occultent.
    - ``bins_hue`` / ``bins_saturation`` : résolution de l'histogramme 2D par
      bande.
    """

    horizontal_crop_ratio: float = 0.15
    bands: int = 3
    band_weights: tuple[float, ...] = (1.0, 1.0, 0.5)
    bins_hue: int = 24
    bins_saturation: int = 8


@dataclass(frozen=True)
class AppearanceContinuityConfig:
    """Garde-fou « inversement d'identifiant » (swap) sous occlusion dense.

    Un **swap** n'est pas une perte de piste : deux identifiants techniques
    restent actifs en continu, mais s'échangent la personne qu'ils suivent. Le
    phénomène se produit **à l'intérieur du matching de BoT-SORT**, avant que
    :mod:`identity_manager` ne voie quoi que ce soit d'anormal : aucun
    identifiant ne disparaît, donc ni ``TECHNICAL_ID_CHANGED`` ni
    ``TRACK_ID_ABSENT`` ne se déclenche. C'est pour cette raison qu'il passait
    inaperçu : les tests vérifiaient la continuité des identifiants, pas leur
    exactitude sémantique.

    Ce garde-fou **diagnostique sans corriger** : il compare le descripteur
    d'apparence d'une piste qui n'a jamais été perdue d'une frame à l'autre et
    journalise ``TRACK_ID_APPEARANCE_DISCONTINUITY`` quand la similarité
    s'effondre. Réaffecter automatiquement les identités risquerait de
    sur-corriger un cas où ce n'était pas un swap ; la métrique est d'abord
    mesurée, la correction automatique restant hors périmètre.

    - ``similarity_threshold`` : similarité cosinus en dessous de laquelle le
      saut d'apparence de deux frames **consécutives** de la même piste est
      considéré comme anormal. Un seuil élevé augmente la sensibilité (et les
      faux positifs de diagnostic) ; un seuil bas ne détecte que les échanges
      francs.
    - ``max_gap_frames`` : écart maximal, en frames traitées, pour qu'une piste
      reste considérée comme **continue**. Au-delà, la piste a réellement été
      perdue : la réassociation normale (galerie, marge de sécurité) s'applique
      et ce n'est plus un swap. ``0`` exige deux frames strictement
      consécutives.
    - ``min_interval_seconds`` : limitation de débit par piste. La
      discontinuité n'est écrite qu'une fois tant que la continuité n'est pas
      rétablie, et au plus une fois par intervalle.
    """

    enabled: bool = True
    similarity_threshold: float = 0.50
    max_gap_frames: int = 2
    min_interval_seconds: float = 1.0
    #: Nombre de frames pendant lesquelles la mise à jour du descripteur de
    #: galerie est **gelée** après une discontinuité d'apparence sur la piste.
    #: Sans ce gel, ``_touch`` mélangeait à chaque frame le descripteur courant
    #: dans ``record.feature`` : après un swap non détecté, la référence dérivait
    #: vers l'autre personne (1 - 0,85^5 ≈ 56 % de l'apparence de B au bout de
    #: 5 frames), et ``_correct_cross_swaps`` — qui compare l'apparence courante à
    #: ``record.feature`` — devenait aveugle exactement quand le swap s'installait.
    #: La position, la hauteur et ``last_seen_s`` continuent d'être mis à jour :
    #: seul le descripteur est gelé. Défaut : équivalent à
    #: ``min_interval_seconds`` en frames à la cadence de traitement mesurée
    #: (~3,5 FPS), avec une borne minimale de 3 frames.
    freeze_gallery_frames: int = 3


@dataclass(frozen=True)
class SwapCorrectionConfig:
    """Correction active des inversions d'identité (swap) entre paires de pistes.

    Réintroduit le comportement de l'ancien ``[RE-ID-LOCK]`` de ``IDManager`` :
    pour chaque paire de pistes techniques connues, la similarité directe
    (``A↔A``, ``B↔B``) est comparée à la similarité croisée (``A↔B``, ``B↔A``).
    Si le croisement est nettement meilleur (marge ``margin``), le mapping
    ``technical_to_person`` est corrigé et un événement
    ``IDENTITY_SWAP_CORRECTED`` est émis — jamais une correction silencieuse.

    Ce mécanisme est **additionnel** au diagnostic existant
    (``TRACK_ID_APPEARANCE_DISCONTINUITY``) : le diagnostic continue de mesurer,
    la correction agit.

    - ``enabled`` : flag d'ablation (``false`` = comportement V2 actuel,
      diagnostic seul) ;
    - ``margin`` : écart minimal ``crossed - direct`` pour déclencher la
      correction. Valeur initiale identique à l'ancien ``[RE-ID-LOCK]``, à
      recalibrer sur l'ensemble de calibration.
    """

    enabled: bool = True
    #: PROVISOIRE — même valeur que l'ancien [RE-ID-LOCK], à recalibrer.
    margin: float = 0.12


@dataclass(frozen=True)
class ExternalReidConfig:
    enabled: bool = False
    model_path: str = "osnet_x0_25_msmt17.pt"
    weights: str = "osnet_x0_25"
    image_size: tuple[int, int] = (256, 128)


@dataclass(frozen=True)
class ReidConfig:
    short_term: str = "native_botsort"
    long_term: LongTermReidConfig = field(default_factory=LongTermReidConfig)
    external_reid: ExternalReidConfig = field(default_factory=ExternalReidConfig)
    appearance_continuity: AppearanceContinuityConfig = field(
        default_factory=AppearanceContinuityConfig
    )
    swap_correction: SwapCorrectionConfig = field(
        default_factory=SwapCorrectionConfig
    )


@dataclass(frozen=True)
class LineConfig:
    """Paramètres de la ligne virtuelle **hors géométrie**.

    Les deux points ne figurent volontairement pas ici : la ligne est
    **obligatoirement définie manuellement** par l'opérateur au démarrage
    (:mod:`calibration`). Aucune ligne par défaut, horizontale ou non, ne peut
    donc être utilisée pour le comptage. ``inside_side`` ne fixe que
    l'orientation **initiale** proposée dans la fenêtre de sélection ; la valeur
    effectivement validée par l'opérateur est celle appliquée au pipeline.
    """

    inside_side: str = "negative"
    on_line_policy: str = "indeterminate"
    min_length_ratio: float = 0.05


@dataclass(frozen=True)
class MultiPersonBoxConfig:
    """Détection d'une boîte contenant probablement deux personnes.

    Une seule bounding box regroupant deux silhouettes proches se reconnaît à
    une **croissance anormale de la largeur** par rapport à l'historique
    stabilisé de la même personne, à hauteur constante ou quasi constante (une
    personne seule ne s'élargit pas brutalement sans bouger en profondeur).

    - ``enabled`` : flag d'ablation ;
    - ``max_growth_ratio`` : ratio ``largeur_courante / largeur_stabilisée``
      au-delà duquel le signal est déclenché. Valeur initiale à calibrer sur
      des cas confirmés de fusion.
    - ``head_separation_ratio`` : second signal, **indépendant de l'historique".
      Le détecteur par largeur exige une largeur d'**avant** la fusion : deux
      personnes assises côte à côte dès la première frame ne produisent donc
      JAMAIS le signal, leur largeur fusionnée *étant* la référence. Le nombre
      de têtes détectées à l'intérieur de la boîte lève cet angle mort : deux
      jeux de points-clés de tête suffisamment confiants et séparés de plus de
      ``head_separation_ratio × largeur_boîte`` dans une seule boîte valent
      fusion, dès la première frame.
    """

    enabled: bool = True
    #: PROVISOIRE — à calibrer sur des cas confirmés de fusion.
    max_growth_ratio: float = 1.6
    #: PROVISOIRE — séparation relative entre deux têtes au-delà de laquelle une
    #: boîte unique est considérée comme contenant deux personnes. N'a d'effet
    #: que si ``presence.head_assist.enabled`` est vrai (les points-clés de tête
    #: proviennent du modèle pose) ; sinon seul le signal par largeur subsiste.
    head_separation_ratio: float = 0.4


@dataclass(frozen=True)
class GeometryConfig:
    """Géométrie relative à l'échelle locale, en fractions (règle 0.3).

    ``occlusion_ambiguity_ratio`` est le rayon — fraction de la hauteur médiane
    des bbox — dans lequel une apparition intérieure peut être confondue avec
    une personne déjà comptée dont l'observation est perdue. C'est le rayon qui
    interdit de confirmer un ``NEW`` (spec 4 du prompt) : ni un pixel absolu, ni
    un seuil codé en dur.
    """

    dead_zone_ratio: float = 0.15
    anchor_edge_margin_ratio: float = 0.02
    scale_window: int = 61
    occlusion_ambiguity_ratio: float = 0.5
    min_aspect_ratio_wh: float = 0.1
    max_aspect_ratio_wh: float = 2.5
    min_box_height_px: float = 10.0
    min_box_width_px: float = 10.0
    multi_person_box: MultiPersonBoxConfig = field(
        default_factory=MultiPersonBoxConfig
    )


@dataclass(frozen=True)
class AnchorConfig:
    """Stabilité de l'ancre pieds lors des chutes anormales de hauteur de box."""

    height_history_window_frames: int = 30
    max_height_drop_ratio: float = 0.35
    height_drop_confirm_frames: int = 3



@dataclass(frozen=True)
class TimingConfig:
    """Fenêtres temporelles du pipeline (toutes en **secondes**, règle 0.2).

    Le warm-up est terminé quand **les deux** conditions sont vraies :

    - ``warmup_min_frames`` frames ont été observées (borne dure, robuste aux
      FPS très élevés où 0,5 s représentent beaucoup d'images) ;
    - ``warmup_seconds`` secondes se sont écoulées sur l'horloge réellement
      mesurée (borne de durée, robuste aux FPS faibles où il faut du temps pour
      voir tout le monde).

    S'exiger d'une seule des deux bornes terminerait le warm-up trop tôt soit à
    faible FPS (durée écoulée mais 3 frames vues), soit à haut FPS (15 frames
    vues en 0,15 s).
    """

    fps_source: str = "measured"
    warmup_seconds: float = 0.5
    warmup_min_frames: int = 15
    confirmation_seconds: float = 0.5
    grace_period_seconds: float = 5.0
    fps_estimate_window: int = 30

    def frame_budget(self, fps: float) -> "FrameBudget":
        """Convertit les durées (secondes) en frames pour un FPS mesuré donné.

        Le FPS doit provenir de la mesure runtime (règle 0.2) ; une valeur
        non positive est refusée plutôt que remplacée par un défaut silencieux.
        """
        if fps <= 0:
            raise ConfigError(
                f"FPS mesuré invalide ({fps!r}) : les durées ne peuvent pas être "
                "converties en frames"
            )
        return FrameBudget(
            fps=float(fps),
            warmup_frames=max(1, round(self.warmup_seconds * fps)),
            warmup_min_frames=max(1, int(self.warmup_min_frames)),
            confirmation_frames=max(1, round(self.confirmation_seconds * fps)),
            grace_period_frames=max(1, round(self.grace_period_seconds * fps)),
        )


@dataclass(frozen=True)
class FrameBudget:
    """Durées de la configuration exprimées en frames pour un FPS mesuré.

    ``warmup_frames`` est la conversion de ``warmup_seconds`` ;
    ``warmup_min_frames`` est la borne dure en frames. La fin du warm-up exige
    les deux (voir :class:`TimingConfig`).
    """

    fps: float
    warmup_frames: int
    warmup_min_frames: int
    confirmation_frames: int
    grace_period_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Diagnostics de détection et d'association (spec 2 et 3 du prompt).

    Chaque situation ambiguë doit être journalisée, mais un journal qui écrit un
    événement par frame devient illisible et masque l'information utile
    (limitation de débit explicite, jamais silencieuse : le compteur de frames
    concernées reste publié dans le résumé de session).
    """

    enabled: bool = True
    #: Intervalle minimal entre deux événements de même nature (secondes). Le
    #: front montant d'une situation est toujours journalisé.
    min_interval_seconds: float = 1.0


@dataclass(frozen=True)
class OccupancyConfig:
    in_progress_disappearance_policy: str = "deferred_confirmation"
    snapshot_interval_seconds: float = 1.0
    in_progress_policy: tuple[str, ...] = ("deferred_confirmation",)


@dataclass(frozen=True)
class OutputConfig:
    session_root: str = "results"
    events_filename: str = "events.jsonl"
    video_filename: str = "annotated.mp4"
    write_mode: str = "append_incremental"
    flush_every_n_events: int = 1
    write_video: bool = False
    annotated_video_codec: str = "mp4v"


@dataclass(frozen=True)
class StabilizationConfig:
    use_bbox_locker: bool = False
    bbox_locker_min_height_ratio: float = 0.70
    use_anchor_stabilizer: bool = False
    anchor_stabilizer_tolerance_ratio: float = 0.15


@dataclass(frozen=True)
class DisplayConfig:
    enabled: bool = True
    window_name: str = DEFAULT_WINDOW_NAME


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    print_state_transitions: bool = False


ALLOWED_HEAD_KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
)


@dataclass(frozen=True)
class HeadAssistConfig:
    """Assistance par détection de la tête lors des occultations de pieds (additif)."""

    enabled: bool = False
    pose_model_path: str = "models/yolo11s-pose.pt"
    pose_model_expected_sha256: str | None = None
    min_keypoint_confidence: float = 0.5  # PROVISOIRE
    keypoints_used: tuple[str, ...] = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    )
    max_presence_extension_seconds: float = 10.0  # PROVISOIRE
    #: Écart maximal (en frames) entre la frame courante et celle où le point
    #: tête a été **observé** pour autoriser un maintien de présence. Défaut 1 :
    #: la frame courante ou la précédente. Les keypoints étant extraits de la
    #: même détection que la boîte, un point plus vieux n'a plus de lien avec
    #: l'image courante.
    max_head_staleness_frames: int = 1


@dataclass(frozen=True)
class PresenceConfig:
    """Configuration du suivi de présence et de ses assistances."""

    head_assist: HeadAssistConfig = field(default_factory=HeadAssistConfig)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration résolue et validée du pipeline."""

    schema_version: int
    source: SourceConfig
    model: ModelConfig
    tracker: TrackerConfig
    reid: ReidConfig
    line: LineConfig
    geometry: GeometryConfig
    timing: TimingConfig
    occupancy: OccupancyConfig
    output: OutputConfig
    stabilization: StabilizationConfig
    display: DisplayConfig
    logging: LoggingConfig
    diagnostics: DiagnosticsConfig
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    config_path: Path = DEFAULT_CONFIG_PATH

    # -- Accès pratiques ---------------------------------------------------
    def resolve_path(self, value: str | Path) -> Path:
        """Résout un chemin relatif par rapport à la racine du dépôt."""
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path)

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable, revalidable telle quelle.

        Ne contient que les sections du schéma : ``config_path`` est une
        métadonnée de provenance, ajoutée séparément lors de l'écriture du
        dossier de session, pour qu'un aller-retour ``to_dict`` /
        :func:`build_config` reste valide.
        """
        data = asdict(self)
        data.pop("config_path", None)
        if not self.presence.head_assist.enabled:
            data.pop("presence", None)
        return data


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def override_config(
    config: PipelineConfig, overrides: Mapping[str, Mapping[str, Any]]
) -> PipelineConfig:
    """Applique des surcharges (CLI, ablations) **en repassant par la validation**.

    Chemin unique de surcharge : un dictionnaire aplati est fusionné avec la
    configuration sérialisée puis revalidé par :func:`build_config`. Aucune
    surcharge ne peut donc produire une configuration invalide en silence
    (spec 5.6).
    """
    raw = copy.deepcopy(config.to_dict())
    for section, values in overrides.items():
        if not isinstance(values, Mapping):
            raise ConfigError(f"Surcharge invalide pour la section {section!r}")
        current = raw.get(section)
        if current is None:
            if section == "presence":
                raw["presence"] = asdict(PresenceConfig())
                current = raw["presence"]
            else:
                raise ConfigError(f"Surcharge invalide : section inconnue {section!r}")
        if not isinstance(current, Mapping):
            raise ConfigError(
                f"Surcharge invalide : la section {section!r} n'est pas un mapping"
            )
        # Fusion récursive d'un niveau : `reid: {long_term: {...}}` reste possible.
        for key, value in values.items():
            if isinstance(value, Mapping) and isinstance(current.get(key), Mapping):
                raw[section][key] = {**current[key], **value}
            else:
                raw[section][key] = value
    return build_config(raw, config_path=config.config_path)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Charge, valide et résout la configuration du pipeline.

    Args:
        path: chemin du YAML. Par défaut ``config/pipeline.yaml``.

    Raises:
        ConfigError: fichier absent, YAML illisible, clé inattendue ou valeur
            invalide. Le message énumère toutes les erreurs trouvées.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Fichier de configuration introuvable : {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - dépend de PyYAML
        raise ConfigError(f"YAML illisible ({config_path}) : {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"La configuration doit être un mapping YAML : {config_path}")

    return build_config(raw, config_path=config_path)


def build_config(raw: Mapping[str, Any], config_path: Path | None = None) -> PipelineConfig:
    """Construit et valide une configuration depuis un mapping déjà chargé."""
    problems: list[str] = []
    known = {
        "schema_version", "source", "model", "tracker", "reid", "line", "geometry",
        "timing", "occupancy", "output", "stabilization", "display", "logging",
        "diagnostics", "anchor", "presence",
    }
    for key in raw:
        if key not in known:
            problems.append(f"section inconnue : {key!r}")

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        problems.append(
            f"schema_version doit valoir 1 (reçu {schema_version!r}) : une version "
            "inconnue ne peut pas être interprétée sans risque"
        )

    line_raw = _section(raw, "line")
    # Les points ne sont plus lisibles depuis la configuration : ils proviennent
    # exclusivement des deux clics de l'opérateur. Les accepter en silence
    # laisserait croire qu'une ligne écrite dans le YAML est utilisée (règle 0.4).
    for legacy_key in ("p1", "p2"):
        if legacy_key in line_raw:
            problems.append(
                f"line.{legacy_key} n'est plus lu : la ligne virtuelle est "
                "définie manuellement au démarrage (deux clics puis « C »). "
                f"Retirer line.{legacy_key} de la configuration."
            )
    inside_side = str(line_raw.get("inside_side", "negative"))
    if inside_side not in ALLOWED_INSIDE_SIDES:
        problems.append(
            f"line.inside_side doit être l'un de {ALLOWED_INSIDE_SIDES} (reçu {inside_side!r})"
        )
    on_line_policy = str(line_raw.get("on_line_policy", "indeterminate"))
    if on_line_policy not in ALLOWED_ON_LINE_POLICIES:
        problems.append(
            f"line.on_line_policy doit être l'un de {ALLOWED_ON_LINE_POLICIES} "
            f"(reçu {on_line_policy!r})"
        )
    min_length_ratio = _positive(
        line_raw.get("min_length_ratio", 0.05), "line.min_length_ratio", problems
    )
    # Le refus d'une ligne dégénérée ou trop courte (spec 5.7) s'applique à la
    # sélection manuelle, au moment de la validation par « C » :
    # :func:`calibration.LineSelection.confirm` appelle ``validate_line``.

    model_raw = _section(raw, "model")
    confidence = _unit_interval(model_raw.get("confidence", 0.10), "model.confidence", problems)
    nms_iou = _unit_interval(model_raw.get("nms_iou", 0.55), "model.nms_iou", problems)
    image_size = _positive_int(model_raw.get("image_size", 960), "model.image_size", problems)
    if image_size is not None and image_size % 32 != 0:
        problems.append(
            f"model.image_size doit être un multiple de 32 (reçu {image_size})"
        )
    classes = tuple(int(c) for c in model_raw.get("classes", (0,)) or ())
    device = str(model_raw.get("device", "cpu"))

    geometry_raw = _section(raw, "geometry")
    dead_zone_ratio = _positive(
        geometry_raw.get("dead_zone_ratio", 0.15), "geometry.dead_zone_ratio", problems
    )
    edge_margin_ratio = _non_negative(
        geometry_raw.get("anchor_edge_margin_ratio", 0.02),
        "geometry.anchor_edge_margin_ratio",
        problems,
    )
    if edge_margin_ratio is not None and edge_margin_ratio >= 0.5:
        problems.append(
            "geometry.anchor_edge_margin_ratio doit rester < 0.5 (reçu "
            f"{edge_margin_ratio}) : au-delà, toute boîte serait non fiable"
        )
    scale_window = _positive_int(
        geometry_raw.get("scale_window", 61), "geometry.scale_window", problems
    )
    occlusion_ambiguity_ratio = _non_negative(
        geometry_raw.get("occlusion_ambiguity_ratio", 0.5),
        "geometry.occlusion_ambiguity_ratio",
        problems,
    )
    min_aspect_ratio_wh = _positive(
        geometry_raw.get("min_aspect_ratio_wh", 0.1),
        "geometry.min_aspect_ratio_wh",
        problems,
    )
    max_aspect_ratio_wh = _positive(
        geometry_raw.get("max_aspect_ratio_wh", 2.5),
        "geometry.max_aspect_ratio_wh",
        problems,
    )
    min_box_height_px = _positive(
        geometry_raw.get("min_box_height_px", 10.0),
        "geometry.min_box_height_px",
        problems,
    )
    min_box_width_px = _positive(
        geometry_raw.get("min_box_width_px", 10.0),
        "geometry.min_box_width_px",
        problems,
    )
    if (
        min_aspect_ratio_wh is not None
        and max_aspect_ratio_wh is not None
        and min_aspect_ratio_wh >= max_aspect_ratio_wh
    ):
        problems.append(
            f"geometry.min_aspect_ratio_wh ({min_aspect_ratio_wh}) doit être < "
            f"geometry.max_aspect_ratio_wh ({max_aspect_ratio_wh})"
        )

    multi_person_raw = _section(geometry_raw, "multi_person_box")
    max_growth_ratio = _positive(
        multi_person_raw.get("max_growth_ratio", 1.6),
        "geometry.multi_person_box.max_growth_ratio",
        problems,
    )
    head_separation_ratio = _unit_interval(
        multi_person_raw.get("head_separation_ratio", 0.4),
        "geometry.multi_person_box.head_separation_ratio",
        problems,
    )

    anchor_raw = _section(raw, "anchor")
    anchor_window = _positive_int(
        anchor_raw.get("height_history_window_frames", 30),
        "anchor.height_history_window_frames",
        problems,
    )
    anchor_drop_ratio = _unit_interval(
        anchor_raw.get("max_height_drop_ratio", 0.35),
        "anchor.max_height_drop_ratio",
        problems,
    )
    anchor_confirm_frames = _positive_int(
        anchor_raw.get("height_drop_confirm_frames", 3),
        "anchor.height_drop_confirm_frames",
        problems,
    )


    timing_raw = _section(raw, "timing")
    fps_source = str(timing_raw.get("fps_source", "measured"))
    if fps_source not in ALLOWED_FPS_SOURCES:
        problems.append(
            f"timing.fps_source doit être 'measured' (reçu {fps_source!r}) : une valeur "
            "supposée est interdite par la règle 0.2"
        )
    warmup_seconds = _non_negative(
        timing_raw.get("warmup_seconds", 0.5), "timing.warmup_seconds", problems
    )
    warmup_min_frames = _positive_int(
        timing_raw.get("warmup_min_frames", 15), "timing.warmup_min_frames", problems
    )
    confirmation_seconds = _non_negative(
        timing_raw.get("confirmation_seconds", 0.5), "timing.confirmation_seconds", problems
    )
    grace_period_seconds = _positive(
        timing_raw.get("grace_period_seconds", 5.0), "timing.grace_period_seconds", problems
    )
    fps_window = _positive_int(
        timing_raw.get("fps_estimate_window", 30), "timing.fps_estimate_window", problems
    )
    if fps_window is not None and fps_window < 2:
        problems.append("timing.fps_estimate_window doit être >= 2")

    reid_raw = _section(raw, "reid")
    long_term_raw = _section(reid_raw, "long_term")
    similarity_threshold = _cosine(
        long_term_raw.get("similarity_threshold", 0.40),
        "reid.long_term.similarity_threshold",
        problems,
    )
    safety_margin = _non_negative(
        long_term_raw.get("safety_margin", 0.10), "reid.long_term.safety_margin", problems
    )
    v_max_ratio = _positive(
        long_term_raw.get("v_max_ratio", 1.5), "reid.long_term.v_max_ratio", problems
    )
    spatial_margin_ratio = _non_negative(
        long_term_raw.get("spatial_margin_ratio", 0.3),
        "reid.long_term.spatial_margin_ratio",
        problems,
    )
    ambiguous_policy = str(long_term_raw.get("ambiguous_policy", "provisional_person"))
    if ambiguous_policy not in ALLOWED_AMBIGUOUS_POLICIES:
        problems.append(
            f"reid.long_term.ambiguous_policy doit être l'un de "
            f"{ALLOWED_AMBIGUOUS_POLICIES} (reçu {ambiguous_policy!r})"
        )
    gallery_min_confidence = _bounded(
        long_term_raw.get("gallery_min_confidence", 0.70),
        "reid.long_term.gallery_min_confidence",
        problems,
        low=0.0,
        high=1.0,
    )
    gallery_min_crop_height = _positive_int(
        long_term_raw.get("gallery_min_crop_height_px", 32),
        "reid.long_term.gallery_min_crop_height_px",
        problems,
    )
    gallery_min_crop_width = _positive_int(
        long_term_raw.get("gallery_min_crop_width_px", 16),
        "reid.long_term.gallery_min_crop_width_px",
        problems,
    )
    gallery_min_sharpness = _non_negative(
        long_term_raw.get("gallery_min_sharpness", 0.0),
        "reid.long_term.gallery_min_sharpness",
        problems,
    )
    gallery_retention_seconds = _optional_positive(
        long_term_raw.get("gallery_retention_seconds", None),
        "reid.long_term.gallery_retention_seconds",
        problems,
    )
    long_absence_seconds = _positive(
        long_term_raw.get("long_absence_seconds", 2.0),
        "reid.long_term.long_absence_seconds",
        problems,
    )
    long_absence_floor = _optional_cosine(
        long_term_raw.get("long_absence_similarity_floor", None),
        "reid.long_term.long_absence_similarity_floor",
        problems,
    )
    feature_momentum = _momentum(
        long_term_raw.get("feature_momentum", 0.85),
        "reid.long_term.feature_momentum",
        problems,
    )
    descriptor = _descriptor_config(
        long_term_raw.get("descriptor"),
        "reid.long_term.descriptor",
        problems,
    )
    if (
        long_absence_floor is not None
        and similarity_threshold is not None
        and long_absence_floor > similarity_threshold
    ):
        problems.append(
            "reid.long_term.long_absence_similarity_floor "
            f"({long_absence_floor}) doit être <= similarity_threshold "
            f"({similarity_threshold}) : une tolérance progressive ne peut que "
            "descendre sous le seuil nominal, jamais le relever."
        )
    # -- Garde-fou de swap (continuité d'apparence d'une piste non perdue) --
    continuity_raw = _section(reid_raw, "appearance_continuity")
    continuity_threshold = _cosine(
        continuity_raw.get("similarity_threshold", 0.50),
        "reid.appearance_continuity.similarity_threshold",
        problems,
    )
    continuity_max_gap = _non_negative_int(
        continuity_raw.get("max_gap_frames", 2),
        "reid.appearance_continuity.max_gap_frames",
        problems,
    )
    continuity_min_interval = _non_negative(
        continuity_raw.get("min_interval_seconds", 1.0),
        "reid.appearance_continuity.min_interval_seconds",
        problems,
    )
    continuity_freeze_frames = _bounded_int(
        continuity_raw.get("freeze_gallery_frames", 3),
        "reid.appearance_continuity.freeze_gallery_frames",
        problems,
        low=3,
    )

    # -- Correction active des inversions d'identité (swap) ----------------
    swap_correction_raw = _section(reid_raw, "swap_correction")
    swap_margin = _positive(
        swap_correction_raw.get("margin", 0.12),
        "reid.swap_correction.margin",
        problems,
    )

    external_raw = _section(reid_raw, "external_reid")
    image_size_pair = external_raw.get("image_size", (256, 128))
    if not (isinstance(image_size_pair, (list, tuple)) and len(image_size_pair) == 2):
        problems.append("reid.external_reid.image_size doit être une paire [h, w]")
        image_size_pair = (256, 128)

    occupancy_raw = _section(raw, "occupancy")
    disappearance_policy = str(
        occupancy_raw.get("in_progress_disappearance_policy", "deferred_confirmation")
    )
    if disappearance_policy not in ALLOWED_DISAPPEARANCE_POLICIES:
        problems.append(
            f"occupancy.in_progress_disappearance_policy doit être l'un de "
            f"{ALLOWED_DISAPPEARANCE_POLICIES} (reçu {disappearance_policy!r})"
        )
    snapshot_interval = _positive(
        occupancy_raw.get("snapshot_interval_seconds", 1.0),
        "occupancy.snapshot_interval_seconds",
        problems,
    )

    output_raw = _section(raw, "output")
    flush_every = _positive_int(
        output_raw.get("flush_every_n_events", 1), "output.flush_every_n_events", problems
    )
    if flush_every is not None and flush_every < 1:
        problems.append("output.flush_every_n_events doit être >= 1")
    write_mode = str(output_raw.get("write_mode", "append_incremental"))
    if write_mode != "append_incremental":
        problems.append(
            "output.write_mode doit être 'append_incremental' (spec 6.1) "
            f"(reçu {write_mode!r})"
        )

    stabilization_raw = _section(raw, "stabilization")
    locker_ratio = _unit_interval(
        stabilization_raw.get("bbox_locker_min_height_ratio", 0.70),
        "stabilization.bbox_locker_min_height_ratio",
        problems,
    )
    tolerance_ratio = _positive(
        stabilization_raw.get("anchor_stabilizer_tolerance_ratio", 0.15),
        "stabilization.anchor_stabilizer_tolerance_ratio",
        problems,
    )

    source_raw = _section(raw, "source")
    no_detection_seconds = _non_negative(
        source_raw.get("no_detection_seconds", 2.0), "source.no_detection_seconds", problems
    )

    # -- BoT-SORT : seuils distincts du seuil YOLO (spec 2 du prompt) -------
    tracker_raw = _section(raw, "tracker")
    track_high_thresh = _unit_interval(
        tracker_raw.get("track_high_thresh", 0.4), "tracker.track_high_thresh", problems
    )
    track_low_thresh = _unit_interval(
        tracker_raw.get("track_low_thresh", 0.1), "tracker.track_low_thresh", problems
    )
    new_track_thresh = _unit_interval(
        tracker_raw.get("new_track_thresh", 0.7), "tracker.new_track_thresh", problems
    )
    if (
        track_low_thresh is not None
        and track_high_thresh is not None
        and track_low_thresh >= track_high_thresh
    ):
        problems.append(
            "tracker.track_low_thresh doit rester strictement inférieur à "
            f"tracker.track_high_thresh (reçu {track_low_thresh} >= {track_high_thresh}) : "
            "sinon l'association à deux étages n'a plus de second étage."
        )
    if (
        track_high_thresh is not None
        and new_track_thresh is not None
        and new_track_thresh < track_high_thresh
    ):
        problems.append(
            "tracker.new_track_thresh doit être >= tracker.track_high_thresh "
            f"(reçu {new_track_thresh} < {track_high_thresh}) : une nouvelle piste ne "
            "peut pas être créée sur une détection moins fiable que celles utilisées "
            "pour l'association existante."
        )
    # -- Persistance de piste et ReID natif BoT-SORT ----------------------
    track_buffer = _positive_int(
        tracker_raw.get("track_buffer", 150), "tracker.track_buffer", problems
    )
    match_thresh = _unit_interval_inclusive(
        tracker_raw.get("match_thresh", 0.9), "tracker.match_thresh", problems
    )
    proximity_thresh = _unit_interval_inclusive(
        tracker_raw.get("proximity_thresh", 0.5), "tracker.proximity_thresh", problems
    )
    appearance_thresh = _unit_interval_inclusive(
        tracker_raw.get("appearance_thresh", 0.25), "tracker.appearance_thresh", problems
    )
    gmc_method = str(tracker_raw.get("gmc_method", "none"))
    if gmc_method not in ALLOWED_GMC_METHODS:
        problems.append(
            f"tracker.gmc_method doit être l'un de {ALLOWED_GMC_METHODS} "
            f"(reçu {gmc_method!r}) : une compensation de mouvement caméra sur une "
            "caméra fixe est une source d'instabilité des boîtes."
        )

    display_raw = _section(raw, "display")
    logging_raw = _section(raw, "logging")
    diagnostics_raw = _section(raw, "diagnostics")
    diagnostics_interval = _positive(
        diagnostics_raw.get("min_interval_seconds", 1.0),
        "diagnostics.min_interval_seconds",
        problems,
    )

    # Le nom de fenêtre est utilisé par `cv2.namedWindow` / `cv2.setMouseCallback` :
    # le backend Qt d'OpenCV 5.0 ne retrouve plus une fenêtre dont le nom contient
    # un caractère non ASCII (« NULL window handler »), ce qui rendrait la
    # sélection manuelle de ligne — et donc tout le comptage — impossible.
    window_name = str(display_raw.get("window_name", DEFAULT_WINDOW_NAME))
    non_ascii = sorted({character for character in window_name if ord(character) > 127})
    if non_ascii:
        problems.append(
            f"display.window_name contient des caractères non ASCII "
            f"({''.join(non_ascii)!r}) : le backend Qt d'OpenCV ne peut retrouver "
            "qu'une fenêtre au nom ASCII (sinon « NULL window handler » et "
            "calibration impossible). Utiliser un nom sans accent."
        )
    if not window_name.strip():
        problems.append("display.window_name est vide : un nom de fenêtre est requis")

    # -- Assistance tête (additif, désactivé par défaut) -------------------
    presence_raw = _section(raw, "presence")
    head_assist_raw = _section(presence_raw, "head_assist")
    head_assist_enabled = bool(head_assist_raw.get("enabled", False))
    ha_min_kp_conf = _bounded(
        head_assist_raw.get("min_keypoint_confidence", 0.5),
        "presence.head_assist.min_keypoint_confidence",
        problems,
        low=0.0,
        high=1.0,
    )
    ha_max_ext_s = _positive(
        head_assist_raw.get("max_presence_extension_seconds", 10.0),
        "presence.head_assist.max_presence_extension_seconds",
        problems,
    )
    ha_max_head_staleness = _positive_int(
        head_assist_raw.get("max_head_staleness_frames", 1),
        "presence.head_assist.max_head_staleness_frames",
        problems,
    )
    ha_keypoints_used = tuple(
        str(k)
        for k in (
            head_assist_raw.get(
                "keypoints_used",
                ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
            )
            or ()
        )
    )
    if not ha_keypoints_used:
        problems.append(
            "presence.head_assist.keypoints_used ne peut pas être vide : au moins un "
            "point-clé COCO doit être spécifié"
        )
    unknown_kps = [k for k in ha_keypoints_used if k not in ALLOWED_HEAD_KEYPOINTS]
    if unknown_kps:
        problems.append(
            f"presence.head_assist.keypoints_used contient des noms de points-clés "
            f"inconnus {unknown_kps} (autorisés : {ALLOWED_HEAD_KEYPOINTS})"
        )

    if problems:
        raise ConfigError(
            "Configuration invalide :\n  - " + "\n  - ".join(problems)
        )

    assert warmup_seconds is not None and confirmation_seconds is not None
    assert warmup_min_frames is not None and diagnostics_interval is not None
    assert gallery_min_crop_height is not None and gallery_min_crop_width is not None
    assert grace_period_seconds is not None and snapshot_interval is not None
    assert dead_zone_ratio is not None and edge_margin_ratio is not None
    assert min_length_ratio is not None and scale_window is not None
    assert occlusion_ambiguity_ratio is not None
    assert fps_window is not None and flush_every is not None
    assert similarity_threshold is not None and safety_margin is not None
    assert v_max_ratio is not None and spatial_margin_ratio is not None
    assert anchor_window is not None and anchor_drop_ratio is not None
    assert anchor_confirm_frames is not None
    assert min_aspect_ratio_wh is not None and max_aspect_ratio_wh is not None
    assert min_box_height_px is not None and min_box_width_px is not None
    assert long_absence_seconds is not None
    assert swap_margin is not None and max_growth_ratio is not None

    return PipelineConfig(
        schema_version=1,
        source=SourceConfig(
            default=str(source_raw.get("default", "0")),
            reconnect_attempts=int(source_raw.get("reconnect_attempts", 0)),
            no_detection_seconds=float(no_detection_seconds or 2.0),
        ),
        model=ModelConfig(
            path=str(model_raw.get("path", "models/yolo11n.pt")),
            expected_sha256=str(model_raw.get("expected_sha256", "")),
            confidence=float(confidence if confidence is not None else 0.10),
            nms_iou=float(nms_iou or 0.55),
            image_size=int(image_size or 960),
            classes=classes,
            device=device,
        ),
        tracker=TrackerConfig(
            config_path=str(tracker_raw.get("config_path", "src/configs/custom_botsort.yaml")),
            persist=bool(tracker_raw.get("persist", True)),
            track_high_thresh=float(track_high_thresh if track_high_thresh is not None else 0.4),
            track_low_thresh=float(track_low_thresh if track_low_thresh is not None else 0.1),
            new_track_thresh=float(new_track_thresh if new_track_thresh is not None else 0.7),
            track_buffer=int(track_buffer if track_buffer is not None else 150),
            with_reid=bool(tracker_raw.get("with_reid", True)),
            match_thresh=float(match_thresh if match_thresh is not None else 0.9),
            proximity_thresh=float(
                proximity_thresh if proximity_thresh is not None else 0.5
            ),
            appearance_thresh=float(
                appearance_thresh if appearance_thresh is not None else 0.25
            ),
            gmc_method=gmc_method,
        ),
        reid=ReidConfig(
            short_term=str(reid_raw.get("short_term", "native_botsort")),
            long_term=LongTermReidConfig(
                enabled=bool(long_term_raw.get("enabled", True)),
                similarity_threshold=similarity_threshold,
                safety_margin=safety_margin,
                v_max_ratio=v_max_ratio,
                spatial_margin_ratio=spatial_margin_ratio,
                ambiguous_policy=ambiguous_policy,
                gallery_min_confidence=float(
                    gallery_min_confidence if gallery_min_confidence is not None else 0.70
                ),
                gallery_min_crop_height_px=int(gallery_min_crop_height),
                gallery_min_crop_width_px=int(gallery_min_crop_width),
                gallery_min_sharpness=float(gallery_min_sharpness or 0.0),
                gallery_retention_seconds=(
                    None if gallery_retention_seconds is None
                    else float(gallery_retention_seconds)
                ),
                long_absence_seconds=float(
                    long_absence_seconds if long_absence_seconds is not None else 2.0
                ),
                long_absence_similarity_floor=(
                    None if long_absence_floor is None else float(long_absence_floor)
                ),
                feature_momentum=float(
                    feature_momentum if feature_momentum is not None else 0.85
                ),
                descriptor=descriptor,
            ),
            external_reid=ExternalReidConfig(
                enabled=bool(external_raw.get("enabled", False)),
                model_path=str(external_raw.get("model_path", "osnet_x0_25_msmt17.pt")),
                weights=str(external_raw.get("weights", "osnet_x0_25")),
                image_size=(int(image_size_pair[0]), int(image_size_pair[1])),
            ),
            appearance_continuity=AppearanceContinuityConfig(
                enabled=bool(continuity_raw.get("enabled", True)),
                similarity_threshold=float(
                    continuity_threshold if continuity_threshold is not None else 0.50
                ),
                max_gap_frames=int(
                    continuity_max_gap if continuity_max_gap is not None else 2
                ),
                min_interval_seconds=float(
                    continuity_min_interval if continuity_min_interval is not None else 1.0
                ),
                freeze_gallery_frames=int(
                    continuity_freeze_frames
                    if continuity_freeze_frames is not None
                    else 3
                ),
            ),
            swap_correction=SwapCorrectionConfig(
                enabled=bool(swap_correction_raw.get("enabled", True)),
                margin=float(swap_margin if swap_margin is not None else 0.12),
            ),
        ),
        line=LineConfig(
            inside_side=inside_side,
            on_line_policy=on_line_policy,
            min_length_ratio=min_length_ratio,
        ),
        geometry=GeometryConfig(
            dead_zone_ratio=dead_zone_ratio,
            anchor_edge_margin_ratio=edge_margin_ratio,
            scale_window=scale_window,
            occlusion_ambiguity_ratio=float(occlusion_ambiguity_ratio or 0.0),
            min_aspect_ratio_wh=float(min_aspect_ratio_wh or 0.1),
            max_aspect_ratio_wh=float(max_aspect_ratio_wh or 2.5),
            min_box_height_px=float(min_box_height_px or 10.0),
            min_box_width_px=float(min_box_width_px or 10.0),
            multi_person_box=MultiPersonBoxConfig(
                enabled=bool(multi_person_raw.get("enabled", True)),
                max_growth_ratio=float(
                    max_growth_ratio if max_growth_ratio is not None else 1.6
                ),
                head_separation_ratio=float(
                    head_separation_ratio
                    if head_separation_ratio is not None
                    else 0.4
                ),
            ),
        ),
        timing=TimingConfig(
            fps_source=fps_source,
            warmup_seconds=warmup_seconds,
            warmup_min_frames=warmup_min_frames,
            confirmation_seconds=confirmation_seconds,
            grace_period_seconds=grace_period_seconds,
            fps_estimate_window=fps_window,
        ),
        occupancy=OccupancyConfig(
            in_progress_disappearance_policy=disappearance_policy,
            snapshot_interval_seconds=snapshot_interval,
            in_progress_policy=tuple(
                str(x) for x in (occupancy_raw.get("in_progress_policy", ("deferred_confirmation",)) or ())
            ),
        ),
        output=OutputConfig(
            session_root=str(output_raw.get("session_root", "results")),
            events_filename=str(output_raw.get("events_filename", "events.jsonl")),
            video_filename=str(output_raw.get("video_filename", "annotated.mp4")),
            write_mode=write_mode,
            flush_every_n_events=flush_every,
            write_video=bool(output_raw.get("write_video", False)),
            annotated_video_codec=str(output_raw.get("annotated_video_codec", "mp4v")),
        ),
        stabilization=StabilizationConfig(
            use_bbox_locker=bool(stabilization_raw.get("use_bbox_locker", False)),
            bbox_locker_min_height_ratio=float(locker_ratio or 0.70),
            use_anchor_stabilizer=bool(stabilization_raw.get("use_anchor_stabilizer", False)),
            anchor_stabilizer_tolerance_ratio=tolerance_ratio,
        ),
        display=DisplayConfig(
            enabled=bool(display_raw.get("enabled", True)),
            window_name=window_name,
        ),
        logging=LoggingConfig(
            level=str(logging_raw.get("level", "INFO")),
            print_state_transitions=bool(logging_raw.get("print_state_transitions", False)),
        ),
        diagnostics=DiagnosticsConfig(
            enabled=bool(diagnostics_raw.get("enabled", True)),
            min_interval_seconds=diagnostics_interval,
        ),
        anchor=AnchorConfig(
            height_history_window_frames=int(anchor_window or 30),
            max_height_drop_ratio=float(anchor_drop_ratio or 0.35),
            height_drop_confirm_frames=int(anchor_confirm_frames or 3),
        ),
        presence=PresenceConfig(
            head_assist=HeadAssistConfig(
                enabled=head_assist_enabled,
                pose_model_path=str(head_assist_raw.get("pose_model_path", "models/yolo11s-pose.pt")),
                pose_model_expected_sha256=(
                    None if head_assist_raw.get("pose_model_expected_sha256") is None
                    else str(head_assist_raw["pose_model_expected_sha256"])
                ),
                min_keypoint_confidence=float(
                    ha_min_kp_conf if ha_min_kp_conf is not None else 0.5
                ),
                keypoints_used=ha_keypoints_used,
                max_presence_extension_seconds=float(
                    ha_max_ext_s if ha_max_ext_s is not None else 10.0
                ),
                max_head_staleness_frames=int(
                    ha_max_head_staleness
                    if ha_max_head_staleness is not None
                    else 1
                ),
            ),
        ),
        config_path=config_path or DEFAULT_CONFIG_PATH,
    )


def coherence_warnings(config: PipelineConfig) -> list[tuple[str, str]]:
    """Incohérences de configuration qui dégradent le pipeline sans l'arrêter.

    Elles ne sont **pas** des erreurs : un opérateur peut volontairement évaluer
    un seuil YOLO élevé (ablation de la section 9). En revanche elles ne peuvent
    pas être silencieuses : ``main`` les journalise en ``CONFIG_WARNING`` au
    démarrage de la session, avec la conséquence exacte sur le pipeline.

    Returns:
        Liste de couples ``(code, message)``, vide si la configuration est
        cohérente.
    """
    warnings: list[tuple[str, str]] = []
    if config.model.confidence > config.tracker.track_low_thresh:
        warnings.append((
            "yolo_confidence_above_tracker_low_thresh",
            (
                f"model.confidence={config.model.confidence} > "
                f"tracker.track_low_thresh={config.tracker.track_low_thresh} : "
                "les détections dont le score est dans "
                "[track_low_thresh, model.confidence[ n'atteignent jamais "
                "BoT-SORT. Les personnes floues restent donc plus souvent "
                "perdues (nouvelle piste technique ou absence), et le risque "
                "de faux NEW augmente."
            ),
        ))
    return warnings


# ---------------------------------------------------------------------------
# Helpers de validation
# ---------------------------------------------------------------------------
def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"La section {key!r} doit être un mapping YAML")
    return value


def _momentum(value: Any, name: str, problems: list[str]) -> float | None:
    """Momentum d'une moyenne glissante : ``0.0 <= x < 1.0``.

    ``1.0`` est refusé : le descripteur de galerie ne serait alors plus jamais mis
    à jour, ce qui figerait la référence par erreur de configuration plutôt que
    par choix explicite.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un nombre (reçu {value!r})")
        return None
    if not (0.0 <= number < 1.0):
        problems.append(f"{name} doit être dans [0, 1[ (reçu {number})")
        return None
    return number


def _bounded_int(
    value: Any, name: str, problems: list[str], *, low: int, high: int | None = None
) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un entier (reçu {value!r})")
        return None
    if number < low or (high is not None and number > high):
        borne = f">= {low}" if high is None else f"dans [{low}, {high}]"
        problems.append(f"{name} doit être {borne} (reçu {number})")
        return None
    return number


def _descriptor_config(
    raw: Any, name: str, problems: list[str]
) -> DescriptorConfig:
    """Valide la section ``reid.long_term.descriptor`` (descripteur HSV par bandes)."""
    if raw is None:
        return DescriptorConfig()
    if not isinstance(raw, Mapping):
        problems.append(f"{name} doit être un mapping (reçu {type(raw).__name__})")
        return DescriptorConfig()
    unknown = sorted(set(raw) - {
        "horizontal_crop_ratio", "bands", "band_weights", "bins_hue",
        "bins_saturation",
    })
    if unknown:
        problems.append(f"{name} contient des clés inconnues {unknown}")

    crop_ratio = _bounded(
        raw.get("horizontal_crop_ratio", 0.15),
        f"{name}.horizontal_crop_ratio", problems, low=0.0, high=0.45,
    )
    bands = _positive_int(raw.get("bands", 3), f"{name}.bands", problems)
    bins_hue = _positive_int(raw.get("bins_hue", 24), f"{name}.bins_hue", problems)
    bins_saturation = _positive_int(
        raw.get("bins_saturation", 8), f"{name}.bins_saturation", problems
    )
    weights_raw = raw.get("band_weights", (1.0, 1.0, 0.5))
    weights: tuple[float, ...] = ()
    if not isinstance(weights_raw, (list, tuple)) or not weights_raw:
        problems.append(f"{name}.band_weights doit être une liste non vide de nombres")
    else:
        parsed: list[float] = []
        for index, weight in enumerate(weights_raw):
            number = _non_negative(weight, f"{name}.band_weights[{index}]", problems)
            parsed.append(0.0 if number is None else number)
        weights = tuple(parsed)
        if all(weight == 0.0 for weight in weights):
            problems.append(
                f"{name}.band_weights ne peut pas être entièrement nul : le "
                "descripteur serait alors toujours nul et inexploitable."
            )
    if bands is not None and weights and len(weights) != bands:
        problems.append(
            f"{name}.band_weights ({len(weights)} valeurs) doit avoir autant "
            f"d'entrées que {name}.bands ({bands}) : un poids manquant rendrait "
            "une bande silencieusement inutilisée."
        )

    return DescriptorConfig(
        horizontal_crop_ratio=float(crop_ratio if crop_ratio is not None else 0.15),
        bands=int(bands if bands is not None else 3),
        band_weights=weights or (1.0, 1.0, 0.5),
        bins_hue=int(bins_hue if bins_hue is not None else 24),
        bins_saturation=int(bins_saturation if bins_saturation is not None else 8),
    )


def _unit_interval(value: Any, name: str, problems: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un nombre (reçu {value!r})")
        return None
    if not (0.0 < number < 1.0):
        problems.append(f"{name} doit être strictement dans ]0, 1[ (reçu {number})")
        return None
    return number


def _unit_interval_inclusive(value: Any, name: str, problems: list[str]) -> float | None:
    """Comme :func:`_unit_interval`, mais ``1.0`` est une valeur légitime.

    BoT-SORT accepte un seuil d'association à 1.0 (toutes les associations
    plausibles passent) ; refuser 1.0 serait une contrainte arbitraire. ``0.0``
    reste refusé : il n'associerait plus rien du tout.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un nombre (reçu {value!r})")
        return None
    if not (0.0 < number <= 1.0):
        problems.append(f"{name} doit être dans ]0, 1] (reçu {number})")
        return None
    return number


def _optional_positive(value: Any, name: str, problems: list[str]) -> float | None:
    """Valeur ``> 0`` facultative (``None`` = non déclarée, donc repli par défaut)."""
    if value is None:
        return None
    return _positive(value, name, problems)


def _optional_cosine(value: Any, name: str, problems: list[str]) -> float | None:
    if value is None:
        return None
    return _cosine(value, name, problems)


def _cosine(value: Any, name: str, problems: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un nombre (reçu {value!r})")
        return None
    if not (-1.0 <= number <= 1.0):
        problems.append(f"{name} doit être dans [-1, 1] (reçu {number})")
        return None
    return number


def _positive(value: Any, name: str, problems: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un nombre (reçu {value!r})")
        return None
    if number <= 0:
        problems.append(f"{name} doit être > 0 (reçu {number})")
        return None
    return number


def _non_negative(value: Any, name: str, problems: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un nombre (reçu {value!r})")
        return None
    if number < 0:
        problems.append(f"{name} doit être >= 0 (reçu {number})")
        return None
    return number


def _bounded(
    value: Any, name: str, problems: list[str], *, low: float, high: float
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un nombre (reçu {value!r})")
        return None
    if not (low <= number <= high):
        problems.append(f"{name} doit être dans [{low}, {high}] (reçu {number})")
        return None
    return number


def _non_negative_int(value: Any, name: str, problems: list[str]) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un entier (reçu {value!r})")
        return None
    if number < 0:
        problems.append(f"{name} doit être >= 0 (reçu {number})")
        return None
    return number


def _positive_int(value: Any, name: str, problems: list[str]) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        problems.append(f"{name} doit être un entier (reçu {value!r})")
        return None
    if number <= 0:
        problems.append(f"{name} doit être > 0 (reçu {number})")
        return None
    return number
