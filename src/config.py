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
    path: str = "models/yolo11n.pt"
    expected_sha256: str = ""
    confidence: float = 0.25
    nms_iou: float = 0.55
    image_size: int = 960
    classes: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class TrackerConfig:
    config_path: str = "src/configs/custom_botsort.yaml"
    persist: bool = True


@dataclass(frozen=True)
class LongTermReidConfig:
    enabled: bool = True
    similarity_threshold: float = 0.40
    safety_margin: float = 0.10
    v_max_ratio: float = 1.5
    spatial_margin_ratio: float = 0.3
    ambiguous_policy: str = "provisional_person"


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
class GeometryConfig:
    dead_zone_ratio: float = 0.15
    anchor_edge_margin_ratio: float = 0.02
    scale_window: int = 61


@dataclass(frozen=True)
class TimingConfig:
    fps_source: str = "measured"
    warmup_seconds: float = 1.0
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
            confirmation_frames=max(1, round(self.confirmation_seconds * fps)),
            grace_period_frames=max(1, round(self.grace_period_seconds * fps)),
        )


@dataclass(frozen=True)
class FrameBudget:
    """Durées de la configuration exprimées en frames pour un FPS mesuré."""

    fps: float
    warmup_frames: int
    confirmation_frames: int
    grace_period_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    confidence = _unit_interval(model_raw.get("confidence", 0.25), "model.confidence", problems)
    nms_iou = _unit_interval(model_raw.get("nms_iou", 0.55), "model.nms_iou", problems)
    image_size = _positive_int(model_raw.get("image_size", 960), "model.image_size", problems)
    if image_size is not None and image_size % 32 != 0:
        problems.append(
            f"model.image_size doit être un multiple de 32 (reçu {image_size})"
        )
    classes = tuple(int(c) for c in model_raw.get("classes", (0,)) or ())

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

    timing_raw = _section(raw, "timing")
    fps_source = str(timing_raw.get("fps_source", "measured"))
    if fps_source not in ALLOWED_FPS_SOURCES:
        problems.append(
            f"timing.fps_source doit être 'measured' (reçu {fps_source!r}) : une valeur "
            "supposée est interdite par la règle 0.2"
        )
    warmup_seconds = _non_negative(
        timing_raw.get("warmup_seconds", 1.0), "timing.warmup_seconds", problems
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
    tracker_raw = _section(raw, "tracker")
    display_raw = _section(raw, "display")
    logging_raw = _section(raw, "logging")

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

    if problems:
        raise ConfigError(
            "Configuration invalide :\n  - " + "\n  - ".join(problems)
        )

    assert warmup_seconds is not None and confirmation_seconds is not None
    assert grace_period_seconds is not None and snapshot_interval is not None
    assert dead_zone_ratio is not None and edge_margin_ratio is not None
    assert min_length_ratio is not None and scale_window is not None
    assert fps_window is not None and flush_every is not None
    assert similarity_threshold is not None and safety_margin is not None
    assert v_max_ratio is not None and spatial_margin_ratio is not None

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
            confidence=float(confidence or 0.25),
            nms_iou=float(nms_iou or 0.55),
            image_size=int(image_size or 960),
            classes=classes,
        ),
        tracker=TrackerConfig(
            config_path=str(tracker_raw.get("config_path", "src/configs/custom_botsort.yaml")),
            persist=bool(tracker_raw.get("persist", True)),
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
            ),
            external_reid=ExternalReidConfig(
                enabled=bool(external_raw.get("enabled", False)),
                model_path=str(external_raw.get("model_path", "osnet_x0_25_msmt17.pt")),
                weights=str(external_raw.get("weights", "osnet_x0_25")),
                image_size=(int(image_size_pair[0]), int(image_size_pair[1])),
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
        ),
        timing=TimingConfig(
            fps_source=fps_source,
            warmup_seconds=warmup_seconds,
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
        config_path=config_path or DEFAULT_CONFIG_PATH,
    )


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
