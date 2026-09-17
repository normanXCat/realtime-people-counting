"""Point d'entrée **unique** du système de comptage de personnes par caméra fixe.

Pipeline officiel (spec section 1) :

    source vidéo
      -> YOLO11 (poids + hash vérifiés, modèle configurable)
      -> BoT-SORT natif (persistance + ReID court terme)
      -> gestionnaire d'identités à deux niveaux (person_id stable)
      -> ancre pieds (bas centre de boîte) + garde-fou bord d'image
      -> ligne virtuelle unique calibrée (côté intérieur explicite)
      -> machine à états avec hystérésis relative
      -> comptage IN/OUT + occupation encadrée
      -> journal d'événements JSON Lines incrémental

Corrections apportées par rapport à l'audit (§2.1) :

- plus de ``input()`` bloquant : le mode sans affichage est réellement headless
  (spec 5.1) ;
- la ligne virtuelle est **obligatoirement définie manuellement** par
  l'opérateur au démarrage (première image affichée, deux clics, « C » pour
  valider, « G » pour recommencer, « I » pour inverser IN/OUT, « Échap » pour
  annuler). Aucune ligne par défaut ne peut servir au comptage et le pipeline
  refuse de démarrer tant que la ligne n'est pas validée ;
- une seule résolution de source (spec 5.2), jamais de chaîne ``"0"`` ambiguë ;
- le FPS de sortie provient d'une **mesure** glissante, jamais d'une réouverture
  de la source (spec 5.3) ;
- distinction explicite fin de flux / absence de détection / erreur de source
  (spec 5.5) ;
- distinction explicite entre « aucune détection » et « boîtes présentes sans
  identifiant technique » : le second cas ne produit **aucun** identifiant
  inventé, aucun ``IN``/``OUT``/``NEW``, et un événement ``TRACK_ID_ABSENT``
  (diagnostic de perte de piste, spec 3 du prompt) ;
- les incohérences de configuration qui dégradent le pipeline sans l'empêcher
  de tourner sont journalisées en ``CONFIG_WARNING`` au démarrage, jamais
  silencieuses ;
- latence instrumentée par étape (spec 6.5) avec moyenne, médiane, minimum, P95 ;
- les événements sont écrits en JSON Lines dans ``results/{session_id}/``, jamais
  dans le répertoire courant (spec 6.1, 6.4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import cv2
import yaml

from anchor_stabilizer import AnchorStabilizer
from bbox_height_locker import BBoxHeightLocker
from calibration import (
    CalibrationCancelled,
    CalibrationUnavailable,
    SourceUnreadable,
    ValidatedLine,
    display_available,
    sanitize_window_name,
    select_line,
)
from config import (
    ALLOWED_INSIDE_SIDES,
    ConfigError,
    FrameBudget,
    PipelineConfig,
    coherence_warnings,
    load_config,
    override_config,
)
from events import EventLogger
from geometry import LineError, VirtualLine
from metrics import FpsEstimator, LatencyProfiler, Timer
from occupancy_manager import Detection, OccupancyManager
from track_diagnostics import (
    BOXES_EMPTY,
    BOXES_MISSING,
    BOXES_PRESENT,
    DetectionDiagnostics,
    DetectionSnapshot,
    box_count,
)

#: Modes d'exécution du point d'entrée. Le comptage est le **seul** mode
#: officiel : il commence obligatoirement par la définition manuelle de la ligne
#: virtuelle, puis enchaîne YOLO11 + BoT-SORT + comptage. Le paramètre existe pour
#: rendre ce choix explicite (et journalisé) plutôt qu'implicite.
ALLOWED_MODES = ("comptage",)

#: Dépendances dont la version est enregistrée dans le dossier de session (spec 6.3).
TRACKED_DEPENDENCIES = (
    "ultralytics", "torch", "torchvision", "opencv-python", "numpy", "PyYAML", "lap",
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comptage de personnes YOLO11 + BoT-SORT (pipeline unique)"
    )
    parser.add_argument("--config", default=None, help="YAML de configuration (défaut : config/pipeline.yaml)")
    parser.add_argument(
        "--mode",
        default=ALLOWED_MODES[0],
        choices=list(ALLOWED_MODES),
        help="Mode d'exécution (seul mode officiel : « comptage »)",
    )
    parser.add_argument("--source", default=None, help="Index caméra ou chemin vidéo (défaut : configuration)")
    parser.add_argument("--output-root", default=None, help="Racine des résultats (défaut : configuration)")
    parser.add_argument("--session-id", default=None, help="Identifiant de session (défaut : horodatage)")
    parser.add_argument(
        "--inside-side",
        default=None,
        choices=list(ALLOWED_INSIDE_SIDES),
        help="Orientation INITIALE proposée dans la fenêtre de sélection "
             "(inversible en direct avec la touche « I »)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Mode headless : aucune prévisualisation pendant le traitement "
             "(la sélection manuelle de la ligne reste obligatoire)",
    )
    parser.add_argument("--write-video", action="store_true", help="Écrire la vidéo annotée")
    parser.add_argument("--max-frames", type=int, default=None, help="Arrêt après N frames (diagnostic)")
    parser.add_argument("--model", default=None, help="Surcharge du chemin des poids YOLO")
    parser.add_argument("--conf", type=float, default=None, help="Surcharge du seuil de confiance")
    parser.add_argument("--iou", type=float, default=None, help="Surcharge du seuil NMS IoU")
    parser.add_argument("--imgsz", type=int, default=None, help="Surcharge de la résolution d'inférence")
    parser.add_argument("--no-long-term-reid", action="store_true", help="Baseline 1 : ReID long terme désactivé")
    parser.add_argument("--external-reid", action="store_true", help="Variante 5 : extracteur d'apparence profond")
    parser.add_argument("--no-safety-margin", action="store_true", help="Variante 2 : marge de sécurité désactivée")
    parser.add_argument("--no-spatial-constraint", action="store_true", help="Variante 3 : contrainte spatio-temporelle désactivée")
    parser.add_argument("--use-bbox-locker", action="store_true", help="Variante 4 : verrouillage de hauteur de bbox")
    parser.add_argument("--use-anchor-stabilizer", action="store_true", help="Variante 4 : stabilisation d'ancre")
    return parser


def apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Applique les surcharges CLI en repassant par la validation de configuration."""
    overrides: dict[str, dict[str, Any]] = {}
    if args.model:
        overrides["model"] = {"path": args.model}
    model_overrides: dict[str, Any] = dict(overrides.get("model", {}))
    if args.conf is not None:
        model_overrides["confidence"] = args.conf
    if args.iou is not None:
        model_overrides["nms_iou"] = args.iou
    if args.imgsz is not None:
        model_overrides["image_size"] = args.imgsz
    if model_overrides:
        overrides["model"] = model_overrides

    if args.source is not None:
        overrides["source"] = {"default": args.source}
    if args.output_root is not None:
        overrides["output"] = {"session_root": args.output_root}
    if args.write_video:
        overrides["output"] = {**overrides.get("output", {}), "write_video": True}
    if args.no_show or not display_available():
        overrides["display"] = {"enabled": False}

    long_term: dict[str, Any] = {}
    if args.no_long_term_reid:
        long_term["enabled"] = False
    if args.no_safety_margin:
        long_term["safety_margin"] = 0.0
    if args.no_spatial_constraint:
        long_term["v_max_ratio"] = 1e6
        long_term["spatial_margin_ratio"] = 1e6
    if long_term:
        overrides["reid"] = {"long_term": {**long_term}}
    if args.external_reid:
        overrides["reid"] = {
            **overrides.get("reid", {}),
            "external_reid": {"enabled": True},
        }

    stabilization: dict[str, Any] = {}
    if args.use_bbox_locker:
        stabilization["use_bbox_locker"] = True
    if args.use_anchor_stabilizer:
        stabilization["use_anchor_stabilizer"] = True
    if stabilization:
        overrides["stabilization"] = stabilization

    if args.inside_side is not None:
        # Orientation initiale seulement : les points proviennent des clics de
        # l'opérateur, la valeur peut être inversée en direct dans la fenêtre.
        overrides["line"] = {"inside_side": args.inside_side}

    # Revalidation complète après surcharge : une CLI ne peut pas produire une
    # configuration invalide silencieusement (spec 5.6).
    return override_config(config, overrides)


# ---------------------------------------------------------------------------
# Session, poids, environnement
# ---------------------------------------------------------------------------
def resolve_source(raw: str) -> int | str:
    """Résout la source **une seule fois** : index entier ou chemin (spec 5.2)."""
    candidate = raw.strip()
    if candidate.isdigit():
        return int(candidate)
    return candidate


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path: Path, expected_sha256: str) -> str:
    """Vérifie l'existence et le hash des poids (spec 0.5, règle de déterminisme)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Poids introuvables : {path}. Aucun téléchargement automatique n'est "
            "effectué : le chemin doit être explicite et vérifiable."
        )
    digest = sha256_of(path)
    if expected_sha256 and digest != expected_sha256:
        raise ConfigError(
            f"Hash des poids incorrect pour {path}\n  attendu : {expected_sha256}\n"
            f"  obtenu  : {digest}\nLes résultats ne seraient pas reproductibles."
        )
    return digest


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in TRACKED_DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def prepare_session(config: PipelineConfig, session_id: str) -> Path:
    """Crée le dossier de session (spec 6.3) et retourne sa racine."""
    root = config.resolve_path(config.output.session_root) / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_resolved_config(
    config: PipelineConfig, session_root: Path, session_id: str
) -> Path:
    """Écrit la configuration **effectivement appliquée** dans la session (spec 6.3).

    Appelé après la validation de la ligne : le fichier contient l'orientation
    intérieur/extérieur réellement retenue par l'opérateur. Il est rechargeable
    tel quel (``--config results/<session>/config_resolved.yaml``) : il ne
    contient que les sections du schéma, le chemin de la configuration source
    restant en commentaire d'en-tête.
    """
    resolved_path = session_root / "config_resolved.yaml"
    header = (
        "# Configuration résolue — copie exacte de la configuration exécutée.\n"
        f"# config_path: {config.config_path}\n"
        f"# session_id : {session_id}\n"
        f"# source     : {config.source.default}\n"
        "# Rejouable : python src/main.py --config <ce fichier>\n"
    )
    resolved_path.write_text(
        header + yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ligne virtuelle : sélection manuelle obligatoire
# ---------------------------------------------------------------------------
def calibrate(
    source_or_config: Any = 0,
    source: int | str | None = None,
    *,
    headless_requested: bool = False,
    no_show: bool = False,
    window_name: str | None = None,
    inside_side: str | None = None,
    min_length_ratio: float | None = None,
) -> ValidatedLine:
    """Affiche la première image et attend que l'utilisateur définisse manuellement la ligne.

    Comportement strict :
    1. Affiche la première image de la source dans une fenêtre OpenCV.
    2. Attend exactement deux clics (point 1 puis point 2) ; aucun troisième clic n'est accepté.
    3. Affiche les deux points et trace immédiatement la ligne entre eux.
    4. C valide la ligne uniquement si deux points distincts de longueur valide sont présents.
    5. G réinitialise les points et recommence la sélection.
    6. Échap annule proprement sans lancer le comptage.

    Règles strictes :
    - Ne jamais utiliser automatiquement une ligne horizontale par défaut à 60 % :
      l'ancien repli (`if no_show: return (0.0, 0.6), (1.0, 0.6)`) a été supprimé.
    - Si --no-show est demandé sans interface graphique disponible, affiche une erreur claire
      indiquant qu'une calibration manuelle est impossible sans affichage.
    - Retourne les deux points en coordonnées normalisées entre 0 et 1.
    """
    is_headless = bool(headless_requested or no_show)
    if isinstance(source_or_config, PipelineConfig):
        cfg = source_or_config
        src = source if source is not None else cfg.source.default
        w_name = window_name or f"{cfg.display.window_name} - ligne de comptage"
        side = inside_side or cfg.line.inside_side
        min_ratio = min_length_ratio if min_length_ratio is not None else cfg.line.min_length_ratio
    else:
        src = source_or_config if source is None else source
        w_name = window_name or "People counting - ligne de comptage"
        side = inside_side or "negative"
        min_ratio = min_length_ratio if min_length_ratio is not None else 0.05

    if is_headless and not display_available():
        raise CalibrationUnavailable(
            "Calibration manuelle impossible : --no-show a été demandé et aucune "
            "interface graphique n'est disponible, la première image ne peut donc "
            "pas être affichée. La ligne virtuelle doit être définie à la main "
            "(deux clics puis « C ») : le comptage n'est pas lancé."
        )

    return select_line(
        src,
        inside_side=side,
        min_length_ratio=min_ratio,
        window_name=w_name,
    )


def perform_line_selection(
    config: PipelineConfig,
    source: int | str,
    *,
    headless_requested: bool = False,
    **kwargs: Any,
) -> ValidatedLine:
    """Point d'entrée du pipeline pour la calibration manuelle obligatoire."""
    return calibrate(
        config,
        source,
        headless_requested=headless_requested,
        **kwargs,
    )


#: Type d'événement et code de sortie par cause d'échec de sélection.
_LINE_FAILURES: tuple[tuple[type[BaseException], str, int], ...] = (
    (SourceUnreadable, "SOURCE_ERROR", 5),
    (CalibrationUnavailable, "LINE_CALIBRATION_UNAVAILABLE", 4),
    (CalibrationCancelled, "LINE_CALIBRATION_CANCELLED", 4),
)


def abort_line_selection(
    *,
    logger: EventLogger,
    session_root: Path,
    session_id: str,
    error: BaseException,
    start_s: float,
) -> int:
    """Journalise explicitement un échec de sélection et arrête sans compter.

    Aucun comptage n'a lieu : ni occupation, ni IN/OUT, ni vidéo annotée. Le
    motif est écrit dans le journal de session et sur la sortie d'erreur.
    """
    event_type, exit_code = "LINE_CALIBRATION_CANCELLED", 4
    for failure_type, failure_event, failure_code in _LINE_FAILURES:
        if isinstance(error, failure_type):
            event_type, exit_code = failure_event, failure_code
            break

    elapsed = time.perf_counter() - start_s
    details: dict[str, Any] = {"frames_processed": 0}
    if event_type == "SOURCE_ERROR":
        details["error"] = f"{type(error).__name__}: {error}"
    else:
        details["reason"] = str(error)
    logger.emit(event_type, elapsed, 0, **details)
    logger.emit(
        "SESSION_END", elapsed, 0, frames_processed=0, duration_s=round(elapsed, 3),
        fps_mean=0.0,
    )
    write_json(
        session_root / "summary.json",
        {
            "session_id": session_id,
            "aborted": event_type,
            "reason": str(error),
            "frames_processed": 0,
            "counters": None,
            "message": "Ligne virtuelle non validée : aucun comptage n'a été lancé.",
        },
    )
    print(f"[LIGNE] {error}", file=sys.stderr)
    print(
        "[LIGNE] Aucun comptage lancé : la ligne virtuelle doit être définie et "
        "validée par l'opérateur (« C »).",
        file=sys.stderr,
    )
    logger.close()
    return exit_code


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        config = apply_cli_overrides(config, args)
    except ConfigError as error:
        print(f"[CONFIG] {error}", file=sys.stderr)
        return 2

    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    source = resolve_source(config.source.default)
    model_path = config.resolve_path(config.model.path)
    tracker_path = config.resolve_path(config.tracker.config_path)
    session_id = args.session_id or new_session_id()
    session_root = prepare_session(config, session_id)

    display_enabled = config.display.enabled
    write_video = config.output.write_video
    no_detection_seconds = config.source.no_detection_seconds

    writer: cv2.VideoWriter | None = None
    video_path = session_root / config.output.video_filename
    buffered_rendered: list[Any] = []
    profiler = LatencyProfiler()
    fps_estimator = FpsEstimator(window=config.timing.fps_estimate_window)
    logger = EventLogger(
        session_root / config.output.events_filename,
        session_id=session_id,
        flush_every_n_events=config.output.flush_every_n_events,
    )
    budget: FrameBudget | None = None
    diagnostics = DetectionDiagnostics(
        config.diagnostics,
        event_sink=logger,
        track_high_thresh=config.tracker.track_high_thresh,
    )
    start_s = time.perf_counter()

    try:
        model_sha256 = verify_weights(model_path, config.model.expected_sha256)
    except (FileNotFoundError, ConfigError) as error:
        print(f"[MODEL] {error}", file=sys.stderr)
        logger.close()
        return 3
    if not tracker_path.exists():
        print(f"[TRACKER] Configuration introuvable : {tracker_path}", file=sys.stderr)
        logger.close()
        return 3

    logger.emit(
        "SESSION_START",
        0.0,
        0,
        config_path=str(config.config_path),
        source=str(source),
        model_path=str(model_path),
        model_sha256=model_sha256,
        fps_source=config.timing.fps_source,
        line_policy="manual_required",
        mode=args.mode,
        tracker_high_thresh=config.tracker.track_high_thresh,
        tracker_low_thresh=config.tracker.track_low_thresh,
        tracker_new_thresh=config.tracker.new_track_thresh,
        warmup_min_frames=config.timing.warmup_min_frames,
        warmup_seconds=config.timing.warmup_seconds,
    )
    # Incohérences de configuration : journalisées explicitement (un seuil YOLO
    # plus haut que le seuil bas du tracker rend la récupération des personnes
    # floues impossible avant même que BoT-SORT voie la détection).
    for code, message in coherence_warnings(config):
        logger.emit(
            "CONFIG_WARNING", 0.0, 0, code=code, details=message, source="config"
        )
        print(f"[CONFIG] {code} : {message}", file=sys.stderr)
    print(
        f"[MODE] {args.mode} | source={source} | étape 1/2 : définition manuelle "
        "de la ligne virtuelle (la première image va s'afficher)"
    )

    # -- Ligne virtuelle : sélection manuelle obligatoire --------------------
    # Aucun repli n'est possible : sans ligne validée par l'opérateur, le
    # comptage ne démarre pas (et le refus est journalisé, jamais silencieux).
    try:
        validated = perform_line_selection(
            config, source, headless_requested=args.no_show
        )
    except (SourceUnreadable, CalibrationUnavailable, CalibrationCancelled) as error:
        return abort_line_selection(
            logger=logger,
            session_root=session_root,
            session_id=session_id,
            error=error,
            start_s=start_s,
        )

    # L'orientation effectivement retenue dans la fenêtre est celle appliquée à
    # tout le reste du pipeline: dessin, distance signée, franchissements,
    # comptage. Elle est figée dans la configuration résolue de la session.
    config = override_config(config, {"line": {"inside_side": validated.inside_side}})
    resolved_config_path = write_resolved_config(config, session_root, session_id)
    calibration_path = session_root / "calibration.json"
    write_json(
        calibration_path,
        {"session_id": session_id, "source": str(source), "line": validated.to_dict()},
    )
    logger.emit(
        "LINE_VALIDATED",
        0.0,
        0,
        p1=list(validated.p1),
        p2=list(validated.p2),
        inside_side=validated.inside_side,
        frame_resolution=[validated.frame_width, validated.frame_height],
        length_normalized=round(validated.length_ratio, 6),
        display_scale=round(validated.display_scale, 6),
        confirmed_at=validated.confirmed_at,
        calibration_path=str(calibration_path),
    )

    try:
        line = VirtualLine(
            p1=validated.p1,
            p2=validated.p2,
            inside_side=validated.inside_side,
            on_line_policy=config.line.on_line_policy,
            min_length_ratio=config.line.min_length_ratio,
        )
    except (LineError, KeyError) as error:  # défense en profondeur : ligne déjà validée
        print(f"[LINE] {error}", file=sys.stderr)
        logger.close()
        return 4

    locker = (
        BBoxHeightLocker(min_height_ratio=config.stabilization.bbox_locker_min_height_ratio)
        if config.stabilization.use_bbox_locker
        else None
    )
    stabilizer = (
        AnchorStabilizer(tolerance_px=1.0) if config.stabilization.use_anchor_stabilizer else None
    )
    occupancy = OccupancyManager(
        config,
        event_sink=logger,
        line=line,
        bbox_locker=locker,
        anchor_stabilizer=stabilizer,
        profiler=profiler,
    )

    frame_index = 0
    last_detection_s = 0.0
    no_detection_announced = False

    print(
        f"[SESSION] {session_id} | source={source} | modèle={model_path.name} "
        f"| ligne(manuelle)={line.p1}->{line.p2} | intérieur={line.inside_side} "
        f"| headless={not display_enabled}"
    )

    try:
        from ultralytics import YOLO  # import tardif : évite de charger torch pour --help

        model = YOLO(str(model_path))
        generator = model.track(
            source=source,
            tracker=str(tracker_path),
            persist=config.tracker.persist,
            classes=list(config.model.classes),
            conf=config.model.confidence,
            iou=config.model.nms_iou,
            imgsz=config.model.image_size,
            show=False,
            stream=True,
            verbose=False,
        )
    except Exception as error:
        # Source inouvrable : cas distinct d'une fin de flux (spec 5.5).
        logger.emit(
            "SOURCE_ERROR",
            0.0,
            0,
            error=f"{type(error).__name__}: {error}",
            frames_processed=0,
        )
        print(f"[SOURCE] Ouverture impossible : {error}", file=sys.stderr)
        _finalize(
            logger=logger, occupancy=occupancy, profiler=profiler,
            fps_estimator=fps_estimator, session_root=session_root,
            session_id=session_id, config=config,
            resolved_config_path=resolved_config_path, model_path=model_path,
            model_sha256=model_sha256, frames_processed=0,
            duration_s=time.perf_counter() - start_s, budget=None,
            diagnostics=diagnostics,
        )
        logger.close()
        return 5

    # Nom de fenêtre assaini : un titre non ASCII fait échouer le backend Qt
    # d'OpenCV (« NULL window handler ») dès qu'un callback souris est installé.
    display_window_name = sanitize_window_name(config.display.window_name)
    window_ready = False
    exit_code = 0
    try:
        while True:
            with Timer() as capture_timer:
                try:
                    result = next(generator)
                except StopIteration:
                    elapsed = time.perf_counter() - start_s
                    logger.emit(
                        "SOURCE_END_OF_STREAM", elapsed, frame_index,
                        frames_processed=frame_index, duration_s=round(elapsed, 3),
                    )
                    print(f"\n[MEDIA] Fin de flux après {frame_index} frames")
                    break
                except Exception as error:  # source illisible, décodage cassé…
                    elapsed = time.perf_counter() - start_s
                    logger.emit(
                        "SOURCE_ERROR", elapsed, frame_index,
                        error=f"{type(error).__name__}: {error}",
                        frames_processed=frame_index,
                    )
                    print(f"\n[SOURCE] Erreur de lecture : {error}", file=sys.stderr)
                    exit_code = 5
                    break
            profiler.record("capture", capture_timer.ms)

            frame_index += 1
            frame = getattr(result, "orig_img", None)
            if frame is None:
                continue
            height, width = frame.shape[:2]

            speed = getattr(result, "speed", None) or {}
            profiler.record("detection", float(speed.get("inference", 0.0)))
            profiler.record("tracking", float(speed.get("postprocess", 0.0)))

            detections: list[Detection] = []
            observed_confidences: list[float] = []
            boxes = getattr(result, "boxes", None)
            boxes_count = box_count(boxes)
            tracker_ids_present = (
                boxes is not None and getattr(boxes, "id", None) is not None
            )
            if tracker_ids_present and boxes_count:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                confidences = (
                    boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(ids)
                )
                for box, track_id, confidence in zip(xyxy, ids, confidences):
                    detections.append(
                        Detection(
                            technical_track_id=int(track_id),
                            bbox=box,
                            confidence=float(confidence),
                        )
                    )
                    observed_confidences.append(float(confidence))

            # Diagnostic systématique des détections brutes YOLO avant filtrage (spec correction 1)
            raw_boxes_array = None
            raw_conf_array = None
            if boxes is not None:
                if hasattr(boxes, "xyxy") and boxes.xyxy is not None:
                    raw_boxes_array = boxes.xyxy.cpu().numpy()
                if hasattr(boxes, "conf") and boxes.conf is not None:
                    raw_conf_array = boxes.conf.cpu().numpy()
            diagnostics.observe_yolo_detections(
                raw_boxes_array,
                raw_conf_array,
                confidence_threshold=config.model.confidence,
                frame_index=frame_index,
            )

            timestamp_s = time.perf_counter() - start_s
            # Diagnostic de la frame : distingue « aucune détection » de « boîtes
            # présentes sans identifiant technique ». Le second cas laisse
            # `detections` vide (aucun identifiant n'est inventé) : les pistes
            # déjà connues suivent donc la politique d'occultation, sans qu'aucun
            # IN/OUT/NEW ne puisse être produit.
            diagnostics.observe(
                DetectionSnapshot(
                    boxes_status=(
                        BOXES_MISSING if boxes is None
                        else BOXES_PRESENT if boxes_count
                        else BOXES_EMPTY
                    ),
                    boxes_count=boxes_count,
                    ids_present=bool(tracker_ids_present and boxes_count),
                    confidences=tuple(observed_confidences),
                ),
                timestamp_s,
                frame_index,
            )
            fps_estimator.tick(timestamp_s)
            if fps_estimator.is_ready:
                budget = config.timing.frame_budget(fps_estimator.require_fps())
                occupancy.set_frame_budget(budget)

            occupancy.process_frame(detections, frame, timestamp_s, frame_index)

            with Timer() as render_timer:
                rendered = frame.copy()
                occupancy.draw_overlay(rendered)
            profiler.record("render", render_timer.ms)

            if not detections:
                if timestamp_s - last_detection_s >= no_detection_seconds:
                    if not no_detection_announced:
                        logger.emit(
                            "SOURCE_NO_DETECTION", timestamp_s, frame_index,
                            consecutive_frames=frame_index,
                            elapsed_s=round(timestamp_s - last_detection_s, 3),
                        )
                        no_detection_announced = True
            else:
                last_detection_s = timestamp_s
                no_detection_announced = False

            if locker is not None or stabilizer is not None:
                _purge_stabilizers(locker, stabilizer, detections)

            if write_video:
                if writer is None and fps_estimator.is_ready:
                    writer = _open_writer(video_path, config, fps_estimator.require_fps(), width, height)
                    for buffered in buffered_rendered:
                        writer.write(buffered)
                    buffered_rendered.clear()
                with Timer() as write_timer:
                    if writer is not None:
                        writer.write(rendered)
                    else:
                        buffered_rendered.append(rendered)
                profiler.record("write", write_timer.ms)

            if display_enabled:
                try:
                    if not window_ready:
                        cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
                        window_ready = True
                    cv2.imshow(display_window_name, rendered)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        print("\n[CONTROL] Arrêt demandé par l'opérateur")
                        break
                except cv2.error as error:
                    print(f"[DISPLAY] Affichage indisponible, passage en headless : {error}")
                    display_enabled = False
            else:
                print(
                    f"\r[LIVE] CONFIRMEE={occupancy.occupancy_confirmed} "
                    f"OBSERVEE={occupancy.occupancy_observed} "
                    f"INCERTAINE={occupancy.occupancy_uncertain} "
                    f"IN={occupancy.total_in} OUT={occupancy.total_out} "
                    f"NEW={occupancy.total_new} PHASE={occupancy.phase}",
                    end="",
                    flush=True,
                )

            if args.max_frames is not None and frame_index >= args.max_frames:
                print(f"\n[CONTROL] Arrêt après {frame_index} frames (--max-frames)")
                break

    except KeyboardInterrupt:
        print("\n[CONTROL] Interruption clavier : arrêt propre")
    finally:
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()
        _finalize(
            logger=logger,
            occupancy=occupancy,
            profiler=profiler,
            fps_estimator=fps_estimator,
            session_root=session_root,
            session_id=session_id,
            config=config,
            resolved_config_path=resolved_config_path,
            model_path=model_path,
            model_sha256=model_sha256,
            frames_processed=frame_index,
            duration_s=time.perf_counter() - start_s,
            budget=budget,
            diagnostics=diagnostics,
        )
        logger.close()

    return exit_code


def _purge_stabilizers(locker, stabilizer, detections: Sequence[Detection]) -> None:
    active = {int(detection.technical_track_id) for detection in detections}
    if locker is not None:
        locker.purge_lost_tracks(active)
    if stabilizer is not None:
        stabilizer.purge_lost_tracks(active)


def _open_writer(
    path: Path, config: PipelineConfig, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = config.output.annotated_video_codec
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo de sortie : {path}")
    return writer


def _finalize(
    *,
    logger: EventLogger,
    occupancy: OccupancyManager,
    profiler: LatencyProfiler,
    fps_estimator: FpsEstimator,
    session_root: Path,
    session_id: str,
    config: PipelineConfig,
    resolved_config_path: Path,
    model_path: Path,
    model_sha256: str,
    frames_processed: int,
    duration_s: float,
    budget: FrameBudget | None,
    diagnostics: DetectionDiagnostics | None = None,
) -> None:
    """Écrit le bilan de session : événement final, résumé, environnement (spec 6.5)."""
    latency = profiler.summary()
    snapshot = occupancy.snapshot()
    fps_mean = fps_estimator.fps
    logger.emit(
        "SESSION_END",
        duration_s,
        frames_processed,
        frames_processed=frames_processed,
        duration_s=round(duration_s, 3),
        fps_mean=round(fps_mean, 3) if fps_mean else 0.0,
        fps_median=round(fps_estimator.fps_median or 0.0, 3),
        fps_min=round(fps_estimator.fps_min or 0.0, 3),
        occupancy_confirmed=snapshot.confirmed,
        occupancy_observed=snapshot.observed,
        occupancy_uncertain=snapshot.uncertain,
        occupancy_operational=snapshot.operational,
        total_in=snapshot.total_in,
        total_out=snapshot.total_out,
        total_new=snapshot.total_new,
        latency_ms=latency,
    )
    summary = {
        "session_id": session_id,
        "config_path": str(config.config_path),
        "resolved_config_path": str(resolved_config_path),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "dependencies": dependency_versions(),
        "frames_processed": frames_processed,
        "duration_s": round(duration_s, 3),
        "fps": {
            "mean": round(fps_mean, 3) if fps_mean else None,
            "median": round(fps_estimator.fps_median, 3) if fps_estimator.fps_median else None,
            "min": round(fps_estimator.fps_min, 3) if fps_estimator.fps_min else None,
            "source": config.timing.fps_source,
            "measured_samples": fps_estimator.samples,
        },
        "frame_budget": None if budget is None else budget.to_dict(),
        "bootstrap_frames": occupancy.bootstrap_frames,
        "counters": {
            "initial_occupancy": snapshot.initial,
            # `occupancy_confirmed` est l'effectif opérationnel (initial + IN +
            # NEW - OUT) : il ne diminue que sur une sortie confirmée.
            "occupancy_confirmed": snapshot.confirmed,
            "occupancy_observed": snapshot.observed,
            "occupancy_uncertain": snapshot.uncertain,
            "occupancy_operational": snapshot.operational,
            "occupancy_range": [snapshot.range_low, snapshot.range_high],
            "total_in": snapshot.total_in,
            "total_out": snapshot.total_out,
            "total_new": snapshot.total_new,
        },
        "diagnostics": None if diagnostics is None else diagnostics.summary(),
        "latency_ms": latency,
        "events_written": logger.event_count,
    }
    write_json(session_root / "summary.json", summary)
    write_json(session_root / "environment.json", dependency_versions())

    print(
        f"\n[BILAN] OCCUPATION_CONFIRMEE={snapshot.confirmed} "
        f"OBSERVEE={snapshot.observed} INCERTAINE={snapshot.uncertain} "
        f"RANGE=[{snapshot.range_low}, {snapshot.range_high}] "
        f"IN={snapshot.total_in} OUT={snapshot.total_out} NEW={snapshot.total_new}"
    )
    if latency:
        worst = sorted(latency.items(), key=lambda item: -item[1]["p95_ms"])
        print("[LATENCE P95] " + ", ".join(f"{stage}={data['p95_ms']:.1f}ms" for stage, data in worst))
    print(f"[RESULTATS] {session_root}")


if __name__ == "__main__":
    raise SystemExit(main())
