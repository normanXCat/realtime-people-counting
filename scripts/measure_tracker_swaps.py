#!/usr/bin/env python
"""Mesure quantitative des **inversements d'identifiant** (swap) par BoT-SORT.

Pourquoi ce script
------------------

Un swap n'est pas une perte de piste : deux identifiants techniques restent
actifs en continu mais s'échangent la personne suivie. Il se produit à
l'intérieur du matching de BoT-SORT, donc aucun ``TECHNICAL_ID_CHANGED`` ni
``TRACK_ID_ABSENT`` ne le signale. Le comparer « à l'œil » sur une vidéo est
impossible à trancher ; ce script le **compte**.

Il exécute le **même** chemin de code que le pipeline (YOLO + BoT-SORT +
:class:`identity_manager.IdentityManager`) sur une vidéo réelle, sans ligne
virtuelle — le comptage de franchissements n'a aucune incidence sur la
stabilité des identités, et la sélection manuelle de ligne empêcherait toute
mesure automatisée.

Sortie : une ligne par configuration mesurée, au format JSON Lines, avec :
``technical_ids`` créés, ``appearance_discontinuities`` (garde-fou de swap),
``technical_id_changes`` (pertes d'identifiant classiques) et les seuils
appliqués. C'est la comparaison **avant/après réglage** demandée : deux lignes du
même tableau, pas une impression.

Usage
-----

    python scripts/measure_tracker_swaps.py --source test/5121204_School_Classroom_1280x720.mp4 \
        --max-frames 300 --track-buffer 150
    python scripts/measure_tracker_swaps.py --source ... --max-frames 300 \
        --track-buffer 75 --appearance-thresh 0.15 --proximity-thresh 0.3

Les valeurs de BoT-SORT sont écrites dans un YAML temporaire
(``results/tracker_variants/``) : la configuration de production n'est jamais
modifiée par une mesure.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, override_config  # noqa: E402
from geometry import compute_anchor  # noqa: E402
from identity_manager import IdentityManager, Observation  # noqa: E402


def build_tracker_variant(
    config,
    *,
    track_buffer: int,
    appearance_thresh: float,
    proximity_thresh: float,
    match_thresh: float,
    new_track_thresh: float,
) -> Path:
    """Écrit un YAML BoT-SORT dérivé de la configuration de production.

    Une mesure ne doit jamais réécrire ``src/configs/custom_botsort.yaml`` : la
    variante est écrite dans ``results/tracker_variants/`` et le fichier de
    référence reste la source de vérité. Les cinq seuils sont écrits ensemble
    pour qu'un seuil isolé (``--match-thresh`` seul, par exemple) ne soit pas
    silencieusement écrasé par la valeur de production des autres.
    """
    base = yaml.safe_load(
        (REPO_ROOT / config.tracker.config_path).read_text(encoding="utf-8")
    )
    base["track_buffer"] = int(track_buffer)
    base["appearance_thresh"] = float(appearance_thresh)
    base["proximity_thresh"] = float(proximity_thresh)
    base["match_thresh"] = float(match_thresh)
    base["new_track_thresh"] = float(new_track_thresh)
    out_dir = REPO_ROOT / config.output.session_root / "tracker_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"botsort_b{int(track_buffer)}_a{float(appearance_thresh):.2f}"
        f"_p{float(proximity_thresh):.2f}_m{float(match_thresh):.2f}"
        f"_n{float(new_track_thresh):.2f}.yaml"
    )
    path.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return path


def measure(args: argparse.Namespace) -> dict:
    """Exécute le suivi sur ``args.max_frames`` frames et retourne les compteurs."""
    from ultralytics import YOLO

    config = load_config(args.config)
    # Chaque seuil est surchargeable ISOLÉMENT : c'est ce qui permet de mesurer
    # l'effet propre d'un seuil au lieu du paquet des trois (une mesure groupée
    # ne peut pas distinguer un gain d'une régression compensée).
    if (
        args.track_buffer is not None
        or args.appearance_thresh is not None
        or args.proximity_thresh is not None
        or args.match_thresh is not None
        or args.new_track_thresh is not None
    ):
        config = override_config(
            config,
            {
                "tracker": {
                    "track_buffer": args.track_buffer or config.tracker.track_buffer,
                    "appearance_thresh": (
                        args.appearance_thresh
                        if args.appearance_thresh is not None
                        else config.tracker.appearance_thresh
                    ),
                    "proximity_thresh": (
                        args.proximity_thresh
                        if args.proximity_thresh is not None
                        else config.tracker.proximity_thresh
                    ),
                    "match_thresh": (
                        args.match_thresh
                        if args.match_thresh is not None
                        else config.tracker.match_thresh
                    ),
                    "new_track_thresh": (
                        args.new_track_thresh
                        if args.new_track_thresh is not None
                        else config.tracker.new_track_thresh
                    ),
                }
            },
        )
    if args.continuity_threshold is not None:
        config = override_config(
            config,
            {
                "reid": {
                    "appearance_continuity": {
                        "similarity_threshold": args.continuity_threshold
                    }
                }
            },
        )
    if args.swap_correction is not None:
        config = override_config(
            config,
            {
                "reid": {
                    "swap_correction": {
                        "enabled": bool(args.swap_correction)
                    }
                }
            },
        )
    tracker_path = build_tracker_variant(
        config,
        track_buffer=config.tracker.track_buffer,
        appearance_thresh=config.tracker.appearance_thresh,
        proximity_thresh=config.tracker.proximity_thresh,
        match_thresh=config.tracker.match_thresh,
        new_track_thresh=config.tracker.new_track_thresh,
    )

    identities = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        appearance_continuity=config.reid.appearance_continuity,
        swap_correction=config.reid.swap_correction,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    model = YOLO(str(config.resolve_path(config.model.path)))
    generator = model.track(
        source=args.source,
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

    frames_processed = 0
    detections_observed = 0
    for result in generator:
        frame = getattr(result, "orig_img", None)
        if frame is None:
            continue
        frames_processed += 1
        timestamp_s = frames_processed / float(args.assumed_fps)
        boxes = getattr(result, "boxes", None)
        observations: list[Observation] = []
        if boxes is not None and getattr(boxes, "id", None) is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(ids)
            for box, track_id, confidence in zip(xyxy, ids, confs):
                bbox = np.asarray(box, dtype=float)
                observations.append(
                    Observation(
                        technical_track_id=int(track_id),
                        bbox=bbox,
                        confidence=float(confidence),
                        anchor=compute_anchor(bbox),
                        bbox_height=float(bbox[3] - bbox[1]),
                    )
                )
                detections_observed += 1
        identities.assign(observations, frame, timestamp_s, frames_processed)

        if args.max_frames is not None and frames_processed >= args.max_frames:
            break

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(args.source),
        "frames_processed": frames_processed,
        "detections_observed": detections_observed,
        "assumed_fps_for_time": float(args.assumed_fps),
        "tracker": {
            "track_buffer": int(config.tracker.track_buffer),
            "with_reid": bool(config.tracker.with_reid),
            "match_thresh": float(config.tracker.match_thresh),
            "proximity_thresh": float(config.tracker.proximity_thresh),
            "appearance_thresh": float(config.tracker.appearance_thresh),
            "new_track_thresh": float(config.tracker.new_track_thresh),
        },
        "appearance_continuity": {
            "similarity_threshold": float(
                config.reid.appearance_continuity.similarity_threshold
            ),
            "max_gap_frames": int(config.reid.appearance_continuity.max_gap_frames),
            "discontinuities": int(identities.appearance_discontinuities),
            # Distribution mesurée : c'est elle qui justifie le seuil retenu.
            "similarity_distribution": identities.continuity_similarity_summary(),
        },
        "swap_correction": {
            "enabled": bool(config.reid.swap_correction.enabled),
            "margin": float(config.reid.swap_correction.margin),
            "swap_corrections_applied": int(identities.swap_corrections_applied),
        },
        "identity": {
            "technical_ids": len({*identities.technical_to_person}),
            "persons_created": int(identities._next_person_id - 1),
            "technical_id_changes": int(identities.technical_id_changes),
            "descriptor_rejections": int(identities.descriptor_rejections),
            "swap_corrections_applied": int(identities.swap_corrections_applied),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Chemin de la vidéo mesurée")
    parser.add_argument("--config", default=None, help="YAML de configuration (défaut : config/pipeline.yaml)")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames traitées (défaut : 300)")
    parser.add_argument("--assumed-fps", type=float, default=30.0, help="FPS supposé pour les horodatages du diagnostic")
    parser.add_argument("--track-buffer", type=int, default=None)
    parser.add_argument("--appearance-thresh", type=float, default=None)
    parser.add_argument("--proximity-thresh", type=float, default=None)
    parser.add_argument(
        "--match-thresh", type=float, default=None,
        help="Seuil de coût de l'association IoU (1 - IoU), mesuré isolément",
    )
    parser.add_argument(
        "--new-track-thresh", type=float, default=None,
        help="Plancher de création d'une nouvelle piste (mesuré isolément)",
    )
    parser.add_argument(
        "--continuity-threshold", type=float, default=None,
        help="Seuil de similarité du garde-fou de swap (permet de balayer les seuils)",
    )
    parser.add_argument(
        "--swap-correction",
        dest="swap_correction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active ou désactive la correction active de swap (défaut : valeur de configuration)",
    )
    parser.add_argument("--out", default=None, help="JSON Lines de sortie (défaut : results/swap_measurements.jsonl)")
    args = parser.parse_args(argv)

    report = measure(args)
    out_path = Path(args.out) if args.out else REPO_ROOT / "results" / "swap_measurements.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[MESURE] ligne ajoutée à {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
