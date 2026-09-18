#!/usr/bin/env python
"""Diagnostic d'occultation : « une box existait-elle avant la redétection ? »

Pourquoi ce script
------------------

Deux symptômes observés — une personne occultée longtemps qui n'est jamais
redétectée, et une personne « détectée beaucoup plus tard » pendant le warm-up —
peuvent avoir deux causes **opposées** :

- **limite de détection** : aucune boîte YOLO n'existe dans la zone, même sous le
  seuil courant. Toucher à ``IdentityManager``, à la rétention de galerie ou au
  warm-up ne changerait rien : ces modules fonctionnent dès qu'une détection
  existe ;
- **bug de logique** : une boîte existe mais n'est pas rattachée au bon
  ``person_id`` (ou le comptage l'ignore). C'est un défaut de rattachement, à
  corriger dans le code.

Les journaux ``events.jsonl`` ne suffisent pas à trancher : ils disent qu'une
frame **quelque part** contenait des boîtes (``TRACK_ID_ABSENT`` /
``DETECTION_ABSENT``), jamais **où**. Ce script rejoue la vidéo et répond
position par position, en réutilisant le chemin de code réel (YOLO + BoT-SORT +
:class:`identity_manager.IdentityManager`) — sans ligne virtuelle, donc sans la
sélection manuelle qui interdirait toute mesure automatisée.

Sortie
------

Un rapport JSON par épisode d'absence, au format demandé :

    Cas observé : person_id=P2, absence de la frame 660 à 760 (jamais redétecté)
    Box détectée avant redétection finale ? OUI / NON
    Si OUI : frame de première apparition, identifiants techniques concernés,
             person_id obtenu
    Conclusion : bug de logique (IdentityManager/warm-up) OU limite de détection

Usage
-----

    python scripts/diagnose_occlusion.py --source test/video2.avi --max-frames 760
    python scripts/diagnose_occlusion.py --source test/video1.mp4 --min-absence-frames 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402
from geometry import compute_anchor  # noqa: E402
from identity_manager import IdentityManager, Observation  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection sur union de deux boîtes ``xyxy``."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def overlaps_area(candidate: np.ndarray, reference: np.ndarray, margin_ratio: float) -> bool:
    """La boîte candidate occupe-t-elle la zone laissée par la personne ?

    Deux critères complémentaires, parce qu'une personne assise ou partiellement
    occultée produit une boîte plus petite que la boîte de référence :

    - recouvrement (IoU) non nul avec la boîte de référence ;
    - ancre (bas-centre) de la candidate **dans** la boîte de référence dilatée
      d'une fraction de sa hauteur.
    """
    if iou(candidate, reference) > 1e-6:
        return True
    height = float(reference[3] - reference[1])
    margin = margin_ratio * height
    x1, y1, x2, y2 = (
        reference[0] - margin,
        reference[1] - margin,
        reference[2] + margin,
        reference[3] + margin,
    )
    anchor_x = float((candidate[0] + candidate[2]) / 2.0)
    anchor_y = float(candidate[3])
    return bool(x1 <= anchor_x <= x2 and y1 <= anchor_y <= y2)


@dataclass
class AbsenceEpisode:
    person_id: int
    last_bbox: np.ndarray
    last_seen_frame: int
    last_seen_bbox_technical_id: int
    #: Frame où la personne est réobservée (``None`` = jamais réobservée dans la
    #: fenêtre mesurée).
    end_frame: int | None = None
    #: Dernière frame de la fenêtre mesurée, pour donner une durée à un épisode
    #: qui ne se referme pas (sinon « absence infinie » n'est pas comparable).
    window_end_frame: int = 0
    box_frames: list[int] = field(default_factory=list)
    box_technical_ids: list[int] = field(default_factory=list)
    box_person_ids: list[int] = field(default_factory=list)
    reassociated: bool = False

    @property
    def absence_frames(self) -> int:
        end = self.end_frame if self.end_frame is not None else self.window_end_frame
        return int(end) - int(self.last_seen_frame)

    def as_report(self) -> dict:
        box_seen = bool(self.box_frames)
        return {
            "person_id": self.person_id,
            "absence_start_frame": self.last_seen_frame,
            "absence_end_frame": self.end_frame,
            "absence_observed_frames": self.absence_frames,
            "never_redetected": self.end_frame is None,
            "reassociated": self.reassociated,
            "box_detected_before_final_redetection": "OUI" if box_seen else "NON",
            "first_box_frame": self.box_frames[0] if box_seen else None,
            "box_count_in_area": len(self.box_frames),
            "technical_track_ids_in_area": sorted(set(self.box_technical_ids)),
            "person_ids_attributed_in_area": sorted(set(self.box_person_ids)),
            "conclusion": (
                "bug de logique (rattachement IdentityManager / warm-up)"
                if box_seen
                else "limite de detection (aucune boite YOLO dans la zone)"
            ),
        }


def diagnose(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    config = load_config(args.config)
    identities = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        appearance_continuity=config.reid.appearance_continuity,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    model = YOLO(str(config.resolve_path(config.model.path)))
    generator = model.track(
        source=args.source,
        tracker=str(config.resolve_path(config.tracker.config_path)),
        persist=config.tracker.persist,
        classes=list(config.model.classes),
        conf=config.model.confidence,
        iou=config.model.nms_iou,
        imgsz=config.model.image_size,
        show=False,
        stream=True,
        verbose=False,
    )

    episodes: dict[int, AbsenceEpisode] = {}
    finalized: list[AbsenceEpisode] = []
    last_bbox: dict[int, np.ndarray] = {}
    frame_index = 0
    for result in generator:
        frame = getattr(result, "orig_img", None)
        if frame is None:
            continue
        frame_index += 1
        timestamp_s = frame_index / float(args.assumed_fps)

        raw_boxes: list[np.ndarray] = []
        raw_ids: list[int | None] = []
        raw_confs: list[float] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            ids = (
                boxes.id.int().cpu().tolist()
                if getattr(boxes, "id", None) is not None
                else [None] * len(xyxy)
            )
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)
            for box, track_id, confidence in zip(xyxy, ids, confs):
                raw_boxes.append(np.asarray(box, dtype=float))
                raw_ids.append(None if track_id is None else int(track_id))
                raw_confs.append(float(confidence))

        observations = [
            Observation(
                technical_track_id=int(track_id),
                bbox=box,
                confidence=confidence,
                anchor=compute_anchor(box),
                bbox_height=float(box[3] - box[1]),
            )
            for box, track_id, confidence in zip(raw_boxes, raw_ids, raw_confs)
            if track_id is not None
        ]
        assignments = identities.assign(observations, frame, timestamp_s, frame_index)
        observed_person_ids = {a.person_id for a in assignments if a.person_id}

        # Une boîte (n'importe quel identifiant, même absent) dans la zone d'une
        # personne actuellement absente est la preuve recherchée.
        for person_id, episode in episodes.items():
            if person_id in observed_person_ids:
                continue
            for box, track_id in zip(raw_boxes, raw_ids):
                if not overlaps_area(box, episode.last_bbox, args.margin_ratio):
                    continue
                episode.box_frames.append(frame_index)
                if track_id is not None:
                    episode.box_technical_ids.append(track_id)
                    assigned = identities.person_of(track_id)
                    if assigned is not None:
                        episode.box_person_ids.append(assigned)
                break

        # Ouverture d'un épisode : la personne vient d'être perdue.
        for person_id, bbox in list(last_bbox.items()):
            if person_id in observed_person_ids or person_id in episodes:
                continue
            record = identities.record(person_id)
            if record is None:
                continue
            episodes[person_id] = AbsenceEpisode(
                person_id=person_id,
                last_bbox=bbox,
                last_seen_frame=frame_index,
                last_seen_bbox_technical_id=int(record.technical_track_id or -1),
            )

        # Clôture : la personne est réobservée.
        for person_id in list(episodes):
            if person_id in observed_person_ids:
                episode = episodes.pop(person_id)
                episode.end_frame = frame_index
                episode.reassociated = True
                finalized.append(episode)

        # Mémorise la dernière boîte connue de chaque personne observée.
        for assignment in assignments:
            if assignment.person_id:
                last_bbox[assignment.person_id] = np.asarray(assignment.bbox, dtype=float)

        for person_id in list(last_bbox):
            if identities.record(person_id) is None:
                # Identité libérée de la galerie : plus suivie, plus d'épisode.
                last_bbox.pop(person_id, None)
                episodes.pop(person_id, None)

        if args.max_frames is not None and frame_index >= args.max_frames:
            break

    for episode in episodes.values():
        episode.window_end_frame = frame_index
    finalized.extend(episodes.values())

    cases = [
        episode.as_report()
        for episode in finalized
        if episode.absence_frames >= args.min_absence_frames
    ]
    cases.sort(key=lambda case: -int(case["absence_observed_frames"]))

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(args.source),
        "frames_processed": frame_index,
        "min_absence_frames": int(args.min_absence_frames),
        "overlap_margin_ratio": float(args.margin_ratio),
        "detector": {
            "model": str(config.model.path),
            "confidence": float(config.model.confidence),
            "image_size": int(config.model.image_size),
        },
        "tracker": {
            "config_path": str(config.tracker.config_path),
            "track_buffer": int(config.tracker.track_buffer),
            "with_reid": bool(config.tracker.with_reid),
            "new_track_thresh": float(config.tracker.new_track_thresh),
        },
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "with_box_in_area": sum(
                1 for case in cases if case["box_detected_before_final_redetection"] == "OUI"
            ),
            "detection_limit": sum(
                1 for case in cases if case["box_detected_before_final_redetection"] == "NON"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Vidéo diagnostiquée")
    parser.add_argument("--config", default=None, help="YAML de configuration (défaut : config/pipeline.yaml)")
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--assumed-fps", type=float, default=30.0)
    parser.add_argument(
        "--min-absence-frames", type=int, default=5,
        help="Absence minimale (frames) pour qu'un épisode soit rapporté",
    )
    parser.add_argument(
        "--margin-ratio", type=float, default=0.25,
        help="Dilatation de la boîte de référence (fraction de sa hauteur)",
    )
    parser.add_argument("--out", default=None, help="Rapport JSON (défaut : results/diagnostic_occlusion.json)")
    args = parser.parse_args(argv)

    report = diagnose(args)
    out_path = (
        Path(args.out) if args.out else REPO_ROOT / "results" / "diagnostic_occlusion.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Source : {report['source']}  ({report['frames_processed']} frames)")
    print(
        f"Cas d'absence >= {report['min_absence_frames']} frames : "
        f"{report['summary']['cases']} "
        f"(box dans la zone : {report['summary']['with_box_in_area']}, "
        f"limite de détection : {report['summary']['detection_limit']})"
    )
    for case in report["cases"]:
        print(
            f"  P{case['person_id']} : absence {case['absence_start_frame']}"
            f" -> {case['absence_end_frame']} "
            f"({case['absence_observed_frames']} frames) | box avant redétection : "
            f"{case['box_detected_before_final_redetection']}"
            + (
                f" (1re box frame {case['first_box_frame']}, "
                f"tids {case['technical_track_ids_in_area']}, "
                f"person_ids {case['person_ids_attributed_in_area']})"
                if case["first_box_frame"] is not None
                else ""
            )
            + f" | {case['conclusion']}"
        )
    print(f"\n[RAPPORT] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
