#!/usr/bin/env python
"""Diagnostic : la résolution séquentielle de galerie est-elle mise en défaut ?

Pourquoi ce script
------------------

À chaque frame, :meth:`identity_manager.IdentityManager.assign` traite les
observations **une par une**. Une observation sans identifiant technique connu
passe par la galerie (:meth:`_plausible_candidates`) et, si deux candidats sont
trop proches, bascule sur ``provisional_person`` — donc sur une **nouvelle**
identité provisoire.

Une correction « globale » (matrice de coût N×N + affectation optimale) ne peut
apporter quelque chose que s'il existe des frames où **plusieurs observations
sans mapping** cherchent un candidat **en même temps**, avec des candidats
mutuellement proches. Ce script mesure exactement cela, sur une vidéo réelle et
avec le chemin de code de production :

- nombre d'observations routées vers la galerie par frame ;
- nombre de ces observations ayant au moins un candidat plausible ;
- frames à ``>= 2`` observations concurrentes ;
- frames où les ensembles de candidats **se recouvrent** (c'est là, et seulement
  là, qu'une affectation jointe peut produire un résultat différent de la
  résolution séquentielle).

Il n'invente rien : il observe ``_plausible_candidates`` via un wrapper, sans
modifier le comportement du pipeline.

Usage
-----

    python scripts/diagnose_joint_assignment.py --source test/fort_occ4.mp4 --max-frames 300

Sortie : un rapport JSON (``--out``) + un résumé lisible sur la sortie standard.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config, override_config  # noqa: E402
from geometry import compute_anchor  # noqa: E402
from identity_manager import IdentityManager, Observation, _distance  # noqa: E402


def rejection_breakdown(manager, observation, timestamp_s) -> dict[str, int]:
    """Pourquoi les identités de galerie ont-elles été écartées pour cette observation ?

    Réplique **de lecture seule** des conditions de
    :meth:`IdentityManager._plausible_candidates` (mêmes seuils, lus dans la
    configuration) : elle n'est utilisée que pour classer un échec de
    réassociation, jamais pour décider. Une observation sans candidat alors que
    des identités existent pointe la cause réelle : fenêtre de galerie dépassée
    ou contrainte de déplacement trop serrée.
    """
    breakdown = {
        "existing_identities": 0,
        "outside_gallery_window": 0,
        "degenerate_height": 0,
        "too_far": 0,
        "plausible": 0,
    }
    window = manager.grace_period_seconds + manager.purge_retention_seconds
    for record in manager.records.values():
        breakdown["existing_identities"] += 1
        if not manager.within_gallery_window(record, timestamp_s):
            breakdown["outside_gallery_window"] += 1
            continue
        if record.bbox_height <= 0:
            breakdown["degenerate_height"] += 1
            continue
        elapsed = timestamp_s - record.last_seen_s
        allowed = (
            manager.long_term.v_max_ratio
            * record.bbox_height
            * min(elapsed, window)
            + manager.long_term.spatial_margin_ratio * record.bbox_height
        )
        if _distance(observation.anchor, record.anchor) > allowed:
            breakdown["too_far"] += 1
        else:
            breakdown["plausible"] += 1
    return breakdown


def measure(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    config = load_config(args.config)
    if args.swap_correction is not None:
        config = override_config(
            config,
            {"reid": {"swap_correction": {"enabled": bool(args.swap_correction)}}},
        )

    identities = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        appearance_continuity=config.reid.appearance_continuity,
        swap_correction=config.reid.swap_correction,
        grace_period_seconds=config.timing.grace_period_seconds,
    )

    # -- Instrumentation non intrusive de ``assign`` ------------------------
    # Mesurer la concurrence réelle impose de **ne pas** hériter du ``_claimed``
    # séquentiel : sinon la seconde observation se voit refuser le candidat déjà
    # réclamé par la première, et le compteur retombe à tort sur « 0 observation
    # concurrente » — exactement l'angle mort qu'on cherche à quantifier.
    # On calcule donc les candidats de *toutes* les observations sans mapping
    # contre le même instantané (toutes les identités réputées non-vivantes,
    # ``_claimed`` vide), puis on restaure l'état avant de laisser ``assign``
    # travailler normalement.
    original_assign = IdentityManager.assign
    original_plausible = IdentityManager._plausible_candidates
    holder: dict[str, int] = {"frame": 0}
    calls: list[dict] = []

    def instrumented_assign(self, observations, frame, timestamp_s, frame_index):  # type: ignore[no-untyped-def]
        holder["frame"] = int(frame_index)
        unmapped = [
            observation
            for observation in observations
            if int(observation.technical_track_id) not in self.technical_to_person
        ]
        if unmapped:
            live_backup = {person: rec.live for person, rec in self.records.items()}
            claimed_backup = set(self._claimed)
            for rec in self.records.values():
                rec.live = False
            self._claimed = set()
            try:
                for observation in unmapped:
                    candidates = original_plausible(self, observation, timestamp_s)
                    rejected = (
                        rejection_breakdown(self, observation, timestamp_s)
                        if not candidates
                        else {}
                    )
                    calls.append(
                        {
                            "frame": int(frame_index),
                            "technical_track_id": int(observation.technical_track_id),
                            "candidates": [int(person) for person, *_rest in candidates],
                            "rejections": rejected,
                        }
                    )
            finally:
                for person, live in live_backup.items():
                    if person in self.records:
                        self.records[person].live = live
                self._claimed = claimed_backup
        return original_assign(self, observations, frame, timestamp_s, frame_index)

    IdentityManager.assign = instrumented_assign  # type: ignore[assignment]
    try:
        model = YOLO(str(config.resolve_path(config.model.path)))
        generator = model.track(
            source=args.source,
            tracker=str(REPO_ROOT / config.tracker.config_path),
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
        observations_routed = 0
        for result in generator:
            frame = getattr(result, "orig_img", None)
            if frame is None:
                continue
            frames_processed += 1
            holder["frame"] = frames_processed
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
            observations_routed += len(observations)
            identities.assign(observations, frame, timestamp_s, frames_processed)

            if args.max_frames is not None and frames_processed >= args.max_frames:
                break
    finally:
        IdentityManager.assign = original_assign  # type: ignore[assignment]

    # -- Agrégation par frame ----------------------------------------------
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for call in calls:
        by_frame[call["frame"]].append(call)

    frames_with_any_candidate = 0
    frames_with_two_concurrent = 0
    frames_with_overlap = 0
    frames_joint_would_differ = 0
    concurrent_examples: list[dict] = []
    distinct_sizes: Counter[int] = Counter()

    for frame_index, group in sorted(by_frame.items()):
        with_candidate = [call for call in group if call["candidates"]]
        if with_candidate:
            frames_with_any_candidate += 1
        if len(with_candidate) >= 2:
            frames_with_two_concurrent += 1
            sets = [set(call["candidates"]) for call in with_candidate]
            union: set[int] = set().union(*sets)
            intersection = set.intersection(*sets) if sets else set()
            if intersection:
                frames_with_overlap += 1
            # L'affectation jointe ne peut différer de la séquentielle que si un
            # candidat est revendiqué par au moins deux observations : c'est la
            # seule situation où l'ordre de traitement décide du résultat.
            claims = Counter()
            for values in sets:
                for person in values:
                    claims[person] += 1
            contested = {person for person, count in claims.items() if count >= 2}
            if contested:
                frames_joint_would_differ += 1
            if len(concurrent_examples) < args.examples:
                concurrent_examples.append(
                    {
                        "frame": frame_index,
                        "observations": [
                            {
                                "technical_track_id": call["technical_track_id"],
                                "candidates": call["candidates"],
                            }
                            for call in with_candidate
                        ],
                        "candidates_union": sorted(union),
                        "candidates_intersection": sorted(intersection),
                        "contested_candidates": sorted(contested),
                    }
                )
        distinct_sizes[len(group)] = distinct_sizes[len(group)] + 1

    # Cause réelle des appels sans candidat : fenêtre de galerie dépassée ou
    # contrainte de déplacement trop serrée (le reste est bruit de fond).
    rejection_totals: Counter[str] = Counter()
    calls_without_candidate = [call for call in calls if not call["candidates"]]
    calls_without_candidate_outside_warmup = [
        call for call in calls_without_candidate if call["frame"] > args.warmup_frames
    ]
    for call in calls_without_candidate:
        for reason, count in (call.get("rejections") or {}).items():
            if reason in ("existing_identities", "plausible"):
                continue
            rejection_totals[reason] += int(count)

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(args.source),
        "frames_processed": frames_processed,
        "observations_routed_to_gallery": observations_routed,
        "gallery_candidate_calls": len(calls),
        "gallery_calls_without_candidate": len(calls_without_candidate),
        "gallery_calls_without_candidate_outside_warmup": len(
            calls_without_candidate_outside_warmup
        ),
        "rejection_reasons_totals": dict(sorted(rejection_totals.items())),
        "frames_with_any_gallery_candidate": frames_with_any_candidate,
        "frames_with_two_concurrent_gallery_observations": frames_with_two_concurrent,
        "frames_with_at_least_one_common_candidate": frames_with_overlap,
        "frames_with_contested_candidate": frames_joint_would_differ,
        "observations_routed_per_frame": dict(sorted(distinct_sizes.items())),
        "concurrent_examples": concurrent_examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Chemin de la vidéo mesurée")
    parser.add_argument("--config", default=None, help="YAML de configuration (défaut : config/pipeline.yaml)")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames traitées (défaut : 300)")
    parser.add_argument("--assumed-fps", type=float, default=30.0, help="FPS supposé pour les horodatages")
    parser.add_argument("--examples", type=int, default=10, help="Nombre d'exemples de frames concurrentes conservés")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames d'amorçage exclues du décompte des échecs de réassociation",
    )
    parser.add_argument(
        "--swap-correction",
        dest="swap_correction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force l'état de la correction de swap (défaut : valeur de configuration)",
    )
    parser.add_argument("--out", default=None, help="JSON de sortie (défaut : results/diagnose_joint_assignment.json)")
    args = parser.parse_args(argv)

    report = measure(args)
    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "results" / "diagnose_joint_assignment.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({key: value for key, value in report.items() if key != "concurrent_examples"}, indent=2, ensure_ascii=False))
    for example in report["concurrent_examples"]:
        print(f"  frame {example['frame']}: {example['observations']}")
    print(f"\n[DIAGNOSTIC] rapport écrit dans {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
