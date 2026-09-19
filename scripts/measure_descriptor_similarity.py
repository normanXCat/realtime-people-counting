#!/usr/bin/env python
"""Mesure de la séparabilité des descripteurs d'apparence (lot B.4).

Pourquoi ce script
------------------

``reid.long_term.similarity_threshold`` (0,40) décide si une apparence inconnue
est réassociée à une identité de la galerie. Changer le descripteur sans mesurer
ce seuil reviendrait à le fixer à l'aveugle. Ce script fournit les **deux
distributions** qui rendent la décision possible :

- ``intra`` : similarité cosinus entre les descripteurs de la **même** piste
  technique à deux frames consécutives (la personne a peu bougé : le descripteur
  doit rester proche) ;
- ``inter`` : similarité cosinus entre les descripteurs de **deux pistes
  différentes** sur la même frame (deux personnes distinctes : le descripteur
  doit rester éloigné).

Un descripteur exploitable a donc une distribution inter nettement plus basse que
la distribution intra. L'écart entre les deux est ce qui manquait au descripteur
historique (histogramme H×S sur le crop **complet**), qui encodait surtout la
couleur des chaises, des tables et du mur.

Les deux descripteurs sont mesurés sur **exactement** les mêmes crops :

- ``bands`` : descripteur courant (crop érodé, bandes normalisées séparément) ;
- ``legacy`` : descripteur historique reproduit par ``horizontal_crop_ratio=0``,
  ``bands=1`` (un seul histogramme sur le crop complet).

Sortie : une ligne JSON Lines par exécution, plus un tableau lisible sur la
sortie standard, avec une valeur de seuil proposée par descripteur (milieu
géométrique entre le p95 inter et le p5 intra).

Usage
-----

    python scripts/measure_descriptor_similarity.py \
        --source test/5121204_School_Classroom_1280x720.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402
from identity_manager import HistogramAppearanceExtractor  # noqa: E402


def legacy_extractor() -> HistogramAppearanceExtractor:
    """Reproduit le descripteur historique : un seul histogramme, crop complet."""
    return HistogramAppearanceExtractor(
        bins_hue=24,
        bins_saturation=8,
        horizontal_crop_ratio=0.0,
        bands=1,
        band_weights=(1.0,),
    )


def summarise(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0, "min": None, "p5": None, "p25": None, "median": None,
            "p75": None, "p95": None, "max": None, "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "min": float(array.min()),
        "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def histogram(values: list[float], bins: int = 10) -> list[int]:
    """Histogramme texte sur [0, 1] : comparable d'un descripteur à l'autre."""
    if not values:
        return [0] * bins
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=bins, range=(0.0, 1.0))
    return [int(count) for count in counts]


def suggested_threshold(intra: dict, inter: dict) -> float | None:
    """Milieu entre le p95 inter (le pire cas « deux personnes ») et le p5 intra.

    Un seuil sous le p95 inter réassocierait deux personnes distinctes ; un seuil
    au-dessus du p5 intra casserait la continuité d'une même personne. Le milieu
    des deux est le compromis le plus simple à défendre, et il est ``None`` si les
    deux distributions se recouvrent (aucun seuil ne les sépare).
    """
    if intra.get("p5") is None or inter.get("p95") is None:
        return None
    low, high = float(inter["p95"]), float(intra["p5"])
    if high <= low:
        return None
    return round((low + high) / 2.0, 4)


def measure(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    config = load_config(args.config)
    current = HistogramAppearanceExtractor.from_config(config.reid.long_term.descriptor)
    legacy = legacy_extractor()

    distributions: dict[str, dict[str, list[float]]] = {
        "bands": {"intra": [], "inter": []},
        "legacy": {"intra": [], "inter": []},
    }
    previous: dict[str, dict[int, np.ndarray]] = {"bands": {}, "legacy": {}}

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

    frames_processed = 0
    for result in generator:
        frame = getattr(result, "orig_img", None)
        if frame is None:
            continue
        frames_processed += 1
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None or not len(boxes):
            for name in previous:
                previous[name].clear()
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            descriptors: dict[str, dict[int, np.ndarray]] = {"bands": {}, "legacy": {}}
            for box, track_id in zip(xyxy, ids):
                for name, extractor in (("bands", current), ("legacy", legacy)):
                    descriptor = extractor.describe(frame, box)
                    if descriptor is not None:
                        descriptors[name][int(track_id)] = descriptor
            for name, per_track in descriptors.items():
                # Inter : deux pistes distinctes de la MÊME frame.
                for first, second in combinations(sorted(per_track), 2):
                    distributions[name]["inter"].append(
                        float(np.dot(per_track[first], per_track[second]))
                    )
                # Intra : même piste, frames consécutives.
                for track_id, descriptor in per_track.items():
                    before = previous[name].get(track_id)
                    if before is not None and before.shape == descriptor.shape:
                        distributions[name]["intra"].append(float(np.dot(before, descriptor)))
                previous[name] = per_track

        if args.max_frames is not None and frames_processed >= args.max_frames:
            break

    report: dict = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(args.source),
        "frames_processed": frames_processed,
        "descriptor_config": {
            "horizontal_crop_ratio": config.reid.long_term.descriptor.horizontal_crop_ratio,
            "bands": config.reid.long_term.descriptor.bands,
            "band_weights": list(config.reid.long_term.descriptor.band_weights),
            "bins_hue": config.reid.long_term.descriptor.bins_hue,
            "bins_saturation": config.reid.long_term.descriptor.bins_saturation,
        },
        "similarity_threshold_configured": float(config.reid.long_term.similarity_threshold),
        "descriptors": {},
    }
    for name, values in distributions.items():
        intra = summarise(values["intra"])
        inter = summarise(values["inter"])
        report["descriptors"][name] = {
            "intra": intra,
            "inter": inter,
            "histogram_intra": histogram(values["intra"]),
            "histogram_inter": histogram(values["inter"]),
            "suggested_similarity_threshold": suggested_threshold(intra, inter),
        }
    return report


def print_table(report: dict) -> None:
    print(f"\nSource : {report['source']} ({report['frames_processed']} frames)")
    print(f"Seuil configuré : {report['similarity_threshold_configured']}")
    for name, stats in report["descriptors"].items():
        print(f"\n== descripteur « {name} » ==")
        print(f"{'':9} {'n':>6} {'min':>7} {'p5':>7} {'p25':>7} {'med':>7} "
              f"{'p75':>7} {'p95':>7} {'max':>7}")
        for kind in ("intra", "inter"):
            row = stats[kind]
            if not row["samples"]:
                print(f"{kind:9} {'0':>6}")
                continue
            print(
                f"{kind:9} {row['samples']:>6} {row['min']:>7.3f} {row['p5']:>7.3f} "
                f"{row['p25']:>7.3f} {row['median']:>7.3f} {row['p75']:>7.3f} "
                f"{row['p95']:>7.3f} {row['max']:>7.3f}"
            )
        print(f"histogramme intra : {stats['histogram_intra']}")
        print(f"histogramme inter : {stats['histogram_inter']}")
        print(f"seuil proposé     : {stats['suggested_similarity_threshold']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Chemin de la vidéo mesurée")
    parser.add_argument("--config", default=None, help="YAML de configuration")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames traitées")
    parser.add_argument(
        "--out", default=None,
        help="JSON Lines de sortie (défaut : results/descriptor_similarity.jsonl)",
    )
    args = parser.parse_args(argv)

    report = measure(args)
    out_path = (
        Path(args.out) if args.out
        else REPO_ROOT / "results" / "descriptor_similarity.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    print_table(report)
    print(f"\n[MESURE] ligne ajoutée à {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
