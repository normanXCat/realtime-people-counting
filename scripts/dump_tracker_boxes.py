"""Dump per-frame BoT-SORT boxes (technical_track_id + anchor) to JSONL.

Usage:
    python scripts/dump_tracker_boxes.py --video test/video2.avi \\
        --start 600 --end 760 --out results/video2_frames_dump.jsonl

Used for the Part B diagnostic: it lets us inspect, frame by frame, whether a
box (any ``technical_track_id``) was present where an occluded person stood,
and how far that box's anchor was from the absent person's last known anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO  # noqa: E402

from config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="test/video2.avi")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10_000)
    parser.add_argument("--out", default="results/video2_frames_dump.jsonl")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    model_path = args.model or cfg.model.path
    tracker_path = cfg.tracker.config_path
    model = YOLO(model_path)

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator = model.track(
        source=str(ROOT / args.video) if not Path(args.video).is_absolute() else args.video,
        tracker=str(tracker_path),
        persist=True,
        classes=[0],
        conf=cfg.model.confidence,
        iou=cfg.model.nms_iou,
        imgsz=cfg.model.image_size,
        show=False,
        stream=True,
        verbose=False,
    )

    with out_path.open("w", encoding="utf-8") as handle:
        for index, result in enumerate(generator, start=1):
            if index > args.end:
                break
            if index < args.start:
                continue
            boxes = getattr(result, "boxes", None)
            rows = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                ids = (
                    boxes.id.int().cpu().tolist()
                    if getattr(boxes, "id", None) is not None
                    else [None] * len(xyxy)
                )
                confs = (
                    boxes.conf.cpu().numpy()
                    if getattr(boxes, "conf", None) is not None
                    else [1.0] * len(xyxy)
                )
                for bbox, tid, conf in zip(xyxy, ids, confs):
                    rows.append(
                        {
                            "tid": tid,
                            "conf": round(float(conf), 3),
                            "bbox": [round(float(v), 1) for v in bbox],
                            "anchor": [
                                round(float((bbox[0] + bbox[2]) / 2), 1),
                                round(float(bbox[3]), 1),
                            ],
                        }
                    )
            handle.write(json.dumps({"frame": index, "boxes": rows}) + "\n")
            handle.flush()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
