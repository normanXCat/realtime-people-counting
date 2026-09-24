"""Lanceur de réévaluation instrumentée sur test/fort_occ4.mp4 avec la calibration diagF_base_code240."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import main as entry
from calibration import ValidatedLine


def main() -> int:
    validated = ValidatedLine(
        p1=(0.191667, 0.971993),
        p2=(0.357407, 0.874794),
        inside_side="negative",
        frame_width=1280,
        frame_height=720,
        display_scale=0.84375,
        clicks_px=((207.0, 590.0), (386.0, 531.0)),
        confirmed_at="2026-09-23T07:26:57+00:00",
    )

    def fake_selection(config, source, **kwargs):
        return validated

    entry.perform_line_selection = fake_selection
    session_id = sys.argv[1] if len(sys.argv) > 1 else "lot_p1_p2_p3"
    max_frames = sys.argv[2] if len(sys.argv) > 2 else "1100"
    return entry.main([
        "--source", "test/fort_occ4.mp4",
        "--max-frames", str(max_frames),
        "--no-show",
        "--output-root", "results",
        "--session-id", session_id,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
