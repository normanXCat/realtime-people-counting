"""Tests du harnais d'évaluation (section 9.2 / 9.3).

Les journaux utilisés ici sont produits par les **vraies** briques du pipeline
(`EventLogger`, donc schéma validé), jamais par des dictionnaires bricolés : si
le schéma change, ces tests échouent avant l'évaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import evaluate_system as evaluation  # noqa: E402
from src.events import EventLogger  # noqa: E402


def _write_events(path: Path, events: list[dict]) -> Path:
    """Écrit un journal JSON Lines en passant par le logger officiel."""
    with EventLogger(path, session_id="session-test") as logger:
        for event in events:
            logger.write(event)
    return path


def _crossing(event_type: str, timestamp_s: float, person_id: int = 1) -> dict:
    return {
        "schema_version": 1,
        "event_id": f"e-{event_type}-{timestamp_s}",
        "session_id": "session-test",
        "type": event_type,
        "timestamp_s": timestamp_s,
        "frame_index": int(timestamp_s * 30),
        "person_id": person_id,
        "technical_track_id": person_id,
        "direction": "in" if event_type in ("IN", "NEW") else "out",
    }


def _snapshot(timestamp_s: float, confirmed: int, uncertain: int = 0) -> dict:
    return {
        "schema_version": 1,
        "event_id": f"e-snap-{timestamp_s}",
        "session_id": "session-test",
        "type": "OCCUPANCY_SNAPSHOT",
        "timestamp_s": timestamp_s,
        "frame_index": int(timestamp_s * 30),
        "occupancy_confirmed": confirmed,
        "occupancy_uncertain": uncertain,
        "occupancy_range": [confirmed, confirmed + uncertain],
    }


def _session_end(frames: int = 300, fps: float = 25.0) -> dict:
    return {
        "schema_version": 1,
        "event_id": "e-end",
        "session_id": "session-test",
        "type": "SESSION_END",
        "timestamp_s": 10.0,
        "frame_index": frames,
        "frames_processed": frames,
        "duration_s": 10.0,
        "fps_mean": fps,
        "fps_median": fps,
        "fps_min": fps - 2,
        "latency_ms": {"detection": {"mean_ms": 20.0, "median_ms": 19.0, "p95_ms": 30.0, "max_ms": 40.0}},
    }


def _ground_truth(crossings: list[tuple[float, str]], occupancy: list[tuple[float, float, int]]) -> dict:
    return {
        "schema_version": 1,
        "video": "clip01.mp4",
        "fps": 30.0,
        "crossings": [{"timestamp_s": time, "direction": direction} for time, direction in crossings],
        "occupancy": [
            {"start_s": start, "end_s": end, "count": count} for start, end, count in occupancy
        ],
    }


def test_appariement_comptage_parfait():
    gt = [{"timestamp_s": 1.0, "direction": "in"}, {"timestamp_s": 4.0, "direction": "out"}]
    pred = [
        {"timestamp_s": 1.2, "direction": "in"},
        {"timestamp_s": 4.1, "direction": "out"},
    ]
    result = evaluation.apparier_evenements(gt, pred, tolerance_s=0.5)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["f1"] == pytest.approx(1.0)


def test_appariement_refuse_hors_tolerance_et_mauvaise_direction():
    gt = [{"timestamp_s": 1.0, "direction": "in"}, {"timestamp_s": 9.0, "direction": "out"}]
    pred = [
        {"timestamp_s": 1.1, "direction": "out"},  # bonne fenêtre, mauvaise direction
        {"timestamp_s": 20.0, "direction": "out"},  # hors tolérance
    ]
    result = evaluation.apparier_evenements(gt, pred, tolerance_s=0.5)
    assert result["true_positives"] == 0
    assert result["direction_errors"] == 1
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 2


def test_appariement_glouton_est_deterministe():
    gt = [{"timestamp_s": 1.0, "direction": "in"}, {"timestamp_s": 1.4, "direction": "in"}]
    pred = [{"timestamp_s": 1.2, "direction": "in"}]
    result = evaluation.apparier_evenements(gt, pred, tolerance_s=0.5)
    assert result["true_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["matches"][0]["gt_index"] == 0  # le plus proche temporellement


def test_evaluation_complete_respecte_le_critere(tmp_path: Path):
    events_path = _write_events(
        tmp_path / "events.jsonl",
        [
            _crossing("IN", 1.0),
            _crossing("OUT", 6.0),
            _snapshot(1.5, confirmed=0),
            _snapshot(3.0, confirmed=1),
            _snapshot(7.0, confirmed=0),
            _session_end(),
        ],
    )
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            _ground_truth(
                crossings=[(1.0, "in"), (6.0, "out")],
                occupancy=[(0.0, 2.0, 0), (2.0, 5.0, 1), (5.0, 9.0, 0)],
            )
        ),
        encoding="utf-8",
    )

    report = evaluation.evaluer_comptage_systeme(gt_path, events_path, min_fps=15.0)

    assert report["counting"]["f1"] == pytest.approx(1.0)
    assert report["occupancy"]["mae"] == pytest.approx(0.0)
    assert report["timing"]["fps_mean"] == pytest.approx(25.0)
    assert report["acceptance"]["accepted"] is True


def test_evaluation_echoue_quand_le_seuil_n_est_pas_atteint(tmp_path: Path):
    events_path = _write_events(
        tmp_path / "events.jsonl",
        [_crossing("IN", 1.0), _session_end()],
    )
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(_ground_truth(crossings=[(1.0, "in"), (9.0, "out")], occupancy=[])),
        encoding="utf-8",
    )

    report = evaluation.evaluer_comptage_systeme(gt_path, events_path, min_f1=0.90)
    assert report["counting"]["false_negatives"] == 1
    assert report["counting"]["f1"] == pytest.approx(2 / 3)
    assert report["acceptance"]["accepted"] is False
    assert report["acceptance"]["checks"]["f1"]["passed"] is False


def test_agregation_multi_videos(tmp_path: Path):
    pairs: list[tuple[str, str]] = []
    for index, (crossing_time, fps) in enumerate(((1.0, 25.0), (2.0, 12.0)), start=1):
        events_path = _write_events(
            tmp_path / f"events{index}.jsonl",
            [_crossing("IN", crossing_time), _session_end(fps=fps)],
        )
        gt_path = tmp_path / f"gt{index}.json"
        gt_path.write_text(
            json.dumps(_ground_truth(crossings=[(crossing_time, "in")], occupancy=[])),
            encoding="utf-8",
        )
        pairs.append((str(gt_path), str(events_path)))

    aggregated = evaluation.evaluer_sessions(pairs, min_fps=15.0)
    assert aggregated["sessions"] == 2
    assert aggregated["micro_f1"] == pytest.approx(1.0)
    # La seconde vidéo est sous le débit minimal : l'agrégat doit être refusé.
    assert aggregated["accepted_all"] is False


def test_verite_terrain_invalide_est_refusee(tmp_path: Path):
    events_path = _write_events(tmp_path / "events.jsonl", [_crossing("IN", 1.0)])
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps({"crossings": [{"timestamp_s": 1.0, "direction": "gauche"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direction"):
        evaluation.evaluer_comptage_systeme(gt_path, events_path)


def test_journal_invalide_est_refuse_par_le_schema(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"schema_version": 1, "type": "IN", "timestamp_s": 1.0, "frame_index": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        evaluation.charger_evenements(bad)
