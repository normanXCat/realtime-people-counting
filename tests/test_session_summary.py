"""Résumé de session : extracteur d'apparence actif et FPS natif journalisés.

Ces deux informations répondent à des questions de diagnostic que le reste du
journal ne permet pas de trancher :

- **quel descripteur a réellement tourné ?** ``reid.external_reid.enabled=true``
  ne garantit pas que le modèle profond a été chargé : une dépendance ou des
  poids manquants font retomber sur le descripteur HSV. Le nom de l'extracteur
  actif (``deep_osnet_x0_25``, ``deep_mobilenetv3`` ou le repli
  ``histogram_hsv_fallback``) doit donc apparaître dans ``summary.json`` ;
- **la session a-t-elle été traitée hors temps réel ?** Le FPS **natif** de la
  vidéo est journalisé à côté des mesures ``mean``/``median``/``min`` pour
  repérer immédiatement un débit de traitement divergent.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from config import load_config
from events import EventLogger
from identity_manager import IdentityManager
from main import _finalize
from metrics import FpsEstimator, LatencyProfiler


def _write_summary(tmp_path: Path, *, native_fps: float | None, fps_source: str | None) -> dict:
    config = load_config()
    identities = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    snapshot = SimpleNamespace(
        confirmed=0, observed=0, uncertain=0, operational=0, initial=0,
        range_low=0, range_high=0, total_in=0, total_out=0, total_new=0,
    )
    occupancy = SimpleNamespace(
        identities=identities,
        bootstrap_frames=0,
        snapshot=lambda: snapshot,
    )
    session_root = tmp_path / "session"
    session_root.mkdir()
    logger = EventLogger(session_root / "events.jsonl", session_id="session-summary")
    resolved = session_root / "config_resolved.yaml"
    resolved.write_text("", encoding="utf-8")

    _finalize(
        logger=logger,
        occupancy=occupancy,
        profiler=LatencyProfiler(),
        fps_estimator=FpsEstimator(window=4, native_fps=native_fps),
        session_root=session_root,
        session_id="session-summary",
        config=config,
        resolved_config_path=resolved,
        model_path=Path("models/yolo11n.pt"),
        model_sha256="0" * 64,
        frames_processed=0,
        duration_s=0.0,
        budget=None,
        diagnostics=None,
        native_fps=native_fps,
        fps_source=fps_source,
    )
    logger.close()
    return json.loads((session_root / "summary.json").read_text(encoding="utf-8"))


def test_summary_reports_the_active_appearance_extractor(tmp_path: Path):
    """Le nom de l'extracteur profond réellement actif apparaît dans le résumé."""
    config = load_config()
    assert config.reid.external_reid.enabled is True

    summary = _write_summary(tmp_path, native_fps=None, fps_source=None)

    extractor = summary["appearance_extractor"]
    assert extractor["enabled"] is True
    assert extractor["name"] in {
        "deep_osnet_x0_25",
        "deep_mobilenetv3",
        "histogram_hsv_fallback",
    }


def test_summary_logs_the_native_fps_of_the_source(tmp_path: Path):
    """``fps.native`` est journalisé à côté de mean/median/min, et la source réelle."""
    summary = _write_summary(tmp_path, native_fps=30.0, fps_source="video_native")

    assert summary["fps"]["native"] == 30.0
    assert summary["fps"]["source"] == "video_native"


def test_summary_native_fps_is_none_for_a_camera(tmp_path: Path):
    summary = _write_summary(tmp_path, native_fps=None, fps_source="measured")

    assert summary["fps"]["native"] is None
    assert summary["fps"]["source"] == "measured"
