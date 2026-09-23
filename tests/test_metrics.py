"""Niveau 1 — FPS mesuré et latence instrumentée (spec 5.3 et 6.5)."""

from __future__ import annotations

import pytest

from metrics import FpsEstimator, LatencyProfiler, percentile


def test_fps_requires_two_ticks_before_being_usable():
    estimator = FpsEstimator(window=5)
    assert estimator.fps is None
    assert estimator.is_ready is False
    estimator.tick(0.0)
    assert estimator.fps is None
    with pytest.raises(RuntimeError):
        estimator.require_fps()
    estimator.tick(0.1)
    assert estimator.is_ready is True
    assert estimator.fps == pytest.approx(10.0)


def test_fps_uses_a_rolling_window():
    estimator = FpsEstimator(window=3)
    for index in range(5):
        estimator.tick(index * 0.1)      # 10 fps
    assert estimator.fps == pytest.approx(10.0, abs=1e-6)
    # Une frame lente : seule la fenêtre glissante est affectée.
    estimator.tick(0.9)
    assert estimator.fps < 10.0
    assert estimator.fps_min < estimator.fps


def test_fps_median_and_elapsed():
    estimator = FpsEstimator(window=10)
    for index in range(6):
        estimator.tick(index * 0.2)
    assert estimator.fps == pytest.approx(5.0, abs=1e-6)
    assert estimator.fps_median == pytest.approx(5.0, abs=1e-6)
    assert estimator.elapsed() == pytest.approx(1.0)


def test_fps_ignores_non_positive_deltas():
    estimator = FpsEstimator(window=5)
    estimator.tick(1.0)
    estimator.tick(1.0)
    estimator.tick(1.1)
    assert estimator.samples == 1


def test_window_must_be_at_least_two():
    with pytest.raises(ValueError):
        FpsEstimator(window=1)


# ---------------------------------------------------------------------------
# FPS natif : horloge de contenu vs débit de traitement (run hors temps réel)
# ---------------------------------------------------------------------------
def test_fps_native_divergence_is_signalled_when_processing_is_slower():
    """Un traitement plus lent que le FPS natif est signalé (> 20 % d'écart)."""
    estimator = FpsEstimator(window=5, native_fps=30.0)
    for index in range(4):
        estimator.tick(index * 0.2)   # 5 ips de traitement, contre 30 ips natifs
    assert estimator.fps == pytest.approx(5.0)
    assert estimator.divergence_ratio == pytest.approx(25 / 30, abs=1e-6)
    assert estimator.diverges_from_native is True
    message = estimator.divergence_message()
    assert message is not None and "hors temps réel" in message


def test_fps_native_is_quiet_when_processing_keeps_up():
    estimator = FpsEstimator(window=5, native_fps=10.0)
    for index in range(4):
        estimator.tick(index * 0.1)   # 10 ips : aucun écart
    assert estimator.divergence_ratio == pytest.approx(0.0, abs=1e-9)
    assert estimator.diverges_from_native is False
    assert estimator.divergence_message() is None


def test_fps_native_clock_keeps_grace_and_track_buffer_in_the_same_referential():
    """Le budget FSM est converti avec le FPS **natif**, pas le débit CPU.

    Sans horloge native, ``grace_period_frames`` serait exprimé dans le
    référentiel du débit de traitement alors que ``track_buffer`` reste en
    frames vidéo natives : les deux seuils ne couvriraient plus la même durée de
    contenu.
    """
    from config import load_config

    native_fps = 30.0
    processing_fps = 5.0  # traitement 6× plus lent que le temps réel
    config = load_config()

    native_estimator = FpsEstimator(window=4, native_fps=native_fps)
    for index in range(5):
        native_estimator.tick(index / native_fps)   # timestamp = frame_index / fps_natif
    budget = config.timing.frame_budget(native_estimator.require_fps())
    assert budget.grace_period_frames / native_fps == pytest.approx(
        config.timing.grace_period_seconds
    )
    assert config.tracker.track_buffer / native_fps >= config.timing.grace_period_seconds

    wall_estimator = FpsEstimator(window=4)
    for index in range(5):
        wall_estimator.tick(index / processing_fps)
    wall_budget = config.timing.frame_budget(wall_estimator.require_fps())
    assert wall_budget.grace_period_frames != budget.grace_period_frames


def test_native_video_fps_reads_cap_prop_and_releases(monkeypatch):
    """Le FPS natif est lu sur ``CAP_PROP_FPS``, jamais supposé (règle 0.2)."""
    import cv2

    from metrics import native_video_fps

    class _FakeCapture:
        def __init__(self, source):
            self.source = source
            self.released = False

        def get(self, prop):
            assert prop == cv2.CAP_PROP_FPS
            return 29.97

        def release(self):
            self.released = True

    capture = _FakeCapture("clip.mp4")
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: capture)
    assert native_video_fps("clip.mp4") == pytest.approx(29.97)
    assert capture.released is True


def test_native_video_fps_returns_none_for_camera(monkeypatch):
    """Une caméra (FPS non déclaré) ne doit pas fournir de valeur supposée."""
    import cv2

    from metrics import native_video_fps

    class _NoFps:
        def get(self, _prop):
            return 0.0

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda source: _NoFps())
    assert native_video_fps(0) is None


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == pytest.approx(1.0)
    assert percentile(values, 1.0) == pytest.approx(4.0)
    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile([42.0], 0.95) == pytest.approx(42.0)
    assert percentile([], 0.5) == 0.0
    with pytest.raises(ValueError):
        percentile(values, 1.5)


def test_latency_profiler_reports_every_stage():
    profiler = LatencyProfiler(stages=("capture", "detection", "fsm"))
    for value in (10.0, 20.0, 30.0):
        profiler.record("detection", value)
    profiler.record("fsm", 5.0)
    summary = profiler.summary()
    assert summary["detection"]["mean_ms"] == pytest.approx(20.0)
    assert summary["detection"]["median_ms"] == pytest.approx(20.0)
    assert summary["detection"]["min_ms"] == pytest.approx(10.0)
    assert summary["detection"]["max_ms"] == pytest.approx(30.0)
    assert summary["detection"]["p95_ms"] >= 29.0
    assert summary["detection"]["samples"] == 3
    assert "capture" not in summary, "une étape sans mesure n'est pas inventée"
    assert profiler.total_mean_ms() == pytest.approx(25.0)


def test_latency_profiler_covers_the_six_declared_stages():
    from metrics import LATENCY_STAGES

    assert set(LATENCY_STAGES) == {
        "capture", "detection", "tracking", "reid", "fsm", "render", "write",
    }
    summary = LatencyProfiler().summary()
    assert summary == {}
