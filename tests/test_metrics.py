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
