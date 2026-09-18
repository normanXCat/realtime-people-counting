"""Tests unitaires pour extract_head_point (géométrie des points-clés de tête)."""

from __future__ import annotations

import numpy as np
import pytest

from geometry import COCO_HEAD_KEYPOINTS, extract_head_point


def test_coco_head_keypoint_indices():
    assert COCO_HEAD_KEYPOINTS["nose"] == 0
    assert COCO_HEAD_KEYPOINTS["left_eye"] == 1
    assert COCO_HEAD_KEYPOINTS["right_eye"] == 2
    assert COCO_HEAD_KEYPOINTS["left_ear"] == 3
    assert COCO_HEAD_KEYPOINTS["right_ear"] == 4


def test_extract_head_point_single_confident_keypoint():
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    # nez confiant à (100, 50)
    xy[0] = [100.0, 50.0]
    conf[0] = 0.9

    pt, min_conf = extract_head_point(xy, conf, ("nose", "left_eye", "right_eye"), min_confidence=0.5)
    assert pt is not None
    assert pt == (100.0, 50.0)
    assert min_conf == pytest.approx(0.9)


def test_extract_head_point_mean_over_multiple_confident_keypoints():
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    # nez à (100, 50) conf 0.8
    xy[0] = [100.0, 50.0]
    conf[0] = 0.8

    # œil gauche à (90, 45) conf 0.7
    xy[1] = [90.0, 45.0]
    conf[1] = 0.7

    # œil droit à (110, 45) conf 0.6
    xy[2] = [110.0, 45.0]
    conf[2] = 0.6

    pt, min_conf = extract_head_point(xy, conf, ("nose", "left_eye", "right_eye"), min_confidence=0.5)
    assert pt is not None
    assert pt[0] == pytest.approx((100.0 + 90.0 + 110.0) / 3.0)
    assert pt[1] == pytest.approx((50.0 + 45.0 + 45.0) / 3.0)
    assert min_conf == pytest.approx(0.6)


def test_extract_head_point_low_confidence_excluded():
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    # nez à (100, 50) conf 0.9 (au-dessus du seuil 0.5)
    xy[0] = [100.0, 50.0]
    conf[0] = 0.9

    # œil gauche à (90, 45) conf 0.3 (sous le seuil 0.5 -> exclu)
    xy[1] = [90.0, 45.0]
    conf[1] = 0.3

    pt, min_conf = extract_head_point(xy, conf, ("nose", "left_eye"), min_confidence=0.5)
    assert pt is not None
    assert pt == (100.0, 50.0)
    assert min_conf == pytest.approx(0.9)


def test_extract_head_point_all_below_threshold_returns_none():
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    xy[0] = [100.0, 50.0]
    conf[0] = 0.4
    xy[1] = [90.0, 45.0]
    conf[1] = 0.2

    pt, min_conf = extract_head_point(xy, conf, ("nose", "left_eye"), min_confidence=0.5)
    assert pt is None
    assert min_conf is None


def test_extract_head_point_unrequested_keypoints_ignored():
    xy = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    # oreilles très confiantes mais non demandées
    xy[3] = [80.0, 40.0]
    conf[3] = 0.95
    xy[4] = [120.0, 40.0]
    conf[4] = 0.95

    # nez sous le seuil
    xy[0] = [100.0, 50.0]
    conf[0] = 0.2

    pt, min_conf = extract_head_point(xy, conf, ("nose",), min_confidence=0.5)
    assert pt is None
    assert min_conf is None
