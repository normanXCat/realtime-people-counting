"""Lot 7 — normalisation CLAHE (canal L de LAB), désactivée par défaut."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from config import DEFAULT_CONFIG_PATH, ClaheConfig, ConfigError, build_config, load_config
from preprocessing import ClaheNormalizer, make_predict_callback


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _frame(seed: int = 0) -> np.ndarray:
    """Frame BGR peu contrastée : dégradé sombre + bruit + un bloc coloré."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(20, 90, 160, dtype=np.float32), (120, 1))
    frame = np.stack([base, base * 0.8, base * 1.1], axis=2)
    frame += rng.normal(0, 4, frame.shape)
    frame[40:80, 50:110] = (30, 40, 120)
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_clahe_disabled_by_default_with_standard_parameters():
    clahe = load_config().preprocessing.clahe
    assert clahe.enabled is False
    assert clahe.clip_limit == pytest.approx(2.0)
    assert clahe.tile_grid_size == (8, 8)


def test_clahe_section_is_optional():
    raw = _raw()
    raw.pop("preprocessing", None)
    assert build_config(raw).preprocessing.clahe.enabled is False


@pytest.mark.parametrize(
    "clahe",
    [
        {"clip_limit": 0},
        {"clip_limit": -1.0},
        {"tile_grid_size": [8]},
        {"tile_grid_size": [0, 8]},
        {"tile_grid_size": "8x8"},
    ],
)
def test_invalid_clahe_values_are_rejected(clahe):
    raw = copy.deepcopy(_raw())
    raw["preprocessing"] = {"clahe": clahe}
    with pytest.raises(ConfigError):
        build_config(raw)


def test_resolved_config_publishes_clahe_and_reloads():
    raw = _raw()
    raw["preprocessing"] = {"clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid_size": [4, 6]}}
    config = build_config(raw)
    dumped = yaml.safe_load(yaml.safe_dump(config.to_dict()))
    assert dumped["preprocessing"]["clahe"] == {
        "enabled": True, "clip_limit": 3.0, "tile_grid_size": [4, 6],
    }
    assert build_config(dumped).preprocessing.clahe.tile_grid_size == (4, 6)


def test_clahe_changes_only_luminance_channel():
    frame = _frame()
    out = ClaheNormalizer(ClaheConfig(enabled=True)).apply(frame)
    assert out.shape == frame.shape and out.dtype == frame.dtype
    lab_in = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(int)
    lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(int)
    # Luminance réellement modifiée (CLAHE est local : pas de critère global).
    assert np.abs(lab_out[..., 0] - lab_in[..., 0]).mean() > 5
    # Chrominance (a, b) conservée, aux arrondis de conversion près.
    assert np.abs(lab_out[..., 1:] - lab_in[..., 1:]).max() <= 3


def test_clahe_matches_opencv_reference_on_l_channel():
    frame = _frame(1)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    expected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    np.testing.assert_array_equal(ClaheNormalizer(ClaheConfig()).apply(frame), expected)


def test_clahe_is_deterministic():
    normalizer = ClaheNormalizer(ClaheConfig())
    frame = _frame(2)
    np.testing.assert_array_equal(normalizer.apply(frame), normalizer.apply(frame.copy()))


def test_predict_callback_normalizes_batch_in_place():
    """YOLO et OSNet voient le même tableau : la frame du lot est modifiée en place."""
    normalizer = ClaheNormalizer(ClaheConfig())
    frames = [_frame(3), _frame(4)]
    expected = [normalizer.apply(f) for f in frames]
    ids = [id(f) for f in frames]
    timings: list[float] = []
    predictor = SimpleNamespace(batch=(["a", "b"], frames, ["", ""]))
    make_predict_callback(normalizer, timings.append)(predictor)
    assert [id(f) for f in predictor.batch[1]] == ids  # mêmes objets, pas de copie
    for got, want in zip(predictor.batch[1], expected):
        np.testing.assert_array_equal(got, want)
    assert len(predictor.batch[1]) == 2  # aucune frame retirée ni ajoutée
    assert len(timings) == 1 and timings[0] >= 0.0


def test_predict_callback_tolerates_missing_batch():
    make_predict_callback(ClaheNormalizer(ClaheConfig()))(SimpleNamespace(batch=None))
