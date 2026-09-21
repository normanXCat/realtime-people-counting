"""Lot 3.a — descripteur profond ONNX, repli sur l'histogramme, calcul à la demande.

Aucun de ces tests ne dépend des poids OSNet (non versionnés) : le moteur ONNX
est éprouvé sur un petit réseau exporté dans le test, et la politique de calcul
sur un extracteur instrumenté.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import yaml

from config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ExternalReidConfig,
    LongTermReidConfig,
    build_config,
    load_config,
)
from events import ListSink
from identity_manager import (
    DeepAppearanceExtractor,
    DeepAppearanceUnavailable,
    HistogramAppearanceExtractor,
    IdentityManager,
    Observation,
)

BOX = [80.0, 100.0, 120.0, 300.0]
FRAME = np.zeros((600, 600, 3), dtype=np.uint8)


def observation(track_id: int, confidence: float = 0.9, anchor=(100.0, 300.0)) -> Observation:
    box = np.asarray(BOX, dtype=float)
    return Observation(
        technical_track_id=track_id,
        bbox=box,
        confidence=confidence,
        anchor=tuple(anchor),
        bbox_height=float(box[3] - box[1]),
    )


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key,value",
    [("backend", "tensorrt"), ("descriptor_policy", "sometimes"), ("gallery_refresh_seconds", 0)],
)
def test_invalid_external_reid_values_are_refused(key, value):
    raw = _raw()
    raw["reid"]["external_reid"][key] = value
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert key in str(error.value)


def test_external_reid_parameters_are_read_and_published():
    config = load_config()
    external = config.reid.external_reid
    assert external.backend == "onnxruntime"
    assert external.model_path == "models/osnet_x0_25_msmt17.onnx"
    assert external.descriptor_policy == "on_demand"
    assert external.fallback_to_histogram is True
    published = config.to_dict()["reid"]["external_reid"]
    for key in ("backend", "model_path", "model_expected_sha256", "descriptor_policy",
                "gallery_refresh_seconds", "fallback_to_histogram"):
        assert key in published


# ---------------------------------------------------------------------------
# Repli sur l'histogramme
# ---------------------------------------------------------------------------
def test_missing_model_falls_back_to_histogram_with_a_warning(tmp_path, capsys):
    sink = ListSink("reid")
    manager = IdentityManager(
        external_reid=ExternalReidConfig(enabled=True, model_path=str(tmp_path / "absent.onnx")),
        event_sink=sink,
    )
    assert isinstance(manager.appearance, HistogramAppearanceExtractor)
    assert manager.on_demand_descriptors is False
    assert "absent" in manager.appearance_fallback_reason
    assert sink.count("CONFIG_WARNING") == 1
    assert sink.events[-1]["code"] == "reid_deep_descriptor_unavailable"
    assert "repli sur l'histogramme" in capsys.readouterr().err
    assert manager.descriptor_summary()["fallback_reason"] is not None


def test_fallback_can_be_forbidden(tmp_path):
    with pytest.raises(DeepAppearanceUnavailable):
        IdentityManager(
            external_reid=ExternalReidConfig(
                enabled=True, model_path=str(tmp_path / "absent.onnx"), fallback_to_histogram=False
            )
        )


def test_wrong_checksum_is_refused_before_reading_the_model(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"pas un modele")
    extractor = DeepAppearanceExtractor(
        ExternalReidConfig(enabled=True, model_path=str(model), model_expected_sha256="0" * 64)
    )
    with pytest.raises(DeepAppearanceUnavailable, match="empreinte"):
        extractor.load()


def test_unreadable_model_is_refused(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"pas un modele")
    extractor = DeepAppearanceExtractor(ExternalReidConfig(enabled=True, model_path=str(model)))
    with pytest.raises(DeepAppearanceUnavailable, match="illisible"):
        extractor.load()


# ---------------------------------------------------------------------------
# Moteur ONNX réel, sur un petit réseau exporté pour le test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, stride=4), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(4, 8),
    ).eval()
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    torch.onnx.export(net, torch.randn(1, 3, 256, 128), str(path), input_names=["input"],
                      output_names=["feat"], dynamic_axes={"input": {0: "batch"}}, dynamo=False)
    return path


def test_onnx_backend_reads_model_path_and_returns_a_unit_vector(tiny_onnx):
    digest = hashlib.sha256(tiny_onnx.read_bytes()).hexdigest()
    extractor = DeepAppearanceExtractor(
        ExternalReidConfig(enabled=True, model_path=str(tiny_onnx), model_expected_sha256=digest)
    )
    extractor.load()
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(400, 300, 3), dtype=np.uint8)
    vector = extractor.describe(frame, [20, 30, 120, 330])
    assert vector.shape == (8,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)
    assert extractor.name == "onnx_tiny"


def test_deep_descriptor_is_used_when_the_model_loads(tiny_onnx):
    manager = IdentityManager(
        external_reid=ExternalReidConfig(enabled=True, model_path=str(tiny_onnx))
    )
    assert isinstance(manager.appearance, DeepAppearanceExtractor)
    assert manager.on_demand_descriptors is True
    assert manager.appearance_fallback_reason is None


# ---------------------------------------------------------------------------
# Calcul à la demande (budget FPS)
# ---------------------------------------------------------------------------
@pytest.fixture
def counting_deep(monkeypatch):
    """Extracteur profond instrumenté : chargement neutre, descripteur constant."""
    calls: list[int] = []

    def fake_describe(self, frame, bbox):
        calls.append(1)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(DeepAppearanceExtractor, "load", lambda self: None)
    monkeypatch.setattr(DeepAppearanceExtractor, "describe", fake_describe)
    return calls


def _deep_manager(sink, policy="on_demand", refresh=1.0):
    return IdentityManager(
        long_term=LongTermReidConfig(enabled=True, similarity_threshold=0.5, safety_margin=0.05),
        external_reid=ExternalReidConfig(
            enabled=True, descriptor_policy=policy, gallery_refresh_seconds=refresh
        ),
        grace_period_seconds=1.0,
        event_sink=sink,
    )


def test_on_demand_describes_a_tracked_person_only_when_the_gallery_refresh_is_due(counting_deep):
    sink = ListSink("reid")
    manager = _deep_manager(sink)
    manager.assign([observation(7)], FRAME, 0.0, 1)          # création : insertion
    assert len(counting_deep) == 1
    for index, t in enumerate((0.1, 0.3, 0.6, 0.9), start=2):  # piste suivie, pas dû
        manager.assign([observation(7)], FRAME, t, index)
    assert len(counting_deep) == 1, "aucun calcul pour une piste suivie avant l'échéance"
    manager.assign([observation(7)], FRAME, 1.2, 6)            # rafraîchissement dû
    assert len(counting_deep) == 2
    assert manager.records[1].feature_updated_s == pytest.approx(1.2)
    # Un descripteur non calculé à dessein n'est pas un rejet de qualité.
    assert sink.count("REID_DESCRIPTOR_REJECTED") == 0


def test_on_demand_skips_samples_the_gallery_would_refuse(counting_deep):
    manager = _deep_manager(ListSink("reid"))
    manager.assign([observation(7, confidence=0.9)], FRAME, 0.0, 1)
    manager.assign([observation(7, confidence=0.2)], FRAME, 5.0, 2)  # dû, mais refusable
    assert len(counting_deep) == 1


def test_on_demand_describes_every_unknown_track_for_reidentification(counting_deep):
    manager = _deep_manager(ListSink("reid"))
    manager.assign([observation(7), observation(8, anchor=(400.0, 300.0))], FRAME, 0.0, 1)
    assert len(counting_deep) == 2


def test_every_frame_policy_describes_each_person_each_frame(counting_deep):
    manager = _deep_manager(ListSink("reid"), policy="every_frame")
    for index, t in enumerate((0.0, 0.1, 0.2), start=1):
        manager.assign([observation(7)], FRAME, t, index)
    assert len(counting_deep) == 3
    assert manager.descriptor_summary()["policy"] == "every_frame"


def test_descriptor_cost_is_published(counting_deep):
    manager = _deep_manager(ListSink("reid"))
    manager.assign([observation(7)], FRAME, 0.0, 1)
    summary = manager.descriptor_summary()
    assert summary["policy"] == "on_demand"
    assert summary["computed"] == 1
    assert summary["mean_ms"] is not None


# ---------------------------------------------------------------------------
# Déclencheur de proximité : la correction des inversions reste alimentée
# ---------------------------------------------------------------------------
def _two_people(manager):
    """Deux identités créées loin l'une de l'autre (descripteur de galerie présent)."""
    manager.assign([observation(7, anchor=(100.0, 300.0)), observation(8, anchor=(500.0, 300.0))],
                   FRAME, 0.0, 1)


def _obs_at(track_id, x1):
    box = np.asarray([x1, 100.0, x1 + 40.0, 300.0], dtype=float)
    return Observation(technical_track_id=track_id, bbox=box, confidence=0.9,
                       anchor=(x1 + 20.0, 300.0), bbox_height=200.0)


def test_touching_tracks_are_described_for_swap_correction(counting_deep):
    manager = _deep_manager(ListSink("reid"))
    _two_people(manager)
    before = len(counting_deep)
    # Boîtes qui se recouvrent, avant l'échéance de rafraîchissement.
    manager.assign([_obs_at(7, 80.0), _obs_at(8, 110.0)], FRAME, 0.2, 2)
    assert len(counting_deep) - before == 2
    assert manager.proximity_descriptors_computed == 2
    assert manager.descriptor_summary()["computed_for_swap_correction"] == 2


def test_distant_tracks_are_not_described(counting_deep):
    manager = _deep_manager(ListSink("reid"))
    _two_people(manager)
    before = len(counting_deep)
    manager.assign([_obs_at(7, 80.0), _obs_at(8, 480.0)], FRAME, 0.2, 2)
    assert len(counting_deep) == before
    assert manager.proximity_descriptors_computed == 0


def test_proximity_descriptors_are_not_written_to_the_gallery(counting_deep):
    manager = _deep_manager(ListSink("reid"))
    _two_people(manager)
    stamp = manager.records[1].feature_updated_s
    manager.assign([_obs_at(7, 80.0), _obs_at(8, 110.0)], FRAME, 0.2, 2)
    assert manager.records[1].feature_updated_s == stamp


def test_swap_correction_sees_crossing_tracks_in_on_demand_mode(monkeypatch):
    """Sans déclencheur, une inversion croisée passerait inaperçue en on_demand."""
    appearance = {80.0: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # piste 7 : apparence de B
                  110.0: np.array([1.0, 0.0, 0.0], dtype=np.float32)}  # piste 8 : apparence de A

    def fake_describe(self, frame, bbox):
        return appearance.get(float(bbox[0]), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    monkeypatch.setattr(DeepAppearanceExtractor, "load", lambda self: None)
    monkeypatch.setattr(DeepAppearanceExtractor, "describe", fake_describe)
    sink = ListSink("reid")
    manager = _deep_manager(sink)
    manager.assign([_obs_at(7, 0.0), _obs_at(8, 500.0)], FRAME, 0.0, 1)
    manager.records[1].feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # A
    manager.records[2].feature = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # B
    manager.assign([_obs_at(7, 80.0), _obs_at(8, 110.0)], FRAME, 0.2, 2)
    assert sink.count("IDENTITY_SWAP_CORRECTED") == 1
    assert manager.person_of(7) == 2 and manager.person_of(8) == 1


def test_proximity_trigger_can_be_disabled(counting_deep):
    manager = IdentityManager(
        long_term=LongTermReidConfig(enabled=True, similarity_threshold=0.5, safety_margin=0.05),
        external_reid=ExternalReidConfig(enabled=True, describe_on_proximity=False),
        grace_period_seconds=1.0,
    )
    _two_people(manager)
    before = len(counting_deep)
    manager.assign([_obs_at(7, 80.0), _obs_at(8, 110.0)], FRAME, 0.2, 2)
    assert len(counting_deep) == before
