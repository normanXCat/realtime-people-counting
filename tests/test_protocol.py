"""Tests du protocole d'évaluation (spec 9.1, 9.4) et de la rejouabilité d'un run.

Deux garanties sont vérifiées ici, car elles conditionnent la validité
scientifique du dispositif :

- la séparation **stricte** calibration / test (9.1) ;
- la **rejouabilité** d'une session : `config_resolved.yaml` doit se recharger
  tel quel et reproduire exactement la configuration exécutée (spec 6.3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import ConfigError, load_config, override_config  # noqa: E402
from run_protocol import build_variant_config, check_disjoint, load_protocol  # noqa: E402

PROTOCOL_PATH = REPO_ROOT / "config" / "protocol.yaml"


def test_protocole_declare_est_valide():
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol["schema_version"] == 1
    assert set(protocol["acceptance"]) >= {"min_f1", "max_occupancy_mae", "tolerance_seconds"}
    # Les neuf étapes de la section 9.4 sont déclarées.
    assert set(protocol["variants"]) == {
        "baseline_1", "baseline_2", "variant_1", "variant_2",
        "variant_3", "variant_4", "variant_5", "variant_6",
    }
    check_disjoint(protocol)


def test_chevauchement_calibration_test_refuse(tmp_path: Path):
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["datasets"] = {
        "calibration": [{"video": "data/videos/a.mp4", "ground_truth": "data/gt/a.json"}],
        "test": [{"video": "data/videos/a.mp4", "ground_truth": "data/gt/a.json"}],
    }
    with pytest.raises(ConfigError, match="chevauchent"):
        check_disjoint(protocol)


def test_ensemble_declare_doit_etre_complet():
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["datasets"] = {"calibration": [{"video": "a.mp4"}], "test": []}
    with pytest.raises(ConfigError, match="ground_truth"):
        check_disjoint(protocol)


@pytest.mark.parametrize("variant_name", ["baseline_1", "variant_1", "variant_4"])
def test_config_de_variante_est_revalidee(tmp_path: Path, variant_name: str):
    protocol = load_protocol(PROTOCOL_PATH)
    variant = {**protocol["variants"][variant_name], "name": variant_name}
    output = build_variant_config(protocol["base_config"], variant, tmp_path / "config.yaml")

    # Le fichier écrit est lui-même une configuration valide : c'est ce qui rend
    # l'expérience rejouable à l'identique.
    reloaded = load_config(output)
    assert reloaded.schema_version == 1
    assert reloaded.model.path == protocol["variants"][variant_name].get("config", {}).get(
        "model", {}
    ).get("path", "models/yolo11n.pt")
    # La variante écrite doit différer de la base exactement par le facteur étudié.
    overrides = protocol["variants"][variant_name].get("config") or {}
    if "stabilization" in overrides:
        assert reloaded.stabilization.use_bbox_locker is True
    if overrides.get("reid", {}).get("long_term", {}).get("enabled") is False:
        assert reloaded.reid.long_term.enabled is False


def test_variante_1_est_la_configuration_de_reference():
    base = load_config()
    protocol = load_protocol(PROTOCOL_PATH)
    reference = override_config(base, protocol["variants"]["variant_1"]["config"])
    assert reference == base


def test_baseline_1_desactive_le_reid_long_terme():
    protocol = load_protocol(PROTOCOL_PATH)
    base = load_config()
    baseline = override_config(base, protocol["variants"]["baseline_1"]["config"])
    assert baseline.reid.long_term.enabled is False
    assert base.reid.long_term.enabled is True


def test_variante_4_active_la_stabilisation():
    protocol = load_protocol(PROTOCOL_PATH)
    variant = override_config(load_config(), protocol["variants"]["variant_4"]["config"])
    assert variant.stabilization.use_bbox_locker is True
    assert variant.stabilization.use_anchor_stabilizer is True


def test_config_resolue_de_session_est_rejouable(tmp_path: Path):
    """`config_resolved.yaml` doit se recharger et redonner la même configuration."""
    from main import prepare_session, write_resolved_config  # import tardif : cv2/torch inutiles ici

    config = override_config(
        load_config(),
        {
            "output": {"session_root": str(tmp_path), "write_video": False},
            # L'orientation validée par l'opérateur dans la fenêtre de sélection.
            "line": {"inside_side": "positive"},
        },
    )
    session_root = prepare_session(config, "session-test")
    resolved_path = write_resolved_config(config, session_root, "session-test")

    assert resolved_path.exists()
    assert (session_root / "config_resolved.yaml").read_text(encoding="utf-8").startswith("#")
    replayed = load_config(resolved_path)
    assert replayed.to_dict() == config.to_dict()
    # La session fige bien l'orientation retenue à l'écran.
    assert replayed.line.inside_side == "positive"


def test_config_resolue_ne_contient_que_des_sections_du_schema(tmp_path: Path):
    from main import prepare_session, write_resolved_config

    config = override_config(load_config(), {"output": {"session_root": str(tmp_path)}})
    session_root = prepare_session(config, "session-schema")
    resolved_path = write_resolved_config(config, session_root, "session-schema")
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert "config_path" not in payload
    assert set(payload) == {
        "schema_version", "source", "model", "tracker", "reid", "line", "geometry",
        "timing", "occupancy", "output", "stabilization", "display", "logging",
        "diagnostics", "anchor",
    }
