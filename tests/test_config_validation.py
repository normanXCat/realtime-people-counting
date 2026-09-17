"""Niveau 1 — Configuration unique : chargement, validation stricte, conversion.

Couvre la spec 2, 5.6 (rejet des valeurs incohérentes), 5.7 (ligne dégénérée :
refusée à la sélection manuelle, et points absents de la configuration),
0.2 (durées en secondes converties avec le FPS mesuré) et 6.3 (config résolue).
"""

from __future__ import annotations

import copy

import pytest
import yaml

from config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    build_config,
    load_config,
    override_config,
)


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_default_config_loads_and_is_valid():
    config = load_config()
    assert config.schema_version == 1
    assert config.line.inside_side == "negative"
    assert config.line.on_line_policy == "indeterminate"
    assert config.timing.fps_source == "measured"
    assert config.output.write_mode == "append_incremental"
    assert config.stabilization.use_bbox_locker is False
    assert config.stabilization.use_anchor_stabilizer is False
    assert config.reid.long_term.enabled is True
    assert config.reid.external_reid.enabled is False


def test_frame_budget_uses_measured_fps():
    config = load_config()
    budget = config.timing.frame_budget(25.0)
    assert budget.warmup_frames == round(config.timing.warmup_seconds * 25.0)
    assert budget.confirmation_frames == round(config.timing.confirmation_seconds * 25.0)
    assert budget.grace_period_frames == round(config.timing.grace_period_seconds * 25.0)
    assert budget.fps == 25.0


def test_frame_budget_refuses_unknown_fps():
    config = load_config()
    with pytest.raises(ConfigError):
        config.timing.frame_budget(0.0)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "confidence", -0.1),
        ("model", "confidence", 1.0),
        ("model", "nms_iou", 0.0),
        ("model", "image_size", -640),
        ("model", "image_size", 1000),          # non multiple de 32
        ("timing", "warmup_seconds", -1.0),
        ("timing", "grace_period_seconds", 0.0),
        ("timing", "fps_estimate_window", 1),
        ("geometry", "dead_zone_ratio", 0.0),
        ("geometry", "anchor_edge_margin_ratio", 0.6),
        ("geometry", "scale_window", 0),
        ("line", "min_length_ratio", -0.2),
        ("reid.long_term", "safety_margin", -0.1),
        ("reid.long_term", "v_max_ratio", 0.0),
        ("reid.long_term", "similarity_threshold", 2.0),
        ("occupancy", "snapshot_interval_seconds", 0.0),
        ("output", "flush_every_n_events", 0),
        ("source", "no_detection_seconds", -1.0),
        ("stabilization", "bbox_locker_min_height_ratio", 1.5),
    ],
)
def test_invalid_numeric_values_are_rejected(section, key, value):
    raw = _raw()
    if "." in section:
        outer, inner = section.split(".")
        raw[outer][inner][key] = value
    else:
        raw[section][key] = value
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert key in str(error.value)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("line", "inside_side", "gauche"),
        ("line", "on_line_policy", "inside_par_defaut"),
        ("timing", "fps_source", "30"),
        ("output", "write_mode", "overwrite"),
        ("occupancy", "in_progress_disappearance_policy", "devine"),
        ("reid.long_term", "ambiguous_policy", "silencieux"),
    ],
)
def test_invalid_enum_values_are_rejected(section, key, value):
    raw = _raw()
    if "." in section:
        outer, inner = section.split(".")
        raw[outer][inner][key] = value
    else:
        raw[section][key] = value
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert key in str(error.value)


@pytest.mark.parametrize(
    "name",
    [
        "People counting — ligne de comptage",   # tiret cadratin
        "Comptage (unité)",                      # accent
        "Ligne n°1",                             # symbole
        "Chinese 中文",
        "",
        "   ",
    ],
)
def test_un_nom_de_fenetre_non_ascii_ou_vide_est_refuse(name):
    """Le backend Qt d'OpenCV 5.0 ne retrouve pas une fenêtre au nom non ASCII :
    `setMouseCallback` échoue en « NULL window handler » et la sélection
    manuelle de la ligne — donc tout le comptage — devient impossible. Le nom
    est donc validé au chargement, avec un message explicite, plutôt que de
    planter au premier clic.
    """
    raw = _raw()
    raw["display"]["window_name"] = name
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert "window_name" in str(error.value)


def test_un_nom_de_fenetre_ascii_reste_accepte():
    raw = _raw()
    raw["display"]["window_name"] = "People counting (camera 0)"
    config = build_config(raw)
    assert config.display.window_name == "People counting (camera 0)"
    assert config.display.window_name.isascii()


def test_unknown_section_is_rejected():
    raw = _raw()
    raw["pipeline_experimental"] = {"x": 1}
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert "pipeline_experimental" in str(error.value)


def test_default_config_has_no_line_points():
    """La configuration livrée ne contient aucune ligne : ni défaut, ni exemple."""
    raw = _raw()
    assert "p1" not in raw["line"]
    assert "p2" not in raw["line"]
    config = load_config()
    assert not hasattr(config.line, "p1")
    assert not hasattr(config.line, "p2")


def test_le_schema_de_ligne_exclut_tout_parametre_de_double_ligne():
    """Une seule ligne existe : ni L1/L2, ni largeur de porte, ni second jeu de points."""
    from dataclasses import fields

    from config import LineConfig

    assert {item.name for item in fields(LineConfig)} == {
        "inside_side",
        "on_line_policy",
        "min_length_ratio",
    }


@pytest.mark.parametrize("key", ["p1", "p2"])
def test_line_points_in_config_are_rejected(key):
    """`line.p1/p2` ne sont plus lus : les accepter laisserait croire à une ligne active."""
    raw = _raw()
    raw["line"][key] = [0.2, 0.5]
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert f"line.{key}" in str(error.value)
    assert "manuellement" in str(error.value)


def test_all_problems_are_reported_together():
    raw = _raw()
    raw["model"]["confidence"] = -1
    raw["timing"]["grace_period_seconds"] = -1
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    message = str(error.value)
    assert "model.confidence" in message
    assert "timing.grace_period_seconds" in message


def test_override_config_revalidates():
    config = load_config()
    updated = override_config(config, {"model": {"confidence": 0.4}})
    assert updated.model.confidence == 0.4
    assert updated.line.inside_side == config.line.inside_side  # le reste est intact
    with pytest.raises(ConfigError):
        override_config(config, {"model": {"confidence": 2.0}})


def test_override_config_nested_section():
    config = load_config()
    updated = override_config(
        config, {"reid": {"long_term": {"enabled": False, "safety_margin": 0.0}}}
    )
    assert updated.reid.long_term.enabled is False
    assert updated.reid.long_term.safety_margin == 0.0
    assert updated.reid.long_term.similarity_threshold == config.reid.long_term.similarity_threshold


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.yaml")


def test_resolved_config_is_serialisable_and_reloadable():
    config = load_config()
    payload = config.to_dict()
    restored = build_config(copy.deepcopy(payload), config_path=config.config_path)
    assert restored.line == config.line
    assert restored.timing == config.timing
    assert restored.geometry == config.geometry


def test_paths_are_resolved_against_repository_root():
    config = load_config()
    assert config.resolve_path(config.model.path).is_absolute()
    assert config.resolve_path("models/yolo11n.pt").name == "yolo11n.pt"
