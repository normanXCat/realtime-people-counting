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
    assert config.timing.warmup_min_frames == 15
    assert config.timing.warmup_seconds == pytest.approx(0.5)
    assert config.line.inside_side == "negative"
    assert config.line.on_line_policy == "indeterminate"
    assert config.timing.fps_source == "measured"
    assert config.output.write_mode == "append_incremental"
    # Le verrouillage de hauteur est ACTIF dans le pipeline officiel (sans lui,
    # une chute de hauteur de boîte produit une fausse transition de zone).
    assert config.stabilization.use_bbox_locker is True
    assert config.stabilization.use_anchor_stabilizer is False
    assert config.reid.long_term.enabled is True
    # L'extracteur ReID profond est ACTIVÉ dans la configuration de référence
    # (avec repli HSV propre si la dépendance/poids sont absents).
    assert config.reid.external_reid.enabled is True
    # Garde-fou de swap : actif, seuil calibré sur le p1 mesuré (0.55).
    assert config.reid.appearance_continuity.enabled is True
    assert config.reid.appearance_continuity.similarity_threshold == pytest.approx(0.55)
    assert config.reid.appearance_continuity.max_gap_frames == 2
    # Calibrations anti-swap (ReID profond).
    assert config.reid.swap_correction.margin == pytest.approx(0.08)
    # Durcissement du ReID natif (A2) : apparence plus proche exigée avant
    # réassociation — doublé dans custom_botsort.yaml (test de synchronisation).
    assert config.tracker.appearance_thresh == pytest.approx(0.35)
    # Résolution globale par groupe (A1) : proximité exprimée en hauteur de bbox.
    assert config.reid.swap_correction.group_proximity_ratio == pytest.approx(0.5)
    # Chemin de secours « galerie » (B3) : désactivé par défaut, plancher de
    # confiance replié sur ``track_low_thresh`` quand il n'est pas déclaré.
    assert config.reid.long_term.gallery_rescue_enabled is False
    assert config.reid.long_term.gallery_rescue_min_confidence == pytest.approx(
        config.tracker.track_low_thresh
    )
    assert config.model.nms_iou == pytest.approx(0.40)
    assert config.timing.grace_period_seconds == pytest.approx(6.0)


def test_frame_budget_uses_measured_fps():
    config = load_config()
    budget = config.timing.frame_budget(25.0)
    assert budget.warmup_frames == round(config.timing.warmup_seconds * 25.0)
    assert budget.confirmation_frames == round(config.timing.confirmation_seconds * 25.0)
    assert budget.grace_period_frames == round(config.timing.grace_period_seconds * 25.0)
    assert budget.fps == 25.0


def test_frame_budget_carries_the_warmup_frame_floor():
    """Les deux bornes du warm-up voyagent ensemble dans le budget."""
    config = load_config()
    budget = config.timing.frame_budget(25.0)
    assert budget.warmup_min_frames == config.timing.warmup_min_frames
    assert budget.to_dict()["warmup_min_frames"] == config.timing.warmup_min_frames


# ---------------------------------------------------------------------------
# Seuils de détection / d'association (spec 2 du prompt)
# ---------------------------------------------------------------------------
def test_yolo_threshold_is_low_enough_for_the_tracker_low_stage():
    """Le seuil d'inférence ne doit pas supprimer les détections faibles utiles.

    Ultralytics filtre avec ``model.confidence`` **avant** BoT-SORT : tout score
    sous ce seuil n'existe plus pour le second étage d'association. La cohérence
    ``model.confidence <= tracker.track_low_thresh`` est donc ce qui rend la
    récupération des personnes floues possible.
    """
    from config import coherence_warnings

    config = load_config()
    assert config.model.confidence == pytest.approx(0.10)
    assert config.tracker.track_low_thresh == pytest.approx(0.1)
    assert config.tracker.track_high_thresh == pytest.approx(0.4)
    assert config.tracker.new_track_thresh == pytest.approx(0.7)
    assert config.model.confidence <= config.tracker.track_low_thresh
    codes = [code for code, _message in coherence_warnings(config)]
    assert "yolo_confidence_above_tracker_low_thresh" not in codes


def test_high_yolo_threshold_is_warned_about_not_silently_accepted():
    """Un seuil YOLO trop haut reste possible (ablation) mais est journalisé."""
    from config import coherence_warnings

    raised = override_config(load_config(), {"model": {"confidence": 0.42}})
    warnings = coherence_warnings(raised)
    codes = [code for code, _message in warnings]
    assert "yolo_confidence_above_tracker_low_thresh" in codes
    yolo_warning = next(
        message
        for code, message in warnings
        if code == "yolo_confidence_above_tracker_low_thresh"
    )
    assert "track_low_thresh" in yolo_warning
    assert "blou" in yolo_warning or "flou" in yolo_warning


def test_tracker_thresholds_must_stay_distinct_and_ordered():
    raw = _raw()
    raw["tracker"]["track_low_thresh"] = raw["tracker"]["track_high_thresh"]
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert "track_low_thresh" in str(error.value)

    raw = _raw()
    raw["tracker"]["new_track_thresh"] = 0.2
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert "new_track_thresh" in str(error.value)


def _tracker_values_match(left, right) -> bool:
    """Égalité de valeurs YAML (booléens, nombres, chaînes) tolérante au typage."""
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    return left == right


def test_botsort_yaml_is_synchronised_with_the_configuration():
    """Divergence silencieuse interdite entre le YAML lu par Ultralytics et la config.

    La synchronisation porte sur **100 % des clés** du fichier lu par Ultralytics
    (``tracker_type``, ``fuse_score``, ``model`` compris), pas seulement sur les
    trois seuils : une clé réglée dans le YAML mais oubliée dans la configuration
    piloterait le comportement sans que le journal ni le protocole ne le sachent.
    """
    from dataclasses import fields

    from config import REPO_ROOT, TrackerConfig

    config = load_config()
    tracker_path = REPO_ROOT / config.tracker.config_path
    tracker_yaml = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))

    # 1. Chaque clé du YAML est portée par TrackerConfig, valeur identique.
    for key, value in tracker_yaml.items():
        assert hasattr(config.tracker, key), (
            f"clé BoT-SORT {key!r} absente de TrackerConfig : divergence silencieuse"
        )
        assert _tracker_values_match(getattr(config.tracker, key), value), (
            f"tracker.{key} diverge : YAML={value!r} config={getattr(config.tracker, key)!r}"
        )

    # 2. Aucune clé de TrackerConfig propre au YAML n'est oubliée (sauf les
    #    métadonnées internes : chemin du fichier et persistance Ultralytics).
    metadata = {"config_path", "persist"}
    config_keys = {field.name for field in fields(TrackerConfig)} - metadata
    assert config_keys == set(tracker_yaml), (
        "le YAML BoT-SORT et TrackerConfig ne portent pas exactement les mêmes clés"
    )

    # 3. pipeline.yaml déclare explicitement les mêmes clés que le YAML lu par
    #    Ultralytics : la duplication est volontaire, la divergence interdite.
    pipeline_tracker_keys = set(_raw()["tracker"]) - metadata
    assert pipeline_tracker_keys == set(tracker_yaml)


def test_track_buffer_covers_the_measured_occlusion_duration():
    """``track_buffer`` est réglé sur les occlusions **mesurées**, pas par défaut.

    Occlusions mesurées dans ``results/*/events.jsonl`` (OCCLUDED -> réapparition) :
    médiane 2,5 s, p90 5,1 s, maximum 13,5 s. Porté à 240 frames ≈ 8 s à 30 ips :
    couvre le p90 + une marge (ce qui réduit les handoffs vers le ReID long
    terme) sans dépasser l'occlusion maximale observée.
    """
    config = load_config()
    source_fps = 30.0
    typical_occlusion_s = 2.5
    p90_occlusion_s = 5.1
    max_occlusion_s = 13.5
    covered_s = config.tracker.track_buffer / source_fps
    assert covered_s >= typical_occlusion_s
    assert covered_s >= p90_occlusion_s + 1.0
    assert covered_s <= max_occlusion_s
    assert config.tracker.track_buffer > 0


def test_tracker_persistence_and_reid_are_enabled_and_camera_is_fixed():
    config = load_config()
    assert config.tracker.with_reid is True
    assert config.tracker.gmc_method == "none"
    assert config.tracker.persist is True
    assert "gmc_method" in config.to_dict()["tracker"]


def test_unknown_gmc_method_is_rejected():
    """Une GMC sur caméra fixe est une source d'instabilité : elle est refusée."""
    raw = _raw()
    raw["tracker"]["gmc_method"] = "magic_flow"
    with pytest.raises(ConfigError) as error:
        build_config(raw)
    assert "gmc_method" in str(error.value)


def test_diagnostics_section_is_exposed_and_configurable():
    config = load_config()
    assert config.diagnostics.enabled is True
    assert config.diagnostics.min_interval_seconds == pytest.approx(1.0)
    updated = override_config(
        config, {"diagnostics": {"enabled": False, "min_interval_seconds": 2.5}}
    )
    assert updated.diagnostics.enabled is False
    assert updated.diagnostics.min_interval_seconds == pytest.approx(2.5)
    assert "diagnostics" in config.to_dict()


def test_gallery_retention_covers_the_longest_measured_occlusion():
    """Grâce (6 s) + rétention (10 s) = 16 s, au-delà de l'occlusion max (13,5 s)."""
    config = load_config()
    assert config.reid.long_term.gallery_retention_seconds == pytest.approx(10.0)
    from identity_manager import IdentityManager

    manager = IdentityManager(
        long_term=config.reid.long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    assert manager.purge_retention_seconds == pytest.approx(10.0)
    # Couverture totale = grâce d'occupation + rétention >= occlusion max mesurée.
    covered = config.timing.grace_period_seconds + manager.purge_retention_seconds
    assert covered >= 13.5
    assert covered == pytest.approx(16.0)


def test_gallery_retention_can_be_decoupled_explicitly():
    config = override_config(
        load_config(), {"reid": {"long_term": {"gallery_retention_seconds": 12.0}}}
    )
    assert config.reid.long_term.gallery_retention_seconds == pytest.approx(12.0)
    with pytest.raises(ConfigError):
        override_config(
            load_config(),
            {"reid": {"long_term": {"gallery_retention_seconds": -1.0}}},
        )


def test_progressive_similarity_floor_cannot_exceed_the_nominal_threshold():
    config = load_config()
    # Assouplissement progressif activé dans la configuration de référence.
    assert config.reid.long_term.long_absence_similarity_floor == pytest.approx(0.30)
    assert (
        config.reid.long_term.long_absence_similarity_floor
        <= config.reid.long_term.similarity_threshold
    )
    with pytest.raises(ConfigError) as error:
        override_config(
            load_config(),
            {"reid": {"long_term": {"long_absence_similarity_floor": 0.95}}},
        )
    assert "long_absence_similarity_floor" in str(error.value)


def test_fps_source_accepts_video_native_for_offline_runs():
    """``video_native`` est accepté pour horodater un fichier hors temps réel."""
    from config import ALLOWED_FPS_SOURCES

    assert "video_native" in ALLOWED_FPS_SOURCES
    updated = override_config(load_config(), {"timing": {"fps_source": "video_native"}})
    assert updated.timing.fps_source == "video_native"
    with pytest.raises(ConfigError) as error:
        override_config(load_config(), {"timing": {"fps_source": "guess"}})
    assert "fps_source" in str(error.value)


def test_progressive_floor_decays_over_the_configured_long_absence_window():
    """Le plancher est atteint linéairement au bout de long_absence_seconds (6 s)."""
    from identity_manager import IdentityManager

    config = load_config()
    long_term = config.reid.long_term
    assert long_term.long_absence_seconds == pytest.approx(6.0)
    assert long_term.safety_margin == pytest.approx(0.10)
    manager = IdentityManager(
        long_term=long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    base = long_term.similarity_threshold
    floor = long_term.long_absence_similarity_floor
    assert floor == pytest.approx(0.30)
    assert manager.effective_similarity_threshold(0.0) == pytest.approx(base)
    assert manager.effective_similarity_threshold(3.0) == pytest.approx((base + floor) / 2)
    assert manager.effective_similarity_threshold(6.0) == pytest.approx(floor)
    assert manager.effective_similarity_threshold(60.0) == pytest.approx(floor)


def test_appearance_continuity_guard_is_exposed_and_configurable():
    """Le garde-fou de swap vit dans la configuration, jamais en dur."""
    config = load_config()
    updated = override_config(
        config,
        {
            "reid": {
                "appearance_continuity": {
                    "enabled": False,
                    "similarity_threshold": 0.7,
                    "max_gap_frames": 0,
                    "min_interval_seconds": 2.0,
                }
            }
        },
    )
    assert updated.reid.appearance_continuity.enabled is False
    assert updated.reid.appearance_continuity.similarity_threshold == pytest.approx(0.7)
    assert updated.reid.appearance_continuity.max_gap_frames == 0
    assert updated.reid.appearance_continuity.min_interval_seconds == pytest.approx(2.0)
    # La sérialisation doit rester revalidable telle quelle.
    restored = build_config(copy.deepcopy(updated.to_dict()), config_path=config.config_path)
    assert restored.reid.appearance_continuity == updated.reid.appearance_continuity


def test_occlusion_ambiguity_radius_is_relative_to_the_local_scale():
    """Le rayon d'ambiguïté d'un NEW est une fraction de hauteur, pas des pixels."""
    config = load_config()
    assert config.geometry.occlusion_ambiguity_ratio > 0.0
    assert config.geometry.occlusion_ambiguity_ratio == pytest.approx(0.5)


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
        ("timing", "warmup_min_frames", 0),
        ("timing", "grace_period_seconds", 0.0),
        ("timing", "fps_estimate_window", 1),
        ("geometry", "dead_zone_ratio", 0.0),
        ("geometry", "anchor_edge_margin_ratio", 0.6),
        ("geometry", "scale_window", 0),
        ("geometry", "occlusion_ambiguity_ratio", -0.1),
        ("line", "min_length_ratio", -0.2),
        ("reid.long_term", "safety_margin", -0.1),
        ("reid.long_term", "v_max_ratio", 0.0),
        ("reid.long_term", "similarity_threshold", 2.0),
        ("reid.long_term", "gallery_min_confidence", 1.5),
        ("reid.long_term", "gallery_min_crop_height_px", 0),
        ("reid.long_term", "gallery_min_crop_width_px", 0),
        ("reid.long_term", "gallery_min_sharpness", -1.0),
        ("tracker", "track_high_thresh", 0.0),
        ("tracker", "track_low_thresh", 1.0),
        ("tracker", "new_track_thresh", 2.0),
        ("tracker", "track_buffer", 0),
        ("tracker", "match_thresh", 0.0),
        ("tracker", "match_thresh", 1.5),
        ("tracker", "proximity_thresh", -0.1),
        ("tracker", "appearance_thresh", 2.0),
        ("reid.long_term", "gallery_retention_seconds", -1.0),
        ("reid.long_term", "long_absence_seconds", 0.0),
        ("diagnostics", "min_interval_seconds", 0.0),
        ("occupancy", "snapshot_interval_seconds", 0.0),
        ("output", "flush_every_n_events", 0),
        ("source", "no_detection_seconds", -1.0),
        ("stabilization", "bbox_locker_min_height_ratio", 1.5),
        ("reid.appearance_continuity", "similarity_threshold", 1.5),
        ("reid.appearance_continuity", "max_gap_frames", -1),
        ("reid.appearance_continuity", "min_interval_seconds", -0.1),
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


def test_default_config_has_head_assist_enabled():
    config = load_config()
    assert hasattr(config, "presence")
    assert config.presence.head_assist.enabled is True
    assert config.presence.head_assist.pose_model_path == "models/yolo11s-pose.pt"
    assert config.presence.head_assist.min_keypoint_confidence == pytest.approx(0.5)
    assert config.presence.head_assist.max_presence_extension_seconds == pytest.approx(10.0)
    assert config.model.device == "cpu"


def test_missing_pose_weights_are_reported_as_a_coherence_warning():
    """Un flag head_assist actif sans poids ne doit pas faire échouer le chargement.

    L'absence est signalée en ``CONFIG_WARNING`` (et ``main`` se rabat proprement
    sur le modèle de détection standard).
    """
    from config import REPO_ROOT, coherence_warnings

    config = load_config()
    pose_path = REPO_ROOT / config.presence.head_assist.pose_model_path
    if pose_path.exists():
        pytest.skip("poids de pose présents : cas non reproductible sur cette machine")
    codes = [code for code, _message in coherence_warnings(config)]
    assert "head_assist_model_missing" in codes


def test_head_assist_invalid_confidence_rejected():
    raw = _raw()
    raw.setdefault("presence", {}).setdefault("head_assist", {})["min_keypoint_confidence"] = 1.5
    with pytest.raises(ConfigError) as exc_info:
        build_config(raw)
    assert "presence.head_assist.min_keypoint_confidence" in str(exc_info.value)

    raw = _raw()
    raw.setdefault("presence", {}).setdefault("head_assist", {})["min_keypoint_confidence"] = -0.1
    with pytest.raises(ConfigError) as exc_info:
        build_config(raw)
    assert "presence.head_assist.min_keypoint_confidence" in str(exc_info.value)


def test_head_assist_invalid_extension_duration_rejected():
    raw = _raw()
    raw.setdefault("presence", {}).setdefault("head_assist", {})["max_presence_extension_seconds"] = 0.0
    with pytest.raises(ConfigError) as exc_info:
        build_config(raw)
    assert "presence.head_assist.max_presence_extension_seconds" in str(exc_info.value)

    raw = _raw()
    raw.setdefault("presence", {}).setdefault("head_assist", {})["max_presence_extension_seconds"] = -5.0
    with pytest.raises(ConfigError) as exc_info:
        build_config(raw)
    assert "presence.head_assist.max_presence_extension_seconds" in str(exc_info.value)


def test_head_assist_unknown_keypoints_rejected():
    raw = _raw()
    raw.setdefault("presence", {}).setdefault("head_assist", {})["keypoints_used"] = ["nose", "wrist"]
    with pytest.raises(ConfigError) as exc_info:
        build_config(raw)
    assert "presence.head_assist.keypoints_used" in str(exc_info.value)


def test_head_assist_empty_keypoints_rejected():
    raw = _raw()
    raw.setdefault("presence", {}).setdefault("head_assist", {})["keypoints_used"] = []
    with pytest.raises(ConfigError) as exc_info:
        build_config(raw)
    assert "presence.head_assist.keypoints_used" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Résolution de groupe de swap (A1) et chemin de secours galerie (B3)
# ---------------------------------------------------------------------------
def test_swap_group_proximity_is_configurable_and_validated():
    config = override_config(
        load_config(), {"reid": {"swap_correction": {"group_proximity_ratio": 1.25}}}
    )
    assert config.reid.swap_correction.group_proximity_ratio == pytest.approx(1.25)
    assert "group_proximity_ratio" in config.to_dict()["reid"]["swap_correction"]

    with pytest.raises(ConfigError) as error:
        override_config(
            load_config(),
            {"reid": {"swap_correction": {"group_proximity_ratio": -0.1}}},
        )
    assert "reid.swap_correction.group_proximity_ratio" in str(error.value)


def test_gallery_rescue_defaults_and_floor_resolution():
    """``null`` se replie sur ``track_low_thresh`` ; une valeur explicite est honorée."""
    config = load_config()
    assert config.reid.long_term.gallery_rescue_enabled is False
    assert config.reid.long_term.gallery_rescue_min_confidence == pytest.approx(
        config.tracker.track_low_thresh
    )
    assert "gallery_rescue_enabled" in config.to_dict()["reid"]["long_term"]

    explicit = override_config(
        load_config(),
        {"reid": {"long_term": {"gallery_rescue_min_confidence": 0.33}}},
    )
    assert explicit.reid.long_term.gallery_rescue_min_confidence == pytest.approx(0.33)

    with pytest.raises(ConfigError) as error:
        override_config(
            load_config(),
            {"reid": {"long_term": {"gallery_rescue_min_confidence": 1.5}}},
        )
    assert "reid.long_term.gallery_rescue_min_confidence" in str(error.value)


def test_gallery_rescue_without_long_term_reid_is_warned_about():
    """Activer le secours sans galerie est journalisé, jamais silencieux."""
    from config import coherence_warnings

    config = override_config(
        load_config(),
        {"reid": {"long_term": {"enabled": False, "gallery_rescue_enabled": True}}},
    )
    codes = [code for code, _message in coherence_warnings(config)]
    assert "gallery_rescue_without_long_term_reid" in codes

    # Cohérent : aucun avertissement de ce type.
    ok = override_config(
        load_config(),
        {"reid": {"long_term": {"enabled": True, "gallery_rescue_enabled": True}}},
    )
    codes = [code for code, _message in coherence_warnings(ok)]
    assert "gallery_rescue_without_long_term_reid" not in codes


def test_provisional_merge_confirmation_frames_validation():
    config = load_config()
    assert config.reid.long_term.provisional_merge_confirmation_frames == 5

    # Valide avec >= 2
    updated = override_config(
        config,
        {"reid": {"long_term": {"provisional_merge_confirmation_frames": 3}}},
    )
    assert updated.reid.long_term.provisional_merge_confirmation_frames == 3

    # Refuse < 2 (règle 0.4 : aucune décision sur signal faible / 1 observation)
    with pytest.raises(ConfigError) as exc_info:
        override_config(
            config,
            {"reid": {"long_term": {"provisional_merge_confirmation_frames": 1}}},
        )
    assert "provisional_merge_confirmation_frames" in str(exc_info.value)

