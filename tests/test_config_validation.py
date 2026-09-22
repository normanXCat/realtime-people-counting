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
    LongTermReidConfig,
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
    # Lot 3.a : descripteur OSNet activé (docs/correctif_occlusion.md §17), avec
    # repli sur l'histogramme si le modèle manque.
    assert config.reid.external_reid.enabled is True
    assert config.reid.external_reid.fallback_to_histogram is True
    # Garde-fou de swap : actif, mais purement diagnostique.
    assert config.reid.appearance_continuity.enabled is True
    assert config.reid.appearance_continuity.similarity_threshold == pytest.approx(0.5)
    assert config.reid.appearance_continuity.max_gap_frames == 2


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
    # 0,4 -> 0,30 au lot 4 (docs/correctif_occlusion.md §15.15), puis rétabli à
    # 0,40 (version E, §18 : surcomptage par NEW sur fragments). L'ordre des
    # seuils vérifié ci-dessous est inchangé.
    assert config.tracker.track_high_thresh == pytest.approx(0.40)
    assert config.tracker.track_low_thresh < config.tracker.track_high_thresh
    # 0,7 -> 0,60 au lot 4 (docs/correctif_occlusion.md §15), puis rétabli à
    # 0,70 (version E, §18). L'invariant vérifié ici est inchangé : le seuil de
    # création reste au-dessus du plancher d'association principal et du seuil
    # YOLO.
    assert config.tracker.new_track_thresh == pytest.approx(0.70)
    assert config.tracker.new_track_thresh >= config.tracker.track_high_thresh
    assert config.model.confidence <= config.tracker.track_low_thresh
    assert coherence_warnings(config) == []


def test_high_yolo_threshold_is_warned_about_not_silently_accepted():
    """Un seuil YOLO trop haut reste possible (ablation) mais est journalisé."""
    from config import coherence_warnings

    raised = override_config(load_config(), {"model": {"confidence": 0.42}})
    warnings = coherence_warnings(raised)
    assert [code for code, _message in warnings] == [
        "yolo_confidence_above_tracker_low_thresh"
    ]
    assert "track_low_thresh" in warnings[0][1]
    assert "blou" in warnings[0][1] or "flou" in warnings[0][1]


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


def test_botsort_yaml_is_synchronised_with_the_configuration():
    """Divergence silencieuse interdite entre le YAML lu par Ultralytics et la config.

    La synchronisation porte sur **toutes** les valeurs de persistance et de ReID
    natif, pas seulement sur les trois seuils : un ``track_buffer`` réglé dans le
    YAML mais oublié dans la configuration piloterait le comportement sans que le
    journal ni le protocole ne le sachent.
    """
    from config import REPO_ROOT

    config = load_config()
    tracker_yaml = yaml.safe_load(
        (REPO_ROOT / config.tracker.config_path).read_text(encoding="utf-8")
    )
    assert tracker_yaml["track_high_thresh"] == pytest.approx(
        config.tracker.track_high_thresh
    )
    assert tracker_yaml["track_low_thresh"] == pytest.approx(
        config.tracker.track_low_thresh
    )
    assert tracker_yaml["new_track_thresh"] == pytest.approx(
        config.tracker.new_track_thresh
    )
    assert tracker_yaml["track_buffer"] == config.tracker.track_buffer
    assert tracker_yaml["with_reid"] == config.tracker.with_reid
    assert tracker_yaml["match_thresh"] == pytest.approx(config.tracker.match_thresh)
    assert tracker_yaml["proximity_thresh"] == pytest.approx(
        config.tracker.proximity_thresh
    )
    assert tracker_yaml["appearance_thresh"] == pytest.approx(
        config.tracker.appearance_thresh
    )
    assert tracker_yaml["gmc_method"] == config.tracker.gmc_method


def test_track_buffer_covers_the_measured_occlusion_duration():
    """``track_buffer`` est réglé sur les occlusions **mesurées**, pas par défaut.

    Occlusions mesurées dans ``results/*/events.jsonl`` (OCCLUDED -> réapparition) :
    médiane 2,5 s, p90 5,1 s. Le buffer doit couvrir au moins la médiane ; 150
    frames valent 5 s à la cadence source de 30 ips, soit le p90 observé.
    """
    config = load_config()
    source_fps = 30.0
    typical_occlusion_s = 2.5
    p90_occlusion_s = 5.1
    covered_s = config.tracker.track_buffer / source_fps
    assert covered_s >= typical_occlusion_s
    assert covered_s == pytest.approx(p90_occlusion_s, abs=0.2)
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


def test_production_gallery_retention_is_12_seconds_and_covers_long_occlusions():
    """La config livrée découple la mémoire d'apparence de la grâce (12,0 s).

    Ce test porte sur la **valeur de production** : il doit donc vérifier 12,0,
    jamais ``None``. Le repli sur ``grace_period_seconds`` quand la clé est omise
    est un comportement distinct, testé séparément ci-dessous.
    """
    config = load_config()
    assert config.reid.long_term.gallery_retention_seconds == pytest.approx(12.0)
    from identity_manager import IdentityManager

    manager = IdentityManager(
        long_term=config.reid.long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    assert manager.purge_retention_seconds == pytest.approx(12.0)
    # Fenêtre totale = grâce 5,0 s + rétention 12,0 s = 17,0 s : elle couvre
    # l'occlusion maximale mesurée (13,5 s, rapport §4.6), qui tombait hors
    # mémoire avec l'ancienne valeur alignée (10,0 s).
    assert (
        config.timing.grace_period_seconds + manager.purge_retention_seconds
    ) > 13.5


def test_gallery_retention_falls_back_on_the_grace_period_when_omitted():
    """Repli : une rétention omise (``None``) s'aligne sur la grâce d'occupation."""
    from identity_manager import IdentityManager

    manager = IdentityManager(
        long_term=LongTermReidConfig(gallery_retention_seconds=None),
        grace_period_seconds=2.5,
    )
    assert manager.purge_retention_seconds == pytest.approx(2.5)


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


def test_feature_momentum_is_configurable_and_bounded():
    """Le momentum du descripteur de galerie vient de la config, borné à [0, 1[.

    Il était codé en dur à 0,85 dans le constructeur de ``IdentityManager`` : non
    réglable, alors qu'il gouverne la vitesse de dérive de la référence après un
    swap. ``1,0`` doit être refusé : il figerait le descripteur pour toujours.
    """
    config = load_config()
    assert config.reid.long_term.feature_momentum == pytest.approx(0.85)

    lowered = override_config(
        load_config(), {"reid": {"long_term": {"feature_momentum": 0.5}}}
    )
    assert lowered.reid.long_term.feature_momentum == pytest.approx(0.5)

    # 0,0 est légitime : la référence suit alors intégralement l'apparence courante.
    zero = override_config(
        load_config(), {"reid": {"long_term": {"feature_momentum": 0.0}}}
    )
    assert zero.reid.long_term.feature_momentum == pytest.approx(0.0)

    for invalid in (1.0, 1.5, -0.1):
        with pytest.raises(ConfigError) as error:
            override_config(
                load_config(),
                {"reid": {"long_term": {"feature_momentum": invalid}}},
            )
        assert "feature_momentum" in str(error.value)


def test_feature_momentum_reaches_the_identity_manager_from_the_configuration():
    """Le câblage config -> IdentityManager est effectif (jamais la valeur en dur)."""
    from identity_manager import IdentityManager

    config = override_config(
        load_config(), {"reid": {"long_term": {"feature_momentum": 0.4}}}
    )
    manager = IdentityManager(
        long_term=config.reid.long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
    )
    assert manager.feature_momentum == pytest.approx(0.4)
    # Une surcharge explicite du constructeur reste prioritaire (tests, ablations).
    assert IdentityManager(
        long_term=config.reid.long_term, feature_momentum=0.9
    ).feature_momentum == pytest.approx(0.9)


def test_freeze_gallery_frames_is_configurable_with_a_minimum_of_three():
    """Le gel du descripteur après une discontinuité est paramétrable (lot B.3)."""
    from config import AppearanceContinuityConfig

    config = load_config()
    assert config.reid.appearance_continuity.freeze_gallery_frames == 3

    raised = override_config(
        load_config(),
        {"reid": {"appearance_continuity": {"freeze_gallery_frames": 8}}},
    )
    assert raised.reid.appearance_continuity.freeze_gallery_frames == 8

    # Borne minimale 3 : en dessous, le gel serait plus court qu'une frame de
    # réaction au FPS mesuré (~3,5 FPS) et ne protégerait plus rien.
    for invalid in (0, 1, 2):
        with pytest.raises(ConfigError) as error:
            override_config(
                load_config(),
                {"reid": {"appearance_continuity": {"freeze_gallery_frames": invalid}}},
            )
        assert "freeze_gallery_frames" in str(error.value)

    assert AppearanceContinuityConfig().freeze_gallery_frames == 3


def test_descriptor_parameters_are_validated():
    """``reid.long_term.descriptor`` : valeurs par défaut et rejets explicites."""
    config = load_config()
    descriptor = config.reid.long_term.descriptor
    assert descriptor.horizontal_crop_ratio == pytest.approx(0.15)
    assert descriptor.bands == 3
    assert descriptor.band_weights == (1.0, 1.0, 0.5)
    assert descriptor.bins_hue == 24
    assert descriptor.bins_saturation == 8

    updated = override_config(
        load_config(),
        {
            "reid": {
                "long_term": {
                    "descriptor": {
                        "horizontal_crop_ratio": 0.25,
                        "bands": 4,
                        "band_weights": [1.0, 1.0, 0.5, 0.25],
                        "bins_hue": 12,
                        "bins_saturation": 4,
                    }
                }
            }
        },
    )
    assert updated.reid.long_term.descriptor.bands == 4
    assert updated.reid.long_term.descriptor.band_weights == (1.0, 1.0, 0.5, 0.25)


@pytest.mark.parametrize(
    "descriptor,expected",
    [
        ({"horizontal_crop_ratio": 0.5}, "horizontal_crop_ratio"),
        ({"horizontal_crop_ratio": -0.1}, "horizontal_crop_ratio"),
        ({"bands": 0}, "bands"),
        ({"bins_hue": 0}, "bins_hue"),
        ({"bands": 3, "band_weights": [1.0, 1.0]}, "band_weights"),
        ({"bands": 2, "band_weights": [0.0, 0.0]}, "band_weights"),
        ({"bands": 2, "band_weights": []}, "band_weights"),
        ({"inconnu": 1}, "inconnues"),
    ],
)
def test_descriptor_invalid_values_are_rejected(descriptor, expected):
    with pytest.raises(ConfigError) as error:
        override_config(
            load_config(), {"reid": {"long_term": {"descriptor": descriptor}}}
        )
    assert expected in str(error.value)


def test_progressive_similarity_floor_cannot_exceed_the_nominal_threshold():
    config = load_config()
    assert config.reid.long_term.long_absence_similarity_floor is None
    with pytest.raises(ConfigError) as error:
        override_config(
            load_config(),
            {"reid": {"long_term": {"long_absence_similarity_floor": 0.95}}},
        )
    assert "long_absence_similarity_floor" in str(error.value)


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


def test_default_config_has_head_assist_disabled():
    config = load_config()
    assert hasattr(config, "presence")
    assert config.presence.head_assist.enabled is False
    assert config.presence.head_assist.pose_model_path == "models/yolo11s-pose.pt"
    assert config.presence.head_assist.min_keypoint_confidence == pytest.approx(0.5)
    assert config.presence.head_assist.max_presence_extension_seconds == pytest.approx(10.0)
    assert config.model.device == "cpu"


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

