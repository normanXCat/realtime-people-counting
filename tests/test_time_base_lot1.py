"""Tests du lot 1 : base de temps unique (§1.0) et `track_buffer` en secondes (§1.1).

Ce que ces tests protègent :

- une mesure faite sur fichier doit décrire le même régime temporel que la
  production (base vidéo sur fichier, murale sur caméra) ;
- l'horizon du tracker doit être une durée de **scène**, pas un nombre de
  frames dont la durée dépend de la machine ;
- la conversion secondes → frames est faite **une seule fois**.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ConfigError, TimingConfig, load_config  # noqa: E402
from main import (  # noqa: E402
    apply_track_buffer,
    resolve_time_base,
    resolved_tracker_config,
    track_buffer_frames_from_seconds,
)


# --- §1.0 base de temps ----------------------------------------------------

def test_auto_suit_la_source():
    """« auto » : temps vidéo sur fichier, temps mural sur caméra."""
    assert resolve_time_base("auto", "test/fort_occ4.mp4") == "video"
    assert resolve_time_base("auto", 0) == "wall"


def test_bases_forcables_pour_investigation():
    """Forcer reste possible : c'est ce qui permet de comparer les deux régimes."""
    assert resolve_time_base("wall", "test/fort_occ4.mp4") == "wall"
    assert resolve_time_base("video", 0) == "video"


def test_time_base_invalide_refusee(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    base = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    base["timing"]["time_base"] = "horloge_murale"
    config_path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ConfigError) as error:
        load_config(str(config_path))
    assert "time_base" in str(error.value)


def test_budget_de_frames_en_base_video():
    """En base vidéo, une seconde vaut toujours fps_source frames.

    C'est le cœur du §1.0 : à 30 ips source, la grâce de 5 s vaut 150 frames,
    alors qu'au FPS de traitement mesuré (~3,8 ips) elle n'en valait que 19 —
    d'où des purges à des instants sans rapport avec la scène.
    """
    timing = TimingConfig(grace_period_seconds=5.0)
    assert timing.frame_budget(30.0).grace_period_frames == 150
    assert timing.frame_budget(3.8).grace_period_frames == 19


# --- §1.1 track_buffer en secondes ----------------------------------------

def test_track_buffer_converti_dans_une_copie(tmp_path):
    """La configuration versionnée n'est pas modifiée : la copie l'est."""
    source = tmp_path / "botsort.yaml"
    source.write_text(
        yaml.safe_dump({"tracker_type": "botsort", "track_buffer": 150, "match_thresh": 0.9}),
        encoding="utf-8",
    )
    session = tmp_path / "session"
    session.mkdir()

    resolved = resolved_tracker_config(source, session, 450)

    assert resolved == session / "tracker_resolved.yaml"
    contenu = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert contenu["track_buffer"] == 450
    # Les autres réglages du tracker sont reportés tels quels.
    assert contenu["match_thresh"] == 0.9
    assert yaml.safe_load(source.read_text(encoding="utf-8"))["track_buffer"] == 150


def test_conversion_secondes_vers_frames_selon_la_cadence():
    """Une seule conversion pour les deux sources : fichier et caméra.

    15 s valent 450 frames à 30 ips (fichier) et 52 à 3,5 ips (caméra) : le
    même réglage couvre la même durée de scène dans les deux cas, ce qui est
    tout l'objet du §1.1.
    """
    assert track_buffer_frames_from_seconds(15.0, 30.0) == 450
    assert track_buffer_frames_from_seconds(15.0, 3.5) == 52
    # Jamais zéro : un horizon nul tuerait toute piste à la frame suivante.
    assert track_buffer_frames_from_seconds(0.01, 1.0) == 1


class _FakeTracker:
    def __init__(self):
        self.max_time_lost = 150


class _FakePredictor:
    def __init__(self, trackers):
        self.trackers = trackers


class _FakeModel:
    def __init__(self, predictor=None):
        self.predictor = predictor


def test_ajustement_du_tracker_vivant():
    """Source caméra : l'horizon est corrigé en place, une fois."""
    tracker = _FakeTracker()
    model = _FakeModel(_FakePredictor([tracker]))

    assert apply_track_buffer(model, 53) is True
    assert tracker.max_time_lost == 53


def test_ajustement_impossible_signale_sans_planter():
    """Si Ultralytics ne s'y prête pas, la mesure est dégradée, pas le comptage."""
    assert apply_track_buffer(_FakeModel(None), 53) is False
    assert apply_track_buffer(_FakeModel(_FakePredictor([])), 53) is False
    assert apply_track_buffer(_FakeModel(_FakePredictor([object()])), 53) is False


def test_track_buffer_seconds_nul_reste_accepte(tmp_path):
    """Compatibilité ascendante : sans la clé secondes, la valeur en frames sert."""
    base = yaml.safe_load(Path("config/pipeline.yaml").read_text(encoding="utf-8"))
    base["tracker"]["track_buffer_seconds"] = None
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")

    config = load_config(str(config_path))

    assert config.tracker.track_buffer_seconds is None
    assert config.tracker.track_buffer == 150


def test_valeurs_publiees_dans_la_configuration_resolue():
    """config_resolved.yaml doit porter la base de temps et l'horizon en secondes."""
    config = load_config()
    resolu = config.to_dict()

    assert resolu["timing"]["time_base"] == "auto"
    assert resolu["tracker"]["track_buffer_seconds"] == 15.0
    assert resolu["tracker"]["fps_warm_up_frames"] == 60
