"""Intégration **réellement appliquée** de ``BBoxHeightLocker`` (spec 7).

Ces tests ne testent pas la logique interne du verrouillage de hauteur : ils
verrouillent son **câblage** dans le chemin d'exécution réel.

1. La clé d'historique est le ``person_id`` stable résolu par
   :mod:`identity_manager`, jamais l'identifiant technique de BoT-SORT : un
   ``TECHNICAL_ID_CHANGED`` (occultation, flou, perte de piste) ne doit pas
   réinitialiser la hauteur de référence.
2. La purge des profils se fait sur les **personnes** encore suivies
   (``OccupancyManager.stabilization_keys()``), pas sur les identifiants
   présents dans la frame : une seule frame manquée ne doit pas effacer la
   correction.
3. La chute de hauteur est mesurée sur la boîte **brute** et signalée
   (``ANCHOR_UNRELIABLE``, prioritaire dans la table de transitions), sans
   produire de fausse transition de zone ni de fausse sortie.
4. Le seuil appliqué vient de la configuration (``bbox_locker_min_height_ratio``),
   jamais d'une valeur écrite en dur dans ``main.py``.
5. Un parcours complet de ``main`` écrit réellement l'événement dans
   ``events.jsonl`` — preuve que le composant est actif dans le pipeline officiel.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from bbox_height_locker import BBoxHeightLocker
from conftest import (
    TEST_FPS,
    StubAppearance,
    identity_vectors_by_x,
    make_test_line,
    person_box,
)
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

INSIDE_Y = 450.0
#: Ancre brute d'une personne dont les jambes sont masquées : elle est passée
#: **au-dessus** de la ligne de test (y = 300), donc côté extérieur.
OCCLUDED_RAW_Y = 250.0
VECTORS = {150.0: (1.0, 0.0, 0.0), 300.0: (0.0, 1.0, 0.0)}


# ---------------------------------------------------------------------------
# Outillage
# ---------------------------------------------------------------------------
def det(track_id, y, x=150.0, confidence=0.9, height=200.0):
    return Detection(track_id, person_box(x, y, height=height), confidence)


class Sim:
    def __init__(self, manager, frame, locker=None):
        self.manager = manager
        self.frame = frame
        self.locker = locker
        self.time = 0.0
        self.index = 0

    def step(self, detections=()):
        self.index += 1
        self.time += 1.0 / TEST_FPS
        self.manager.process_frame(list(detections), self.frame, self.time, self.index)
        self.purge()
        return self

    def steps(self, count, detections=()):
        for _ in range(count):
            self.step(detections)
        return self

    def purge(self):
        """Applique la purge telle que ``main`` la déclenche après chaque frame."""
        if self.locker is not None:
            self.locker.purge_lost_tracks(self.manager.stabilization_keys())


def build(config, sink, locker=None, appearance=None):
    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=appearance or StubAppearance(describe_fn=identity_vectors_by_x(VECTORS)),
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=make_test_line(inside_side=config.line.inside_side),
        bbox_locker=locker,
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


def make_locker(config) -> BBoxHeightLocker:
    """Verrouillage de hauteur construit **depuis la configuration**, comme ``main``."""
    return BBoxHeightLocker(
        min_height_ratio=config.stabilization.bbox_locker_min_height_ratio
    )


def enter(sim, track_id=77, x=150.0):
    """Franchissement propre : la personne est comptée présente à l'intérieur."""
    sim.steps(2, [det(track_id, 250.0, x=x)])     # côté extérieur
    sim.steps(2, [det(track_id, INSIDE_Y, x=x)])  # franchissement entrant
    return sim


# ---------------------------------------------------------------------------
# 1. Clé d'historique : person_id, jamais l'identifiant technique
# ---------------------------------------------------------------------------
def test_history_is_keyed_by_person_id_not_by_technical_id(test_config, sink, frame):
    """L'identifiant technique (77) n'apparaît pas : la clé est le ``person_id`` (1)."""
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)

    sim.steps(4)                      # warm-up (aucune détection)
    sim.steps(2, [det(77, INSIDE_Y)])  # piste technique 77 -> person_id 1

    assert 1 in locker.tracks_profile
    assert 77 not in locker.tracks_profile
    assert 1 in manager.tracks


def test_a_frame_purge_on_volatile_ids_would_destroy_the_reference(
    test_config, sink, frame
):
    """Politique précédente : purger sur les identifiants de la frame courante.

    Une seule frame manquée suffisait à perdre la hauteur de référence.
    """
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)
    sim.steps(4)
    sim.steps(2, [det(77, INSIDE_Y)])
    assert 1 in locker.tracks_profile

    # ``main`` purgeait avec les identifiants **de la frame** : la piste 77 est
    # momentanément absente, donc le profil de la personne 1 était supprimé.
    locker.purge_lost_tracks(set())
    assert 1 not in locker.tracks_profile


def test_one_missed_frame_does_not_erase_the_height_reference(test_config, sink, frame):
    """Purge indexée sur les personnes suivies : l'occultation ne coûte rien."""
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)
    sim.steps(4)
    sim.steps(4, [det(77, INSIDE_Y)])

    sim.step()  # frame sans aucune détection (occultation courte)
    assert manager.stabilization_keys() == {1}
    assert 1 in locker.tracks_profile
    assert locker.tracks_profile[1]["h_stable"] == pytest.approx(200.0)


def test_the_profile_is_released_once_the_person_is_gone(test_config, sink, frame):
    """Aucune fuite : le profil disparaît quand la personne quitte le suivi.

    L'occupation, elle, reste tenue par le registre opérationnel : libérer un
    cache n'est pas constater une sortie.
    """
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)
    sim.steps(4)
    sim.steps(6, [det(77, INSIDE_Y)])
    assert 1 in locker.tracks_profile

    # La personne n'est libérée du suivi qu'après la FENÊTRE de galerie réelle
    # (grâce + rétention, dérivée de la configuration) : depuis le lot A.2 la
    # rétention vaut 12 s et non plus la grâce — un littéral de frames mentirait.
    window_frames = int(
        (
            manager.identities.grace_period_seconds
            + manager.identities.purge_retention_seconds
        )
        * TEST_FPS
    ) + 2
    sim.steps(window_frames)

    assert manager.stabilization_keys() == set()
    assert locker.tracks_profile == {}
    assert manager.occupancy_operational == 1


# ---------------------------------------------------------------------------
# 2. Survie à l'occultation ET au changement d'identifiant technique
# ---------------------------------------------------------------------------
def test_history_survives_occlusion_and_a_technical_id_change(test_config, sink, frame):
    """Scénario du prompt : hauteur stable, occultation, nouvel ID technique, chute.

    Résultat attendu : la hauteur de référence (200 px) est conservée, donc la
    boîte tronquée est corrigée sur la base de la personne debout — et non de la
    boîte déjà effondrée.
    """
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)

    sim.steps(4)                     # warm-up
    enter(sim, track_id=77)          # entrée franche, hauteur stable 200 px
    assert manager.total_in == 1
    assert locker.tracks_profile[1]["h_stable"] == pytest.approx(200.0)
    sink.events.clear()

    sim.steps(4)                     # piste perdue (occultation)
    assert 1 in locker.tracks_profile, "l'occultation ne doit pas effacer l'historique"

    # Réapparition : nouvel identifiant technique (42), même personne (ReID),
    # jambes masquées -> boîte de 100 px dont le bas brut est côté extérieur.
    sim.steps(3, [det(42, OCCLUDED_RAW_Y, x=155.0, height=100.0)])

    changes = sink.of_type("TECHNICAL_ID_CHANGED")
    assert len(changes) == 1
    assert changes[0]["person_id"] == 1
    assert changes[0]["technical_track_id"] == 42

    stabilization = sink.of_type("STABILIZATION")
    assert stabilization, "la boîte corrigée doit être tracée"
    first = stabilization[0]
    assert first["component"] == "bbox_locker"
    assert first["raw_bbox"][3] == pytest.approx(150.0 + 100.0)
    # 150 (y1) + 200 (hauteur debout conservée) : l'historique n'a pas été perdu.
    assert first["corrected_bbox"][3] == pytest.approx(350.0)

    assert 1 in locker.tracks_profile
    assert manager.total_out == 0, "une occultation n'est pas une sortie"
    assert manager.occupancy_operational == 1
    assert manager.tracks[1].anchor[1] == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# 3. Chute de hauteur : signal « ancre non fiable » + aucune fausse transition
# ---------------------------------------------------------------------------
def test_sudden_height_drop_signals_anchor_unreliable_with_the_locker_on(
    test_config, sink, frame
):
    """Le signal consommé par la FSM doit rester atteignable quand le locker agit."""
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)

    sim.steps(4)
    enter(sim, track_id=77)
    sink.events.clear()

    sim.step([det(77, OCCLUDED_RAW_Y, height=100.0)])

    events = sink.of_type("ANCHOR_UNRELIABLE")
    assert len(events) == 1, "la chute mesurée sur la boîte brute doit être signalée"
    event = events[0]
    assert event["reason"] == "height_drop"
    assert event["person_id"] == 1
    assert event["previous_height"] == pytest.approx(200.0)
    assert event["current_height"] == pytest.approx(100.0)
    assert event["drop_ratio"] == pytest.approx(0.5)
    assert manager.tracks[1].state.name == "PRESENTE"


def test_height_drop_alone_never_produces_a_false_exit(test_config, sink, frame):
    """Chute de hauteur sans franchissement : l'occupation ne bouge pas."""
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)

    sim.steps(4)
    enter(sim, track_id=77)
    sim.steps(8, [det(77, OCCLUDED_RAW_Y, height=100.0)])

    assert manager.total_out == 0
    assert sink.count("OUT") == 0
    assert manager.occupancy_operational == 1
    assert manager.occupancy_confirmed == 1
    assert manager.tracks[1].state.name == "PRESENTE"


def test_without_the_locker_the_same_scenario_produces_a_false_exit(
    test_config, sink, frame
):
    """Comparaison avant/après sur le scénario identique (preuve du correctif).

    Sans verrouillage, la chute de hauteur fait remonter l'ancre pieds
    au-dessus de la ligne : la FSM confirme une **sortie** alors que la
    personne est toujours dans la salle.
    """
    manager = build(test_config, sink)  # aucun verrouillage : comportement d'avant
    sim = Sim(manager, frame)
    sim.steps(4)
    enter(sim, track_id=77)
    sim.steps(8, [det(77, OCCLUDED_RAW_Y, height=100.0)])

    assert manager.total_out == 1, "sans correction, la chute est interprétée comme une sortie"


def test_the_locker_prevents_the_same_false_exit(test_config, sink, frame):
    """Même scénario, verrouillage actif : aucune sortie, ancre debout conservée."""
    locker = make_locker(test_config)
    manager = build(test_config, sink, locker=locker)
    sim = Sim(manager, frame, locker=locker)
    sim.steps(4)
    enter(sim, track_id=77)
    sim.steps(8, [det(77, OCCLUDED_RAW_Y, height=100.0)])

    assert manager.total_out == 0
    assert manager.occupancy_operational == 1
    # L'ancre reste celle de la personne debout (bas de la boîte corrigée).
    assert manager.tracks[1].anchor[1] == pytest.approx(350.0)
    assert manager.tracks[1].last_zone == "INTERIEURE"


# ---------------------------------------------------------------------------
# 4. Le seuil vient de la configuration
# ---------------------------------------------------------------------------
def test_the_threshold_comes_from_the_configuration(make_config, sink, frame):
    """Deux seuils différents sur le même scénario : la décision suit la config."""
    scenario = [det(77, INSIDE_Y - 40.0, height=160.0)]  # 160 / 200 = 0,8

    def run(ratio):
        config = make_config({"stabilization": {"bbox_locker_min_height_ratio": ratio}})
        locker = make_locker(config)
        manager = build(config, sink, locker=locker)
        sim = Sim(manager, frame, locker=locker)
        sim.steps(4)
        sim.steps(6, [det(77, INSIDE_Y, height=200.0)])
        sink.events.clear()
        sim.step(scenario)
        return config, locker, sink.of_type("STABILIZATION")

    config, locker, events = run(0.90)
    assert locker.min_height_ratio == pytest.approx(config.stabilization.bbox_locker_min_height_ratio)
    assert len(events) == 1
    assert events[0]["threshold"] == pytest.approx(0.90)

    _, _, events_without = run(0.70)
    assert events_without == [], "0,8 > 0,70 : aucune correction attendue"


# ---------------------------------------------------------------------------
# 5. Preuve de bout en bout : l'événement est écrit dans events.jsonl
# ---------------------------------------------------------------------------
FRAME_SIDE = 600


def _scripted_detections() -> list[list[tuple[int, list[float], float]]]:
    """Extérieur -> intérieur (IN), puis jambes masquées (hauteur 150 -> 90)."""
    outside = [240.0, 50.0, 320.0, 200.0]      # hauteur 150, ancre brute 200 (dehors)
    inside = [240.0, 300.0, 320.0, 450.0]      # hauteur 150, ancre brute 450 (dedans)
    occluded = [240.0, 120.0, 320.0, 210.0]    # hauteur 90 : chute de 40 %
    script: list[list[tuple[int, list[float], float]]] = []
    script.extend([[(1, outside, 0.9)]] * 20)
    script.extend([[(1, inside, 0.9)]] * 20)
    script.extend([[(1, occluded, 0.9)]] * 12)
    return script


def _install_scripted_ultralytics(monkeypatch, script) -> None:
    """Double d'``ultralytics`` rendant une liste de détections **par frame**."""
    from test_main_e2e import _FakeResult

    module = types.ModuleType("ultralytics")
    image = np.zeros((FRAME_SIDE, FRAME_SIDE, 3), dtype=np.uint8)

    class _Model:
        def __init__(self, path):
            self.path = path

        def track(self, **_kwargs):
            for detections in script:
                yield _FakeResult(image, detections)

    module.YOLO = _Model
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def _fast_config(tmp_path: Path) -> Path:
    """Copie de la **vraie** configuration livrée, avec un warm-up raccourci."""
    data = yaml.safe_load((REPO_ROOT / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
    data["timing"]["warmup_seconds"] = 0.001
    data["timing"]["warmup_min_frames"] = 2
    data["timing"]["confirmation_seconds"] = 0.001
    data["timing"]["grace_period_seconds"] = 0.02
    data["timing"]["fps_estimate_window"] = 2
    data["display"]["enabled"] = False
    path = tmp_path / "pipeline_fast.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_end_to_end_occlusion_writes_anchor_unreliable_in_events_jsonl(
    tmp_path, monkeypatch
):
    """Le pipeline complet, lancé avec la configuration livrée, journalise la chute."""
    import main as entry_point
    from events import read_events
    from test_main_e2e import validated_line

    config_path = _fast_config(tmp_path)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["stabilization"][
        "use_bbox_locker"
    ] is True

    # Le verrouillage est construit **depuis la configuration** : seuil de chute
    # et fenêtre de lissage (celle de l'ancre, jamais un second seuil en dur).
    from config import load_config

    expected = load_config(config_path)
    captured_locker_kwargs: dict = {}
    real_locker = entry_point.BBoxHeightLocker

    def _recording_locker(**kwargs):
        captured_locker_kwargs.update(kwargs)
        return real_locker(**kwargs)

    monkeypatch.setattr(entry_point, "BBoxHeightLocker", _recording_locker)
    monkeypatch.setattr(
        entry_point,
        "perform_line_selection",
        lambda config, source, **kwargs: validated_line(),
    )
    _install_scripted_ultralytics(monkeypatch, _scripted_detections())

    code = entry_point.main(
        [
            "--config", str(config_path),
            "--source", "clip.mp4",
            "--output-root", str(tmp_path),
            "--session-id", "session-locker",
            "--no-show",
        ]
    )
    assert code == 0

    session = tmp_path / "session-locker"
    events = read_events(session / "events.jsonl")
    unreliable = [event for event in events if event["type"] == "ANCHOR_UNRELIABLE"]
    assert unreliable, "le verrouillage doit signaler la chute de hauteur"
    assert unreliable[0]["reason"] == "height_drop"
    assert unreliable[0]["person_id"] == 1
    assert unreliable[0]["previous_height"] == pytest.approx(150.0)
    assert unreliable[0]["current_height"] == pytest.approx(90.0)
    assert unreliable[0]["drop_ratio"] == pytest.approx(0.4)

    stabilization = [event for event in events if event["type"] == "STABILIZATION"]
    assert stabilization, "le composant doit être réellement instancié par main"
    assert stabilization[0]["component"] == "bbox_locker"
    # Boîte brute (210) corrigée sur la base de la hauteur debout conservée (150).
    assert stabilization[0]["raw_bbox"][3] == pytest.approx(210.0)
    assert stabilization[0]["corrected_bbox"][3] == pytest.approx(270.0)

    assert not [event for event in events if event["type"] == "OUT"]
    assert captured_locker_kwargs["history_window"] == (
        expected.anchor.height_history_window_frames
    )
    assert captured_locker_kwargs["min_height_ratio"] == pytest.approx(
        expected.stabilization.bbox_locker_min_height_ratio
    )
    summary = json.loads((session / "summary.json").read_text(encoding="utf-8"))
    assert summary["counters"]["total_in"] == 1
    assert summary["counters"]["total_out"] == 0
    assert summary["counters"]["occupancy_operational"] == 1
