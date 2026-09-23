"""Chemin de secours « galerie » : rattacher une observation non trackée.

Symptôme visé : une personne assise et immobile dont la réapparition a une image
de moindre qualité (angle bas, mobilier) produit un score dans
``[tracker.track_low_thresh, tracker.new_track_thresh[``. BoT-SORT n'associe pas
cette détection et **ne crée aucun identifiant technique** : sans le chemin de
secours, ``IdentityManager`` ne la voit jamais et la personne reste indéfiniment
dans la borne haute de l'occupation.

Ce que ces tests verrouillent :

1. une observation non trackée, visuellement et spatialement cohérente avec une
   identité **déjà en galerie**, est rattachée (``REID_GALLERY_RESCUE``), et
   jamais promue en nouvelle identité ;
2. le chemin est **désactivé par défaut** et n'ouvre aucune tolérance
   supplémentaire : même seuil d'apparence, même marge de sécurité que le chemin
   nominal ; une observation ambiguë ou trop faible est ignorée ;
3. ``main`` ne filtre plus silencieusement ces détections : la gamme
   ``[track_low_thresh, new_track_thresh[`` est bien transmise quand le flag est
   actif, et rien ne l'est quand il ne l'est pas.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    TEST_FPS,
    StubAppearance,
    identity_vectors_by_x,
    make_test_line,
    person_box,
)
from config import (
    AppearanceContinuityConfig,
    LongTermReidConfig,
    SwapCorrectionConfig,
    load_config,
    override_config,
)
from events import ListSink
from identity_manager import IdentityManager, Observation
from occupancy_manager import Detection, OccupancyManager

FRAME = np.zeros((600, 600, 3), dtype=np.uint8)
INSIDE_Y = 450.0
OUTSIDE_Y = 250.0


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


def observation(
    track_id: int,
    *,
    anchor=(100.0, 300.0),
    feature=None,
    confidence: float = 0.5,
) -> Observation:
    box = np.asarray([80.0, 100.0, 120.0, 300.0], dtype=float)
    return Observation(
        technical_track_id=track_id,
        bbox=box,
        confidence=confidence,
        anchor=tuple(anchor),
        bbox_height=200.0,
        feature=None if feature is None else np.asarray(feature, dtype=np.float32),
    )


def _long_term(**overrides) -> LongTermReidConfig:
    base = dict(
        enabled=True,
        similarity_threshold=0.4,
        safety_margin=0.1,
        gallery_retention_seconds=10.0,
        gallery_rescue_enabled=True,
    )
    base.update(overrides)
    return LongTermReidConfig(**base)


@pytest.fixture
def sink() -> ListSink:
    return ListSink("gallery_rescue")


def _manager(sink: ListSink, long_term: LongTermReidConfig) -> IdentityManager:
    return IdentityManager(
        long_term=long_term,
        grace_period_seconds=5.0,
        appearance_continuity=AppearanceContinuityConfig(enabled=False),
        swap_correction=SwapCorrectionConfig(enabled=False),
        event_sink=sink,
    )


# ---------------------------------------------------------------------------
# 1. Rattachement d'une observation non trackée
# ---------------------------------------------------------------------------
def test_observation_non_trackee_est_rattachee_a_la_galerie(sink):
    """Confiance 0,5 : apparence cohérente -> même ``person_id``, aucun NEW."""
    manager = _manager(sink, _long_term())
    desc = unit(1, 0, 0)
    manager.assign([observation(7, feature=desc, confidence=0.9)], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)  # la piste est perdue, l'identité reste en galerie
    assert manager.person_of(7) == 1
    sink.events.clear()

    rescued = manager.try_gallery_rescue(
        observation(0, anchor=(104.0, 300.0), feature=desc, confidence=0.5),
        FRAME,
        1.0,
        3,
    )

    assert rescued is not None
    assert rescued.person_id == 1
    assert rescued.matched is True
    assert rescued.reason == "gallery_rescue"
    # Identifiant technique **synthétique** : négatif, donc jamais confondu avec
    # une piste BoT-SORT réelle.
    assert rescued.technical_track_id == -1
    assert manager.person_of(-1) == 1
    assert manager.gallery_rescues == 1
    # Jamais de nouvelle identité : le compteur de personnes n'a pas bougé.
    assert manager._next_person_id == 2
    assert manager.records[1].live is True

    events = sink.of_type("REID_GALLERY_RESCUE")
    assert len(events) == 1
    event = events[0]
    assert event["person_id"] == 1
    assert event["technical_track_id"] == -1
    assert event["previous_technical_track_id"] == 7
    assert event["confidence"] == pytest.approx(0.5)
    assert event["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert event["threshold_applied"] == pytest.approx(0.4)
    assert event["reason"] == "gallery_rescue_untracked_observation"
    # Événement dédié : mesurable séparément du chemin nominal.
    assert sink.count("REID_MATCH") == 0
    assert sink.count("REID_NEW") == 0
    # Un rattachement par secours n'est pas un changement d'identifiant BoT-SORT :
    # le compteur ``technical_id_changes`` reste réconciliable avec ses événements.
    assert manager.technical_id_changes == 0
    assert sink.count("TECHNICAL_ID_CHANGED") == 0


def test_seuil_et_marge_sont_ceux_du_chemin_nominal(sink):
    """Aucune tolérance supplémentaire : une apparence trop éloignée est ignorée."""
    manager = _manager(sink, _long_term(similarity_threshold=0.8))
    manager.assign([observation(7, feature=unit(1, 0, 0), confidence=0.9)], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)

    # Similarité ~0,7 < seuil 0,8 : le chemin nominal refuserait, le secours aussi.
    rescued = manager.try_gallery_rescue(
        observation(0, feature=unit(0.7, 0.713, 0.0), confidence=0.5), FRAME, 1.0, 3
    )
    assert rescued is None
    assert manager.gallery_rescues == 0
    assert sink.count("REID_GALLERY_RESCUE") == 0


def test_observation_ambigue_n_est_jamais_rattachee(sink):
    """Deux candidats équivalents : le secours s'abstient au lieu de trancher."""
    manager = _manager(sink, _long_term(similarity_threshold=0.4, safety_margin=0.1))
    probe = unit(1, 0, 0)
    # Deux identités créées dans la MÊME frame : la seconde ne peut pas être
    # réappariée à la première (elle est vivante et déjà réclamée), elle devient
    # donc une identité distincte, d'apparence très proche de la sonde.
    manager.assign(
        [
            observation(7, feature=probe, confidence=0.9),
            observation(9, feature=unit(0.96, 0.28, 0.0), confidence=0.9),
        ],
        FRAME,
        0.0,
        1,
    )
    assert manager.records[1].feature is not None and manager.records[2].feature is not None
    manager.assign([], FRAME, 0.1, 2)  # les deux identités passent en galerie

    rescued = manager.try_gallery_rescue(
        observation(0, feature=probe, confidence=0.5), FRAME, 1.0, 3
    )
    assert rescued is None
    assert sink.count("REID_GALLERY_RESCUE") == 0
    assert manager.gallery_rescues == 0


def test_plancher_de_confiance_refuse_les_observations_trop_faibles(sink):
    """``gallery_rescue_min_confidence`` refuse le bruit avant toute comparaison."""
    manager = _manager(sink, _long_term(gallery_rescue_min_confidence=0.6))
    manager.assign([observation(7, feature=unit(1, 0, 0), confidence=0.9)], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)

    assert manager.try_gallery_rescue(
        observation(0, feature=unit(1, 0, 0), confidence=0.5), FRAME, 1.0, 3
    ) is None
    assert manager.gallery_rescues == 0
    # Juste au-dessus du plancher : rattachée.
    rescued = manager.try_gallery_rescue(
        observation(0, feature=unit(1, 0, 0), confidence=0.61), FRAME, 1.1, 4
    )
    assert rescued is not None and rescued.person_id == 1


def test_chemin_desactive_par_defaut(sink):
    """Flag livré à ``false`` : aucune observation n'est rattachée."""
    assert load_config().reid.long_term.gallery_rescue_enabled is False

    manager = _manager(sink, _long_term(gallery_rescue_enabled=False))
    manager.assign([observation(7, feature=unit(1, 0, 0), confidence=0.9)], FRAME, 0.0, 1)
    manager.assign([], FRAME, 0.1, 2)

    assert manager.try_gallery_rescue(
        observation(0, feature=unit(1, 0, 0), confidence=0.5), FRAME, 1.0, 3
    ) is None
    assert manager.gallery_rescues == 0
    assert sink.count("REID_GALLERY_RESCUE") == 0


# ---------------------------------------------------------------------------
# 2. Câblage dans OccupancyManager : la personne revient dans « observée »
# ---------------------------------------------------------------------------
class Sim:
    def __init__(self, manager, frame=FRAME):
        self.manager = manager
        self.frame = frame
        self.time = 0.0
        self.index = 0

    def step(self, detections=(), untracked=()):
        self.index += 1
        self.time += 1.0 / TEST_FPS
        self.manager.process_frame(
            list(detections), self.frame, self.time, self.index, untracked=list(untracked)
        )
        return self

    def steps(self, count, detections=(), untracked=()):
        for _ in range(count):
            self.step(detections, untracked)
        return self


def det(track_id, y, x=150.0, height=200.0, confidence=0.9):
    return Detection(track_id, person_box(x, y, height=height), confidence)


def build(config, sink, appearance=None):
    identity = IdentityManager(
        long_term=config.reid.long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
        appearance_continuity=config.reid.appearance_continuity,
        event_sink=sink,
        appearance=appearance
        or StubAppearance(describe_fn=identity_vectors_by_x({150.0: (1.0, 0.0, 0.0)})),
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=make_test_line(inside_side=config.line.inside_side),
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


def _rescue_config(test_config):
    return override_config(
        test_config,
        {"reid": {"long_term": {"gallery_rescue_enabled": True}}},
    )


def test_une_personne_occultee_est_recuperee_par_la_galerie(test_config, sink):
    """Après une occultation courte, une observation non trackée la ramène."""
    config = _rescue_config(test_config)
    manager = build(config, sink)
    sim = Sim(manager).steps(5)

    sim.steps(2, [det(1, OUTSIDE_Y)])
    sim.steps(2, [det(1, INSIDE_Y)])
    assert manager.total_in == 1
    assert manager.occupancy_observed == 1
    sink.events.clear()

    sim.steps(4)  # occultation : plus aucune détection, dans la grâce
    assert manager.tracks[1].state.name == "OCCULTEE"
    assert manager.total_new == 0

    # Réapparition de moindre qualité : confiance 0,5 (sous new_track_thresh),
    # aucun identifiant technique. Le même endroit, la même apparence.
    sim.steps(3, untracked=[Detection(0, person_box(150.0, INSIDE_Y), 0.5)])

    # Une observation non trackée par frame tant que la détection reste faible :
    # chaque rattachement est journalisé, et le compteur suit exactement.
    assert sink.count("REID_GALLERY_RESCUE") >= 1
    assert sink.count("REID_GALLERY_RESCUE") == manager.identities.gallery_rescues
    assert sink.count("REID_MATCH") == 0
    assert sink.count("NEW") == 0
    assert manager.total_new == 0, "une personne déjà en galerie n'est jamais recréée"
    assert manager.total_in == 1
    assert manager.occupancy_operational == 1
    assert manager.occupancy_observed == 1
    assert manager.occupancy_uncertain == 0
    assert manager.tracks[1].state.name != "OCCULTEE", "la personne est de nouveau observée"
    assert manager.identities.gallery_rescues >= 1


def test_une_apparence_etrangere_n_est_jamais_rattachee(test_config, sink):
    """Une observation non trackée incohérente est ignorée : rien n'est créé."""
    config = _rescue_config(test_config)
    manager = build(config, sink)
    sim = Sim(manager).steps(5)
    sim.steps(2, [det(1, OUTSIDE_Y)])
    sim.steps(2, [det(1, INSIDE_Y)])
    assert manager.total_in == 1
    person_id = manager.identities.person_of(1)
    sink.events.clear()
    sim.steps(4)

    # Personne jamais vue, à l'autre bout de la scène, confiance intermédiaire.
    sim.steps(4, untracked=[Detection(0, person_box(520.0, INSIDE_Y), 0.5)])

    assert sink.count("REID_GALLERY_RESCUE") == 0
    assert sink.count("REID_NEW") == 0
    assert manager.total_new == 0
    assert manager.total_in == 1
    assert manager.occupancy_operational == 1
    assert manager.identities.person_of(1) == person_id


def test_sans_rescue_les_observations_non_trackees_sont_ignorees(test_config, sink):
    """Ablation : avec le flag à ``false``, rien ne change (comportement actuel)."""
    config = override_config(
        test_config,
        {"reid": {"long_term": {"gallery_rescue_enabled": False}}},
    )
    manager = build(config, sink)
    sim = Sim(manager).steps(5)
    sim.steps(2, [det(1, OUTSIDE_Y)])
    sim.steps(2, [det(1, INSIDE_Y)])
    sim.steps(4)
    sim.steps(4, untracked=[Detection(0, person_box(150.0, INSIDE_Y), 0.5)])

    assert sink.count("REID_GALLERY_RESCUE") == 0
    assert manager.tracks[1].state.name == "OCCULTEE"
    assert manager.occupancy_operational == 1
    assert manager.total_new == 0


# ---------------------------------------------------------------------------
# 3. `main` ne filtre plus silencieusement la gamme intermédiaire
# ---------------------------------------------------------------------------
def _boxes(rows, *, with_ids):
    from test_track_diagnostics import _FakeBoxes

    return _FakeBoxes(rows, with_ids=with_ids)


def test_main_transmet_la_gamme_track_low_new_track():
    from main import collect_untracked_detections

    config = override_config(
        load_config(), {"reid": {"long_term": {"gallery_rescue_enabled": True}}}
    )
    boxes = _boxes(
        [
            (0, [10, 10, 20, 60], 0.5),   # dans la gamme : transmise
            (0, [30, 10, 40, 60], 0.05),  # sous track_low_thresh : non
            (0, [50, 10, 60, 60], 0.85),  # au-dessus de new_track_thresh : non
        ],
        with_ids=False,
    )

    collected = collect_untracked_detections(boxes, config, ids_present=False)

    assert len(collected) == 1
    assert collected[0].confidence == pytest.approx(0.5)
    # Aucun identifiant n'est inventé : placeholder non tracké.
    assert collected[0].technical_track_id == 0


def test_main_ne_transmet_rien_quand_le_secours_est_desactive():
    from main import collect_untracked_detections

    config = load_config()
    assert config.reid.long_term.gallery_rescue_enabled is False
    boxes = _boxes([(0, [10, 10, 20, 60], 0.5)], with_ids=False)

    assert collect_untracked_detections(boxes, config, ids_present=False) == []


def test_main_ne_transmet_rien_quand_des_pistes_existent():
    from main import collect_untracked_detections

    config = override_config(
        load_config(), {"reid": {"long_term": {"gallery_rescue_enabled": True}}}
    )
    boxes = _boxes([(7, [10, 10, 20, 60], 0.5)], with_ids=True)

    assert collect_untracked_detections(boxes, config, ids_present=True) == []
    assert collect_untracked_detections(None, config, ids_present=False) == []
