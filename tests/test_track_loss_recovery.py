"""Niveau 3 — Perte de piste par flou et récupération avec un nouvel ID.

Le mode de défaillance visé (spec 2 à 5 du prompt) est une chaîne :

    personne légèrement floue
      -> confiance YOLO plus basse, boîte moins stable
      -> BoT-SORT perd la piste technique
      -> nouvelle piste, ou piste absente
      -> risque de OCCULTEE incorrect, de NEW en double, de double occupation

Ces tests vérifient que la chaîne est **traitée** et non contournée :

- une confiance faible ne suffit pas à perdre une personne ni à polluer la
  galerie d'apparences ;
- une absence de quelques frames ne diminue pas l'occupation ;
- un nouvel identifiant technique est réassocié au **même** ``person_id``, et
  le changement est journalisé (``TECHNICAL_ID_CHANGED``) ;
- quand la réassociation est impossible, aucune apparition intérieure n'est
  confirmée tant qu'une personne comptée est occultée à proximité
  (``new_deferred_occluded_person_nearby``) ;
- une apparence non exploitable est refusée explicitement
  (``REID_DESCRIPTOR_REJECTED``) au lieu de remplacer une identité connue.
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
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

INSIDE_Y = 450.0
OUTSIDE_Y = 250.0
#: Descripteurs indexés par position horizontale : chaque personne garde sa
#: signature, la galerie peut donc fonctionner normalement.
VECTORS = {
    150.0: (1.0, 0.0, 0.0),
    300.0: (0.0, 1.0, 0.0),
    450.0: (0.0, 0.0, 1.0),
}


class Sim:
    def __init__(self, manager, frame, dt=1.0 / TEST_FPS):
        self.manager = manager
        self.frame = frame
        self.dt = dt
        self.time = 0.0
        self.index = 0

    def step(self, detections=()):
        self.index += 1
        self.time += self.dt
        self.manager.process_frame(list(detections), self.frame, self.time, self.index)
        return self

    def steps(self, count, detections=()):
        for _ in range(count):
            self.step(detections)
        return self

    def warmup(self):
        return self.steps(5)


def det(track_id, y, x=300.0, confidence=0.9, height=200.0):
    return Detection(track_id, person_box(x, y, height=height), confidence)


def build(config, sink, appearance=None, identity=None):
    identity = identity or IdentityManager(
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
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


@pytest.fixture
def sim(test_config, sink, frame):
    return Sim(build(test_config, sink), frame)


def enter(manager_sim, track_id=1, x=150.0, confidence=0.9):
    """Franchissement propre : la personne est comptée présente."""
    sim = manager_sim
    sim.steps(2, [det(track_id, OUTSIDE_Y, x=x, confidence=confidence)])
    sim.steps(2, [det(track_id, INSIDE_Y, x=x, confidence=confidence)])
    return sim


# ---------------------------------------------------------------------------
# Confiance faible mais boîte présente
# ---------------------------------------------------------------------------
def test_low_confidence_box_keeps_the_person_counted(sim, sink):
    """0,15 est au-dessus de ``model.confidence`` (0,10) : la piste tient."""
    manager = sim.manager
    sim.warmup()
    enter(sim, confidence=0.15)
    assert manager.total_in == 1
    assert manager.occupancy_confirmed == 1

    sim.steps(3, [det(1, INSIDE_Y, x=150.0, confidence=0.15)])
    assert manager.total_out == 0
    assert manager.occupancy_operational == 1
    assert manager.tracks[1].state.name == "PRESENTE"


def test_low_confidence_descriptor_is_rejected_and_reported(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim, confidence=0.15)
    rejections = sink.of_type("REID_DESCRIPTOR_REJECTED")
    assert rejections, "une apparence faible ne doit pas entrer en galerie"
    assert rejections[0]["reason"] == "low_confidence"
    assert manager.identities.records[1].feature is None


def test_confidence_defaults_stay_coherent(sim):
    """Le seuil YOLO ne peut pas être plus haut que le seuil bas du tracker.

    Ultralytics filtre avec ``conf=model.confidence`` **avant** BoT-SORT : tout
    score sous ce seuil n'existe plus pour l'association. Pour que le second
    étage de BoT-SORT (gamme ``[track_low_thresh, track_high_thresh[``) soit
    réellement utilisable, il faut donc ``model.confidence <= track_low_thresh``.
    Les valeurs livrées sont **égales** (0,10), ce qui satisfait les deux
    lectures de la contrainte et laisse toute la gamme faible au tracker.
    """
    from config import coherence_warnings, load_config

    config = load_config()
    assert config.model.confidence <= config.tracker.track_low_thresh
    assert config.tracker.track_low_thresh <= config.model.confidence
    assert config.tracker.track_low_thresh < config.tracker.track_high_thresh
    assert config.tracker.new_track_thresh >= config.tracker.track_high_thresh
    codes = [code for code, _message in coherence_warnings(config)]
    assert "yolo_confidence_above_tracker_low_thresh" not in codes


# ---------------------------------------------------------------------------
# Seuil d'inférence : ce qui atteint réellement BoT-SORT
# ---------------------------------------------------------------------------
def inference_gate(config, detections):
    """Reproduit le filtrage qu'Ultralytics applique avant BoT-SORT (``conf=``)."""
    return [item for item in detections if item.confidence >= config.model.confidence]


def test_box_above_the_inference_gate_reaches_the_pipeline(test_config, sink, frame):
    """Juste au-dessus du seuil : la boîte est transmise et la piste se crée."""
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    gate = test_config.model.confidence
    weak = det(1, INSIDE_Y, x=450.0, confidence=gate + 0.01)
    assert inference_gate(test_config, [weak]) == [weak]

    sim.steps(3, inference_gate(test_config, [weak]))
    assert 1 in manager.tracks, "une détection faible exploitable doit créer la piste"
    assert manager.total_new == 1
    assert manager.occupancy_operational == 1


def test_box_below_the_inference_gate_is_filtered_before_the_tracker(
    test_config, sink, frame
):
    """Sous le seuil : rien n'atteint le tracker, donc aucune fausse piste."""
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    gate = test_config.model.confidence
    noise = det(1, INSIDE_Y, x=450.0, confidence=gate - 0.01)
    assert inference_gate(test_config, [noise]) == []

    sim.steps(6, inference_gate(test_config, [noise]))
    assert manager.tracks == {}, "aucune piste ne doit être créée sous le seuil"
    assert manager.total_in == 0 and manager.total_new == 0
    assert manager.occupancy_operational == 0
    assert sink.count("NEW") == 0 and sink.count("OCCLUDED") == 0


def test_progressive_blur_never_breaks_the_track_above_the_gate(test_config, sink, frame):
    """Flou progressif : la piste tient tant que le score reste exploitable."""
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    enter(sim)
    assert manager.occupancy_operational == 1

    for confidence in (0.9, 0.6, 0.45, 0.3, 0.2, 0.15, 0.12, 0.11):
        step = inference_gate(test_config, [det(1, INSIDE_Y, x=150.0, confidence=confidence)])
        assert step, f"la détection à {confidence} doit franchir le seuil d'inférence"
        sim.steps(1, step)
        assert manager.tracks[1].state.name != "OCCULTEE", (
            f"la piste ne doit pas se casser à confiance {confidence}"
        )

    assert sink.count("OCCLUDED") == 0
    assert manager.total_in == 1
    assert manager.occupancy_operational == 1


# ---------------------------------------------------------------------------
# Les garde-fous anti-faux-positifs ne dépendent pas du seuil de confiance
# ---------------------------------------------------------------------------
def test_a_single_weak_detection_is_not_counted(test_config, sink, frame):
    """Un éclair de détection faible ne compte pas : la confirmation reste due."""
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    blip = det(9, INSIDE_Y, x=450.0, confidence=0.12)
    sim.step(inference_gate(test_config, [blip]))
    assert manager.total_new == 0
    assert manager.total_in == 0
    assert manager.occupancy_operational == 0
    person_id = manager.identities.person_of(9)
    assert person_id is not None, "la piste existe, mais rien n'est encore confirmé"
    assert manager.tracks[person_id].state.name == "EXTERIEUR"
    assert manager.tracks[person_id].inside_occupancy is False


def test_new_track_threshold_stays_strict_despite_the_low_yolo_gate(test_config):
    """Le garde-fou de création de piste reste bien au-dessus du seuil YOLO."""
    from track_diagnostics import BOXES_PRESENT

    assert test_config.model.confidence < test_config.tracker.track_high_thresh
    assert test_config.tracker.new_track_thresh > test_config.tracker.track_high_thresh
    assert test_config.tracker.new_track_thresh > test_config.model.confidence
    # Le filtre de détection ne fait que *laisser passer* : il ne crée rien.
    assert BOXES_PRESENT == "present"


def test_aberrant_box_is_still_suspended_at_low_confidence(test_config, sink, frame):
    """Une boîte aberrante (bord d'image) reste suspendue, même à confiance faible."""
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    truncated = Detection(1, np.array([0.0, 250.0, 80.0, 450.0]), 0.12)
    sim.steps(4, inference_gate(test_config, [truncated]))
    assert sink.of_type("ANCHOR_UNRELIABLE")[0]["reason"] == "edge_margin"
    assert manager.total_in == 0 and manager.total_new == 0
    assert manager.occupancy_operational == 0
    # Le refus de descripteur est explicite, quelle que soit la raison retenue
    # en premier (ici low_confidence avant edge_truncated : un score faible est
    # déjà rédhibitoire pour la galerie).
    reasons = {event["reason"] for event in sink.of_type("REID_DESCRIPTOR_REJECTED")}
    assert reasons
    assert reasons <= {
        "low_confidence", "tiny_crop", "edge_truncated", "blurred_or_unreliable",
    }
    assert manager.identities.records[manager.identities.person_of(1)].feature is None


# ---------------------------------------------------------------------------
# Absence de boîte pendant quelques frames
# ---------------------------------------------------------------------------
def test_missing_boxes_for_a_few_frames_do_not_decrease_occupancy(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    assert manager.occupancy_operational == 1

    sim.steps(4)  # boîtes absentes, dans la fenêtre de grâce (1,0 s = 10 frames)
    assert manager.total_out == 0
    assert sink.count("OUT") == 0
    assert manager.occupancy_operational == 1
    assert manager.tracks[1].state.name == "OCCULTEE"
    assert sink.count("OCCLUDED") == 1


def test_missing_boxes_beyond_grace_keep_the_person_in_the_ledger(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    sim.steps(20)  # au-delà de la grâce : purge, puis libération de galerie
    assert manager.total_out == 0
    assert sink.count("OUT") == 0
    # Ni la purge ni l'oubli de la galerie ne font baisser l'occupation.
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_observed == 0
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_operational == 1
    assert manager.occupancy_range == (0, 1)
    purge = sink.of_type("PURGE")[0]
    assert purge["purge_kind"] == "technical"
    assert purge["is_exit"] is False


# ---------------------------------------------------------------------------
# Nouvel identifiant technique après occultation
# ---------------------------------------------------------------------------
def test_new_technical_id_is_reassociated_and_traced(sim, sink):
    manager = sim.manager
    sim.warmup()
    enter(sim)
    assert manager.total_in == 1

    sim.steps(4)                                  # occultation courte
    sim.steps(3, [det(42, INSIDE_Y, x=155.0)])    # nouvel ID technique, même personne

    assert manager.total_in == 1, "aucun double comptage"
    assert manager.total_new == 0, "aucun faux NEW"
    assert manager.occupancy_confirmed == 1
    assert manager.occupancy_operational == 1
    assert sink.count("REID_MATCH") == 1
    changes = sink.of_type("TECHNICAL_ID_CHANGED")
    assert len(changes) == 1
    assert changes[0]["previous_technical_track_id"] == 1
    assert changes[0]["technical_track_id"] == 42
    assert changes[0]["person_id"] == 1
    assert changes[0]["decision"] == "reassigned_to_existing_person"
    assert changes[0]["similarity"] is not None
    assert changes[0]["distance"] is not None


def test_blurred_person_seen_outside_then_inside_is_counted_once(
    test_config, sink, frame
):
    """Le scénario de la spec 4 : vu dehors, flou, piste perdue, piste intérieure.

    Attendu : pas de double NEW, pas de double IN, pas de double occupation — la
    personne est comptée une fois et une seule, quelle que soit la branche
    empruntée (réassociation au même ``person_id``, ou nouvelle identité dont
    l'entrée est alors confirmée).
    """
    appearance = StubAppearance(describe_fn=identity_vectors_by_x(VECTORS))
    manager = build(test_config, sink, appearance=appearance)
    sim = Sim(manager, frame).warmup()

    # P1 est observée à l'extérieur, deux frames.
    sim.steps(2, [det(1, OUTSIDE_Y, x=150.0)])
    # Elle devient floue : la piste technique est perdue (aucune détection).
    sim.steps(2)
    # Une nouvelle piste apparaît à l'intérieur, au même endroit.
    sim.steps(6, [det(7, INSIDE_Y, x=150.0)])

    assert manager.total_in + manager.total_new == 1, "un seul comptage d'entrée"
    assert not (manager.total_in and manager.total_new), "jamais IN et NEW à la fois"
    assert manager.total_out == 0
    assert manager.occupancy_operational == 1
    assert manager.occupancy_range == (1, 1)
    assert sink.count("NEW") + sink.count("IN") == 1


def _flip_vector_appearance(described: dict, key: float = 150.0, tolerance: float = 30.0):
    """Apparence d'une seule personne, modifiable en cours de test.

    Modélise une personne dont l'apparence change après l'occultation (flou,
    changement d'éclairage, vêtement) : la réassociation par similarité devient
    impossible et la contrainte spatio-temporelle est saturée par le déplacement.
    """

    def describe(_frame, bbox):
        x_center = float((bbox[0] + bbox[2]) / 2.0)
        if abs(x_center - key) > tolerance:
            return None
        return np.asarray(described["vector"], dtype=np.float32)

    return StubAppearance(describe_fn=describe)


def test_unrecoverable_reappearance_does_not_double_count_a_counted_person(
    test_config, sink, frame
):
    """P1 comptée, piste perdue, nouvelle piste au même endroit, non réassociable.

    Sans garde-fou, elle serait comptée comme une apparition intérieure et
    l'occupation doublerait. ``NEW`` est donc différé et journalisé, une seule
    fois par épisode.
    """
    described = {"vector": (1.0, 0.0, 0.0)}
    manager = build(test_config, sink, appearance=_flip_vector_appearance(described))
    sim = Sim(manager, frame).warmup()
    enter(sim, track_id=1, x=150.0)
    assert manager.occupancy_confirmed == 1
    person_id = manager.identities.person_of(1)

    described["vector"] = (0.0, 1.0, 0.0)     # apparence devenue inexploitable
    sim.steps(2)                              # P1 occultée, toujours comptée
    sim.steps(6, [det(99, INSIDE_Y, x=150.0, confidence=0.9)])

    assert manager.total_new == 0, "NEW doit être différé, pas confirmé"
    assert manager.total_in == 1
    assert manager.occupancy_operational == 1, "aucun doublon d'occupation"
    assert manager.occupancy_range == (1, 1)
    deferrals = [
        event
        for event in sink.of_type("INCONSISTENT_STATE")
        if event["kind"] == "new_deferred_occluded_person_nearby"
    ]
    assert len(deferrals) == 1, "un seul événement par épisode, pas un par frame"
    assert deferrals[0]["person_id"] != person_id
    assert manager.identities.person_of(1) == person_id


def test_deferral_clears_when_the_counted_person_leaves(test_config, sink, frame):
    """L'ambiguïté levée (P1 ressortie), l'apparition intérieure est comptée."""
    described = {"vector": (1.0, 0.0, 0.0)}
    manager = build(test_config, sink, appearance=_flip_vector_appearance(described))
    sim = Sim(manager, frame).warmup()
    enter(sim, track_id=1, x=150.0)
    described["vector"] = (0.0, 1.0, 0.0)

    sim.steps(2)
    sim.steps(6, [det(99, INSIDE_Y, x=150.0)])     # NEW différé
    assert manager.total_new == 0

    # P1 réapparaît puis sort par la ligne : la sortie est confirmée, la
    # personne comptée occultée disparaît de l'ambiguïté.
    sim.steps(2, [det(1, INSIDE_Y, x=150.0), det(99, INSIDE_Y, x=150.0)])
    sim.steps(3, [det(1, OUTSIDE_Y, x=150.0), det(99, INSIDE_Y, x=150.0)])
    assert manager.total_out == 1
    sim.steps(4, [det(99, INSIDE_Y, x=150.0)])
    assert manager.total_new == 1, "NEW confirmé après disparition de l'ambiguïté"
    assert manager.occupancy_operational == 1
    assert manager.occupancy_confirmed == 1


# ---------------------------------------------------------------------------
# Qualité des apparences écrites en galerie
# ---------------------------------------------------------------------------
def test_tiny_crop_descriptor_is_rejected(test_config, sink, frame):
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    tiny = Detection(1, np.array([295.0, 440.0, 305.0, 460.0]), 0.95)  # 10 × 20
    sim.steps(3, [tiny])
    rejections = sink.of_type("REID_DESCRIPTOR_REJECTED")
    assert rejections
    assert rejections[0]["reason"] == "tiny_crop"
    assert manager.identities.records[1].feature is None


def test_descriptor_from_an_edge_truncated_box_is_rejected(test_config, sink, frame):
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    truncated = Detection(1, np.array([0.0, 250.0, 80.0, 450.0]), 0.95)
    sim.steps(3, [truncated])
    reasons = {event["reason"] for event in sink.of_type("REID_DESCRIPTOR_REJECTED")}
    assert "edge_truncated" in reasons


def test_blur_gate_rejects_a_flat_crop_when_enabled(make_config, sink):
    """``gallery_min_sharpness`` > 0 : un crop uniforme est refusé explicitemement."""
    config = make_config({
        "reid": {"long_term": {"gallery_min_sharpness": 50.0}},
    })
    identity = IdentityManager(
        long_term=config.reid.long_term,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=StubAppearance(default=(1.0, 0.0, 0.0)),
    )
    manager = build(config, sink, identity=identity)
    sim = Sim(manager, np.zeros((600, 600, 3), dtype=np.uint8)).warmup()
    sim.steps(2, [det(1, INSIDE_Y, x=300.0)])
    rejections = sink.of_type("REID_DESCRIPTOR_REJECTED")
    assert rejections, "l'image uniforme n'est pas exploitable"
    assert rejections[0]["reason"] == "blurred_or_unreliable"
    assert rejections[0]["threshold"] == pytest.approx(50.0)
    assert "sharpness" in rejections[0]


def test_rejected_descriptor_never_replaces_a_known_appearance(test_config, sink, frame):
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [det(1, INSIDE_Y, x=150.0, confidence=0.95)])
    original = manager.identities.records[1].feature
    assert original is not None
    sim.steps(2, [det(1, INSIDE_Y, x=150.0, confidence=0.2)])
    assert np.allclose(manager.identities.records[1].feature, original)
    reasons = {event["reason"] for event in sink.of_type("REID_DESCRIPTOR_REJECTED")}
    assert reasons == {"low_confidence"}


def test_rejection_is_reported_once_per_reason(test_config, sink, frame):
    """Un refus permanent ne doit pas écrire une ligne par frame."""
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    sim.steps(10, [det(1, INSIDE_Y, x=150.0, confidence=0.2)])
    assert sink.count("REID_DESCRIPTOR_REJECTED") == 1


@pytest.mark.parametrize(
    "yolo_gate,lost_without_the_gate",
    [(0.25, True), (0.10, False)],
)
def test_yolo_gate_variants_on_the_same_blurred_scenario(
    test_config, sink, frame, yolo_gate, lost_without_the_gate
):
    """Compare deux seuils d'inférence YOLO sur le **même** scénario flou.

    Le seuil ``model.confidence`` s'applique dans Ultralytics **avant**
    BoT-SORT ; on le reproduit ici en retirant les détections sous le seuil,
    exactement comme le fait l'inférence. Comparaison :

    - ``0.25`` (ancienne valeur) : la détection à 0,12 n'existe plus, la piste
      est perdue (``OCCULTEE``) alors que la personne est toujours devant la
      caméra ; elle devra être récupérée par la galerie ou recompter comme
      entrée — source directe des faux ``NEW`` ;
    - ``0.10`` (valeur retenue) : la détection faible reste disponible pour
      l'association, la piste survit, et le garde-fou contre les faux positifs
      reste porté par ``new_track_thresh``.

    Compromis : un seuil plus bas augmente le rappel de détection (moins de
    pertes de piste) au prix de davantage de candidats faibles à associer ; un
    seuil plus haut réduit les faux positifs mais casse la persistance.
    """
    manager = build(test_config, sink)
    sim = Sim(manager, frame).warmup()
    enter(sim)
    assert manager.total_in == 1

    blurred = det(1, INSIDE_Y, x=152.0, confidence=0.12)
    visible = [] if blurred.confidence < yolo_gate else [blurred]
    assert bool(visible) is (not lost_without_the_gate)

    sim.steps(3, visible)
    assert (manager.tracks[1].state.name == "OCCULTEE") is lost_without_the_gate
    assert sink.count("OCCLUDED") == (1 if lost_without_the_gate else 0)
    # Dans les deux cas, aucune fausse sortie et aucune occupation inventée.
    assert manager.total_out == 0
    assert manager.occupancy_operational == 1


def association_gate(config, detections):
    """Second étage de BoT-SORT : sous ``track_low_thresh``, la détection ne
    peut plus servir d'association, la piste existante n'est donc plus nourrie."""
    return [
        item for item in detections if item.confidence >= config.tracker.track_low_thresh
    ]


@pytest.mark.parametrize(
    "yolo_gate,track_low,blurred_confidence,track_survives",
    [
        (0.25, 0.10, 0.12, False),  # ancien réglage : perte à l'inférence
        (0.10, 0.10, 0.12, True),  # réglage retenu : piste maintenue
        (0.10, 0.30, 0.12, False),  # seuil tracker trop haut exigé pour associer
        (0.10, 0.10, 0.08, False),  # sous le plancher d'exploitation du tracker
    ],
)
def test_threshold_matrix_on_the_same_blurred_scenario(
    test_config, sink, frame, yolo_gate, track_low, blurred_confidence, track_survives
):
    """Matrice ``model.confidence`` x ``tracker.track_low_thresh`` sur **un seul**
    scénario flou (mesure reproductible, résultats du prompt de correction) :

    ==================  ==============  ==============  =================
    model.confidence    track_low       flou (conf.)    piste maintenue ?
    ==================  ==============  ==============  =================
    0,25                0,10            0,12            non (filtrée amont)
    0,10                0,10            0,12            oui
    0,10                0,30            0,12            non (seuil tracker)
    0,10                0,10            0,08            non (sous le plancher)
    ==================  ==============  ==============  =================

    Les deux étages sont indépendants : le seuil d'inférence décide si la
    détection *existe*, ``track_low_thresh`` décide si elle peut *nourrir* une
    piste. Un seuil d'inférence plus bas que ``track_low_thresh`` ne sert donc à
    rien, d'où l'invariant ``model.confidence <= track_low_thresh``. Quel que
    soit le jeu de seuils, aucune fausse sortie ni occupation inventée : la
    perte de piste reste une incertitude, pas un comptage.
    """
    from config import override_config

    config = override_config(
        test_config,
        {
            "model": {"confidence": yolo_gate},
            "tracker": {"track_low_thresh": track_low},
        },
    )
    manager = build(config, sink)
    sim = Sim(manager, frame).warmup()
    enter(sim)
    assert manager.total_in == 1

    blurred = det(1, INSIDE_Y, x=152.0, confidence=blurred_confidence)
    usable = association_gate(config, inference_gate(config, [blurred]))
    sim.steps(4, usable)

    assert (manager.tracks[1].state.name != "OCCULTEE") is track_survives
    assert sink.count("OCCLUDED") == (0 if track_survives else 1)
    # Invariants indépendants des seuils : la personne reste comptée.
    assert manager.total_out == 0 and manager.total_new == 0
    assert manager.occupancy_operational == 1
    assert manager.occupancy_confirmed == 1
    assert sink.count("OUT") == 0


def test_descriptor_quality_thresholds_come_from_the_configuration():
    from config import LongTermReidConfig

    limits = LongTermReidConfig()
    assert limits.gallery_min_confidence == pytest.approx(0.70)
    assert limits.gallery_min_crop_height_px == 32
    assert limits.gallery_min_crop_width_px == 16
    assert limits.gallery_min_sharpness == pytest.approx(0.0)
