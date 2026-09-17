"""Niveau 2 — Trajectoires synthétiques : comptage exact attendu.

Chaque scénario rejoue des positions artificielles avec des identités connues et
vérifie les compteurs IN/OUT/occupation, y compris : entrée simple, sortie
simple, plusieurs entrées/sorties, scène vide au démarrage, personne déjà
présente au démarrage, franchissement rapide entre deux frames, hésitation
répétée autour de la ligne, disparition en cours de franchissement, ancre au
bord d'image et zone morte **relative**.

Repère géométrique des tests : image 600×600, ligne horizontale à y = 300,
``inside_side: negative`` donc l'intérieur est **sous** la ligne.
Avec des boîtes de hauteur 200, la zone morte vaut 0,15 × 200 = 30 px :

    y <= 270          -> EXTERIEUR
    270 < y < 330      -> ZONE MORTE
    y >= 330           -> INTERIEUR
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TEST_FPS, StubAppearance, identity_vectors_by_x, make_test_line, person_box
from fsm import Action, TrackState
from identity_manager import IdentityManager
from occupancy_manager import Detection, OccupancyManager

LINE_Y = 300.0
# Toutes les positions restent à plus de 2 % du bord (marge 12 px sur 600 px),
# sinon la décision est suspendue par le garde-fou de la spec 5.4.
OUTSIDE_Y = 250.0          # 50 px au-dessus de la ligne : EXTERIEUR
INSIDE_Y = 450.0           # 150 px sous la ligne : INTERIEUR
IN_BUFFER_INSIDE = 315.0   # 15 px sous la ligne : dans la zone morte (30 px)
IN_BUFFER_OUTSIDE = 285.0  # 15 px au-dessus de la ligne : dans la zone morte


class Sim:
    """Rejoue une séquence de frames à FPS fixe contre un OccupancyManager."""

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

    def warmup(self, detections=()):
        """Fait tourner le warm-up complet (0,5 s à 10 fps = 5 frames)."""
        return self.steps(5, detections)


def detection(track_id: int, y: float, x: float = 300.0, height: float = 200.0) -> Detection:
    return Detection(
        technical_track_id=track_id,
        bbox=person_box(x, y, height=height),
        confidence=0.9,
    )


def build_manager(config, sink, appearance=None, line=None) -> OccupancyManager:
    identity = IdentityManager(
        long_term=config.reid.long_term,
        external_reid=config.reid.external_reid,
        grace_period_seconds=config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=appearance,
    )
    manager = OccupancyManager(
        config,
        event_sink=sink,
        identity_manager=identity,
        line=line or make_test_line(inside_side=config.line.inside_side),
    )
    manager.set_frame_budget(config.timing.frame_budget(TEST_FPS))
    return manager


# ---------------------------------------------------------------------------
# Scène vide / warm-up
# ---------------------------------------------------------------------------
def test_empty_scene_counts_nothing(manager, frame, sink):
    sim = Sim(manager, frame).steps(40)
    assert manager.total_in == 0
    assert manager.total_out == 0
    assert manager.total_new == 0
    assert manager.occupancy_confirmed == 0
    assert manager.occupancy_range == (0, 0)
    assert sink.count("IN") == 0 and sink.count("OUT") == 0 and sink.count("NEW") == 0
    assert sink.count("OCCUPANCY_SNAPSHOT") > 0


def test_person_already_present_is_counted_at_warmup(manager, frame, sink):
    sim = Sim(manager, frame).warmup([detection(1, INSIDE_Y)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    assert manager.initial_occupancy == 1
    assert manager.occupancy_confirmed == 1
    # Aucun IN : la personne était déjà dans la salle au démarrage.
    assert sink.count("IN") == 0
    assert sink.count("NEW") == 0
    assert sink.count("WARMUP_END") == 1


def test_unstable_track_is_excluded_from_initial_occupancy(manager, frame, sink):
    """Spec 5.8 : une piste non stable ne contamine pas l'effectif initial."""
    sim = Sim(manager, frame)
    sim.step([detection(1, INSIDE_Y)])   # une seule frame intérieure
    sim.steps(4)                          # puis disparition pendant le warm-up
    sim.steps(3)
    assert manager.initial_occupancy == 0
    assert manager.occupancy_confirmed == 0


# ---------------------------------------------------------------------------
# Entrées / sorties simples
# ---------------------------------------------------------------------------
def test_simple_entry(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(2, [detection(1, INSIDE_Y)])
    assert manager.total_in == 1
    assert manager.total_out == 0
    assert manager.occupancy_confirmed == 1
    events = sink.of_type("IN")
    assert len(events) == 1
    assert events[0]["direction"] == "in"
    assert events[0]["person_id"] == 1


def test_simple_exit(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(2, [detection(1, INSIDE_Y)])
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    assert manager.total_in == 1
    assert manager.total_out == 1
    assert manager.occupancy_confirmed == 0
    assert sink.count("IN") == 1 and sink.count("OUT") == 1


def test_multiple_entries_and_exits(frame, sink, test_config):
    """Trois personnes distinctes entrent puis deux sortent.

    Chaque personne a une apparence propre : avec des descripteurs identiques,
    la galerie les fusionnerait à juste titre (c'est le rôle des tests
    d'ambiguïté de couvrir ce cas).
    """
    appearance = StubAppearance(
        describe_fn=identity_vectors_by_x({
            150.0: (1.0, 0.0, 0.0),
            300.0: (0.0, 1.0, 0.0),
            450.0: (0.0, 0.0, 1.0),
        })
    )
    manager = build_manager(test_config, sink, appearance=appearance)
    sim = Sim(manager, frame).warmup()
    positions = {1: 150.0, 2: 300.0, 3: 450.0}
    for track_id, x in positions.items():
        sim.steps(2, [detection(track_id, OUTSIDE_Y, x=x)])
        sim.steps(2, [detection(track_id, INSIDE_Y, x=x)])
    assert manager.total_in == 3
    assert manager.occupancy_confirmed == 3
    for track_id in (1, 2):
        sim.steps(2, [detection(track_id, OUTSIDE_Y, x=positions[track_id])])
    assert manager.total_out == 2
    assert manager.occupancy_confirmed == 1
    assert sink.count("AMBIGUOUS_CROSSING") == 0


def test_fast_crossing_between_two_frames(manager, frame, sink):
    """Un saut complet entre deux frames doit produire exactement un IN."""
    sim = Sim(manager, frame).warmup()
    sim.step([detection(1, OUTSIDE_Y)])
    sim.step([detection(1, INSIDE_Y)])
    assert manager.total_in == 1
    assert sink.count("IN") == 1


def test_fast_exit_between_two_frames(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.step([detection(1, OUTSIDE_Y)])
    sim.step([detection(1, INSIDE_Y)])
    sim.step([detection(1, OUTSIDE_Y)])
    assert manager.total_out == 1
    assert sink.count("OUT") == 1


def test_new_presence_inside_without_crossing(manager, frame, sink):
    """Apparition directe à l'intérieur : NEW après confirmation, pas d'IN."""
    sim = Sim(manager, frame).warmup()
    sim.steps(6, [detection(1, INSIDE_Y)])
    assert manager.total_new == 1
    assert manager.total_in == 0
    assert manager.occupancy_confirmed == 1
    events = sink.of_type("NEW")
    assert len(events) == 1
    assert events[0]["direction"] == "in"
    assert events[0]["stable_frames"] >= 2


# ---------------------------------------------------------------------------
# Hystérésis : hésitation autour de la ligne
# ---------------------------------------------------------------------------
def test_hesitation_in_the_buffer_never_counts(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    for _ in range(8):
        sim.step([detection(1, IN_BUFFER_OUTSIDE)])
        sim.step([detection(1, IN_BUFFER_INSIDE)])
    assert manager.total_in == 0
    assert manager.total_out == 0
    assert manager.occupancy_confirmed == 0


def test_hesitation_then_commit_counts_once(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    for _ in range(8):
        sim.step([detection(1, IN_BUFFER_OUTSIDE)])
        sim.step([detection(1, IN_BUFFER_INSIDE)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    assert manager.total_in == 1
    assert manager.total_out == 0
    assert sink.count("IN") == 1


def test_counted_person_hesitating_inside_is_not_counted_out(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    assert manager.occupancy_confirmed == 1
    for _ in range(6):
        sim.step([detection(1, IN_BUFFER_INSIDE)])
        sim.step([detection(1, INSIDE_Y)])
    assert manager.total_out == 0
    assert manager.occupancy_confirmed == 1


# ---------------------------------------------------------------------------
# Disparition en cours de franchissement (spec 4.3)
# ---------------------------------------------------------------------------
def test_disappearance_during_entry_is_ambiguous(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.step([detection(1, OUTSIDE_Y)])
    sim.step([detection(1, LINE_Y)])              # entre dans la zone morte
    sim.step([detection(1, IN_BUFFER_INSIDE)])    # franchissement amorcé
    assert manager.tracks[1].state is TrackState.ENTREE_EN_COURS

    sim.steps(12)  # disparition au-delà de la grâce (1,0 s = 10 frames)
    assert manager.total_in == 0, "un IN ne doit jamais être confirmé sur disparition"
    assert manager.total_out == 0
    assert manager.occupancy_confirmed == 0
    assert manager.occupancy_uncertain == 0
    ambiguous = sink.of_type("AMBIGUOUS_CROSSING")
    assert len(ambiguous) == 1
    assert ambiguous[0]["direction"] == "in"
    assert ambiguous[0]["pending_direction"] == "in"
    assert sink.count("PURGE") == 1
    assert sink.of_type("PURGE")[0]["reason"] == "grace_expired_in_progress"


def test_disappearance_during_exit_is_ambiguous_and_bounded(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    sim.step([detection(1, LINE_Y)])
    sim.step([detection(1, IN_BUFFER_OUTSIDE)])
    assert manager.tracks[1].state is TrackState.SORTIE_EN_COURS

    sim.steps(12)
    assert manager.total_out == 0
    assert sink.of_type("AMBIGUOUS_CROSSING")[0]["direction"] == "out"
    # La personne était comptée présente et aucune sortie n'est observée :
    # elle reste dans la borne haute de l'occupation (spec 4.4).
    assert manager.occupancy_confirmed == 0
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (0, 1)


def test_reappearance_resolves_the_pending_crossing(manager, frame, sink):
    """Spec 4.3 : la réapparition cohérente confirme le franchissement."""
    sim = Sim(manager, frame).warmup()
    sim.step([detection(1, OUTSIDE_Y)])
    sim.step([detection(1, LINE_Y)])
    sim.step([detection(1, IN_BUFFER_INSIDE)])
    assert manager.tracks[1].state is TrackState.ENTREE_EN_COURS
    assert manager.tracks[1].pending_crossing == "in"

    sim.steps(4)  # disparition courte, dans la fenêtre de grâce
    sim.step([detection(9, INSIDE_Y)])  # réapparition avec un nouvel ID technique
    sim.steps(2, [detection(9, INSIDE_Y)])

    assert manager.total_in == 1, "le franchissement amorcé doit être confirmé"
    assert sink.count("AMBIGUOUS_CROSSING") == 0
    assert sink.count("REID_MATCH") == 1
    events = sink.of_type("IN")
    assert events[0]["reason"] == "deferred_confirmation"


def test_recovery_outside_confirms_the_exit(manager, frame, sink):
    """Une personne comptée présente qui réapparaît dehors est sortie une fois."""
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    assert manager.occupancy_confirmed == 1

    sim.steps(4)                        # occultation courte (0,5 s)
    # Réapparition cohérente juste au-delà de la zone morte, côté sortie :
    # la contrainte spatio-temporelle autorise 1,5 × 200 × 0,5 + 60 = 210 px.
    sim.steps(2, [detection(9, 265.0)])
    assert manager.total_out == 1
    assert manager.occupancy_confirmed == 0
    assert manager.occupancy_uncertain == 0
    assert sink.of_type("OUT")[0]["reason"] == "recovery_outside_coherent"


def test_ambiguous_immediate_policy_decides_at_loss(frame, sink, make_config):
    config = make_config({
        "timing": {"warmup_seconds": 0.5, "confirmation_seconds": 0.2, "grace_period_seconds": 1.0},
        "occupancy": {"in_progress_disappearance_policy": "ambiguous_immediate"},
    })
    manager = build_manager(config, sink)
    sim = Sim(manager, frame).warmup()
    sim.step([detection(1, OUTSIDE_Y)])
    sim.step([detection(1, LINE_Y)])
    sim.step([detection(1, IN_BUFFER_INSIDE)])
    sim.step([])  # disparition
    assert sink.count("AMBIGUOUS_CROSSING") == 1
    assert manager.tracks[1].state is TrackState.ABSENTE
    assert manager.total_in == 0


# ---------------------------------------------------------------------------
# Occupation encadrée (spec 4.4)
# ---------------------------------------------------------------------------
def test_purge_without_exit_feeds_the_uncertain_bound(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    assert manager.occupancy_confirmed == 1

    sim.steps(13)  # disparition prolongée
    assert manager.total_out == 0
    assert manager.occupancy_confirmed == 0
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (0, 1)
    purge = sink.of_type("PURGE")[0]
    assert purge["reason"] == "grace_expired"
    assert purge["occupancy_impact"] == "uncertain"
    # Le snapshot publié porte bien les trois valeurs (spec 4.4).
    snapshot = sink.of_type("OCCUPANCY_SNAPSHOT")[-1]
    assert snapshot["occupancy_confirmed"] == 0
    assert snapshot["occupancy_uncertain"] == 1
    assert snapshot["occupancy_range"] == [0, 1]


def test_purged_identity_revives_within_retention(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    sim.steps(13)                       # purge, puis rétention de galerie
    assert manager.occupancy_uncertain == 1

    sim.step([detection(9, INSIDE_Y)])  # nouvel ID technique, même personne
    sim.steps(2, [detection(9, INSIDE_Y)])
    assert manager.occupancy_uncertain == 0
    assert manager.occupancy_confirmed == 1
    assert manager.total_in == 1
    assert sink.count("REID_MATCH") == 1


def test_reappearance_after_release_creates_a_new_identity(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    sim.steps(3, [detection(1, INSIDE_Y)])
    sim.steps(40)  # bien au-delà de la rétention de galerie

    sim.steps(5, [detection(9, INSIDE_Y)])
    assert sink.count("REID_MATCH") == 0
    # Nouvelle identité confirmée comme apparition intérieure.
    assert manager.total_new == 1
    assert manager.occupancy_confirmed == 1
    # L'ancienne personne reste dans la borne haute : jamais retirée en silence.
    assert manager.occupancy_uncertain == 1
    assert manager.occupancy_range == (1, 2)


def test_snapshot_is_published_at_the_configured_interval(manager, frame, sink):
    sim = Sim(manager, frame).steps(10)
    snapshots = sink.of_type("OCCUPANCY_SNAPSHOT")
    assert len(snapshots) == 10
    for snapshot in snapshots:
        assert set(snapshot) >= {
            "occupancy_confirmed", "occupancy_uncertain", "occupancy_range",
            "visible_count", "occluded_count",
        }


# ---------------------------------------------------------------------------
# Ligne inclinée : aucune contrainte d'axe pour la ligne validée
# ---------------------------------------------------------------------------
def test_crossing_an_inclined_line_is_counted(frame, sink, make_config):
    """Une ligne inclinée validée sert telle quelle à la détection de franchissement."""
    config = make_config({
        "timing": {"warmup_seconds": 0.5, "confirmation_seconds": 0.2, "grace_period_seconds": 1.0},
    })
    # Diagonale (0,2 ; 0,2) -> (0,8 ; 0,8), soit la droite y = x dans une image
    # 600×600 : la distance signée vaut signe(x - y), l'intérieur est négatif.
    manager = build_manager(config, sink, line=make_test_line((0.2, 0.2), (0.8, 0.8)))
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, 400.0, x=500.0)])   # x > y : extérieur
    sim.steps(2, [detection(1, 400.0, x=100.0)])   # x < y : intérieur

    assert manager.total_in == 1
    assert manager.occupancy_confirmed == 1
    assert sink.of_type("IN")[0]["direction"] == "in"


def test_vertical_line_is_used_as_validated(frame, sink, make_config):
    """Ligne verticale : l'intérieur est le demi-plan x >= 0,5 (orientation validée)."""
    config = make_config({
        "timing": {"warmup_seconds": 0.5, "confirmation_seconds": 0.2, "grace_period_seconds": 1.0},
    })
    # x = 0,5 dans une image 600×600 : distance signée = signe(x - 300).
    manager = build_manager(config, sink, line=make_test_line((0.5, 0.0), (0.5, 1.0)))
    sim = Sim(manager, frame).warmup()
    sim.steps(2, [detection(1, 300.0, x=520.0)])   # à droite de la ligne
    sim.steps(2, [detection(1, 300.0, x=80.0)])    # à gauche : intérieur

    assert manager.total_in == 1
    assert manager.occupancy_confirmed == 1


# ---------------------------------------------------------------------------
# Garde-fous de cohérence (spec 4.5)
# ---------------------------------------------------------------------------
def test_out_without_in_is_reported_not_counted(manager, sink):
    from occupancy_types import PersonTrack

    track = PersonTrack(person_id=42, state=TrackState.SORTIE_EN_COURS)
    assert track.inside_occupancy is False
    manager._count_out(track, 7, Action.COMPTE_SORTIE, 1.0, 10)
    assert manager.total_out == 0
    events = sink.of_type("INCONSISTENT_STATE")
    assert len(events) == 1
    assert events[0]["kind"] == "out_without_in"
    assert events[0]["person_id"] == 42


def test_negative_occupancy_is_reported(manager, sink):
    manager.initial_occupancy = -1
    manager._check_consistency(1.0, 10)
    events = sink.of_type("INCONSISTENT_STATE")
    assert [event["kind"] for event in events] == ["negative_occupancy_before_clamp"]


def test_side_change_without_crossing_is_reported(frame, sink, make_config):
    """Une dérive de demi-plan sans intersection capturée n'est pas silencieuse.

    La ligne ne couvre que la partie centrale de l'image : un déplacement
    latéralement hors de son étendue ne peut produire aucune intersection.
    """
    config = make_config({
        "timing": {"warmup_seconds": 0.5, "confirmation_seconds": 0.2, "grace_period_seconds": 1.0},
    })
    # La ligne ne couvre que la partie centrale : c'est la ligne « validée par
    # l'opérateur » pour ce scénario (elle ne vient plus de la configuration).
    manager = build_manager(config, sink, line=make_test_line((0.3, 0.5), (0.7, 0.5)))
    sim = Sim(manager, frame).warmup()
    # x = 520 est au-delà de l'extrémité de la ligne (0,7 × 600 = 420).
    sim.steps(3, [detection(1, INSIDE_Y, x=520.0)])
    assert manager.occupancy_confirmed == 1
    assert manager.tracks[1].state is TrackState.PRESENTE
    sim.steps(2, [detection(1, OUTSIDE_Y, x=520.0)])
    events = sink.of_type("INCONSISTENT_STATE")
    assert any(event["kind"] == "side_change_without_crossing" for event in events)
    # Le franchissement reste soumis à confirmation et aboutit à une sortie.
    assert manager.total_out == 1


# ---------------------------------------------------------------------------
# Ancre au bord de l'image (spec 5.4)
# ---------------------------------------------------------------------------
def test_edge_anchor_suspends_the_crossing_decision(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    # Ancre fiable établie côté extérieur.
    sim.steps(2, [detection(1, OUTSIDE_Y)])
    assert manager.total_in == 0
    # Boîte collée au bord gauche (x1 = 0) pendant la traversée : ancre non fiable.
    sim.steps(2, [Detection(technical_track_id=1, bbox=np.array([0.0, 250.0, 80.0, 400.0]))])
    assert manager.total_in == 0, "aucune décision sur une ancre au bord"
    assert sink.count("ANCHOR_UNRELIABLE") == 1
    assert sink.of_type("ANCHOR_UNRELIABLE")[0]["reason"] == "edge_margin"
    assert manager.tracks[1].last_rule == "suspend_unreliable_anchor"

    # La boîte revient dans le champ : la décision reprend depuis la dernière
    # ancre fiable (côté extérieur), donc le franchissement est compté une fois.
    sim.steps(2, [detection(1, INSIDE_Y)])
    assert manager.total_in == 1
    assert sink.count("IN") == 1
    assert manager.tracks[1].anchor_suspend_streak == 0


def test_edge_suspension_is_reported_once_per_streak(manager, frame, sink):
    sim = Sim(manager, frame).warmup()
    touching = Detection(technical_track_id=1, bbox=np.array([0.0, 250.0, 80.0, 450.0]))
    sim.steps(5, [touching])
    assert sink.count("ANCHOR_UNRELIABLE") == 1
    assert manager.tracks[1].anchor_suspend_streak == 5
    assert manager.total_in == 0 and manager.total_new == 0


# ---------------------------------------------------------------------------
# Zone morte relative (règle 0.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("height,expected_in", [(200.0, 1), (400.0, 0)])
def test_dead_zone_follows_local_bbox_height(sink, make_config, height, expected_in):
    """Même position, zones mortes différentes selon l'échelle locale (règle 0.3)."""
    import numpy as _np

    config = make_config({
        "timing": {"warmup_seconds": 0.5, "confirmation_seconds": 0.2, "grace_period_seconds": 1.0},
    })
    manager = build_manager(config, sink)
    big_frame = _np.zeros((1200, 1200, 3), dtype=_np.uint8)
    sim = Sim(manager, big_frame).warmup()
    # Ligne à y = 600. Boîtes de hauteur 200 -> zone morte 30 px ; 400 -> 60 px.
    sim.steps(2, [detection(1, 500.0, height=height)])
    # Position test : 50 px sous la ligne -> INTERIEURE si zone morte 30 px,
    # encore dans la bande tampon si elle vaut 60 px.
    sim.steps(2, [detection(1, 650.0, height=height)])
    assert manager.total_in == expected_in
    if expected_in == 0:
        assert manager.tracks[1].last_zone == "MORTE"
    else:
        assert manager.tracks[1].last_zone == "INTERIEURE"


def test_bootstrap_without_measured_fps_takes_no_decision(test_config, sink, frame):
    manager = build_manager(test_config, sink)
    manager.budget = None  # aucun FPS mesuré
    sim = Sim(manager, frame)
    sim.steps(3, [detection(1, INSIDE_Y)])
    assert manager.phase == "BOOTSTRAP"
    assert manager.bootstrap_frames == 3
    assert manager.total_in == 0 and manager.total_new == 0
    assert sink.count("IN") == 0
