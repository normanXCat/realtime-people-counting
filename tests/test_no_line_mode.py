"""Mode SANS ligne (vidéo sans porte) : effectif = warm-up + NEW, jamais de IN/OUT.

Le mode est toujours explicite (``--no-line`` ou touche « N ») ; sans ce choix,
l'absence de ligne reste refusée, et le mode avec ligne n'est pas modifié.
"""

from __future__ import annotations

import pytest

from calibration import KEY_NO_LINE, LineSelection
from conftest import TEST_FPS, StubAppearance, identity_vectors_by_x, person_box
from geometry import LineError
from identity_manager import IdentityManager
import main as entry_point
from main import build_argument_parser, perform_line_selection
from occupancy_manager import Detection, OccupancyManager

# Positions qui, avec la ligne de test horizontale à y = 300, seraient
# « dehors » (250) puis « dedans » (450) : sans ligne, elles ne signifient rien.
OUTSIDE_Y = 250.0
INSIDE_Y = 450.0
VECTORS = {150.0: (1.0, 0.0, 0.0), 300.0: (0.0, 1.0, 0.0), 450.0: (0.0, 0.0, 1.0)}


def det(track_id, y, x=150.0):
    return Detection(track_id, person_box(x, y), 0.9)


class Sim:
    def __init__(self, manager, frame):
        self.manager, self.frame, self.time, self.index = manager, frame, 0.0, 0

    def steps(self, count, detections=()):
        for _ in range(count):
            self.index += 1
            self.time += 1.0 / TEST_FPS
            self.manager.process_frame(list(detections), self.frame, self.time, self.index)
        return self


@pytest.fixture
def sim(test_config, sink, frame):
    identity = IdentityManager(
        long_term=test_config.reid.long_term,
        external_reid=test_config.reid.external_reid,
        grace_period_seconds=test_config.timing.grace_period_seconds,
        event_sink=sink,
        appearance=StubAppearance(describe_fn=identity_vectors_by_x(VECTORS)),
    )
    manager = OccupancyManager(
        test_config, event_sink=sink, identity_manager=identity, line=None, no_line=True
    )
    manager.set_frame_budget(test_config.timing.frame_budget(TEST_FPS))
    return Sim(manager, frame)


def test_absence_de_ligne_reste_refusee_sans_choix_explicite(test_config, sink):
    with pytest.raises(ValueError):
        OccupancyManager(test_config, event_sink=sink, line=None)


def test_presente_au_warmup_comptee_dans_l_effectif_initial(sim, sink):
    sim.steps(8, [det(1, INSIDE_Y)])
    assert sim.manager.occupancy_operational == 1
    assert sim.manager.total_new == 0
    assert sink.count("IN") == 0 and sink.count("OUT") == 0


def test_nouvelle_personne_apres_warmup_comptee_en_new(sim, sink):
    sim.steps(5)
    sim.steps(6, [det(1, INSIDE_Y)])
    assert sim.manager.total_new == 1
    assert sink.count("NEW") == 1
    assert sim.manager.occupancy_operational == 1


def test_aucun_in_ni_out_en_traversant_l_image(sim, sink):
    sim.steps(5)
    # Aller-retour qui franchirait la ligne de test dans les deux sens.
    for y in (OUTSIDE_Y, INSIDE_Y, OUTSIDE_Y, INSIDE_Y):
        sim.steps(3, [det(1, y)])
    sim.steps(40)  # disparition longue : purge technique, jamais un OUT
    assert sink.count("IN") == 0 and sink.count("OUT") == 0
    assert sim.manager.total_in == 0 and sim.manager.total_out == 0
    assert sim.manager.total_new == 1


def test_selection_sans_ligne_explicite(test_config):
    assert perform_line_selection(test_config, "clip.mp4", no_line=True) is None
    assert (
        perform_line_selection(test_config, "clip.mp4", headless_requested=True, no_line=True)
        is None
    )


def test_no_show_sans_line_ouvre_la_question_de_demarrage(test_config, monkeypatch):
    """Attente **modifiée le 2026-09-23** (§26) : plus de refus (code 6), la
    fenêtre de démarrage s'ouvre ; « N » y choisit le mode sans ligne."""
    monkeypatch.setattr(entry_point, "display_available", lambda: True)
    monkeypatch.setattr(entry_point, "select_line", lambda source, **kw: None)
    assert perform_line_selection(test_config, "clip.mp4", headless_requested=True) is None


def test_line_et_no_line_incompatibles(test_config):
    with pytest.raises(LineError):
        perform_line_selection(
            test_config, "clip.mp4", line_argument="0.1,0.5,0.9,0.5", no_line=True
        )


def test_option_no_line_en_ligne_de_commande():
    parser = build_argument_parser()
    assert parser.parse_args(["--no-line"]).no_line is True
    assert parser.parse_args([]).no_line is False


def test_touche_n_indiquee_dans_la_fenetre(monkeypatch):
    import calibration

    assert ord("n") in KEY_NO_LINE and ord("N") in KEY_NO_LINE
    written = []
    monkeypatch.setattr(calibration.cv2, "putText", lambda _c, text, *a, **k: written.append(text))
    selection = LineSelection(
        frame_width=640, frame_height=480, display_width=640, display_height=480,
        inside_side="negative", min_length_ratio=0.05,
    )
    selection._draw_help(None, 640, 480)
    assert any(text.startswith("N : demarrer SANS ligne") for text in written)


def test_no_line_court_circuite_la_fenetre_de_demarrage(test_config, monkeypatch):
    """`--no-line` : aucune fenêtre, avec ou sans `--no-show` (§26)."""
    appels: list = []
    monkeypatch.setattr(entry_point, "select_line", lambda source, **kw: appels.append(source))
    for headless in (False, True):
        assert (
            perform_line_selection(
                test_config, "clip.mp4", headless_requested=headless, no_line=True
            )
            is None
        )
    assert appels == []
