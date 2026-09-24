"""Relecture d'une session (``--replay``) : compteurs selon le journal, dessin partagé.

Aucune détection ni suivi : la vidéo est relue, les compteurs suivent
``events.jsonl`` (horloge de la vidéo), la ligne et la zone morte sont
dessinées par ``calibration.draw_line_overlay``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import main as entry_point
from calibration import ValidatedLine
from replay import DEAD_ZONE_FILENAME, ReplayCounters, ReplayError, load_session, replay
from test_main_e2e import FRAME_SIDE, run_main, write_video
from web_view import WebState, render_line_overlay


class _FakeClock:
    """Horloge simulée : ``sleep`` avance le temps au lieu d'attendre."""

    def __init__(self) -> None:
        self.now = 100.0
        self.slept = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds


def _write_session(root: Path, video: Path, events: list[dict], *, line=True, dead_zone=True,
                   frames_processed: int | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    calibration = {
        "session_id": root.name,
        "source": str(video),
        "line": {"p1_normalized": [0.1, 0.5], "p2_normalized": [0.9, 0.5], "inside_side": "negative"}
        if line else None,
        "line_origin": "argument" if line else "no_line_argument",
    }
    (root / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    (root / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    if dead_zone:
        (root / DEAD_ZONE_FILENAME).write_text(
            json.dumps({"timestamp_s": 0.0, "frame_index": 1, "dead_zone_px": None}) + "\n"
            + json.dumps({"timestamp_s": 0.5, "frame_index": 6, "dead_zone_px": 12.5}) + "\n",
            encoding="utf-8",
        )
    if frames_processed is not None:
        (root / "summary.json").write_text(json.dumps({"frames_processed": frames_processed}), encoding="utf-8")
    return root


EVENTS = [
    {"type": "SESSION_START", "timestamp_s": 0.0},
    {"type": "WARMUP_END", "timestamp_s": 0.2, "initial_occupancy": 2},
    {"type": "IN", "timestamp_s": 0.5},
    {"type": "NEW", "timestamp_s": 0.8},
    {"type": "OCCUPANCY_SNAPSHOT", "timestamp_s": 1.0, "occupancy_confirmed": 4,
     "total_in": 1, "total_out": 0, "total_new": 1},
    {"type": "OUT", "timestamp_s": 1.4},
    # Horodatés à l'horloge murale (bien au-delà de la vidéo) : appliqués à la fin.
    {"type": "SOURCE_END_OF_STREAM", "timestamp_s": 812.0},
    {"type": "SESSION_END", "timestamp_s": 812.1, "occupancy_confirmed": 3,
     "total_in": 1, "total_out": 1, "total_new": 1},
]


def test_compteurs_reconstruits_depuis_le_journal():
    counters = ReplayCounters()
    for event in EVENTS[:4]:
        counters.apply(event)
    assert (counters.confirmed, counters.total_in, counters.total_new) == (4, 1, 1)
    counters.apply(EVENTS[5])
    assert (counters.confirmed, counters.total_out) == (3, 1)
    # Un relevé du pipeline fait foi.
    counters.apply({"type": "OCCUPANCY_SNAPSHOT", "occupancy_confirmed": 7,
                    "total_in": 5, "total_out": 1, "total_new": 1})
    assert (counters.confirmed, counters.initial) == (7, 2)


def test_relecture_a_vitesse_reelle_et_compteurs_finaux(tmp_path: Path):
    video = write_video(tmp_path / "clip.mp4", frames=20)  # 10 ips : 2 s
    session = load_session(_write_session(tmp_path / "s", video, EVENTS))
    state = WebState()
    state.add_viewer()
    seen: list[tuple[int, float | None]] = []
    published: list[dict] = []
    original_publish = state.publish

    def spy(frame, **counters):
        published.append(counters)
        original_publish(frame, **counters)

    state.publish = spy

    def render(frame, line, dead_zone_px):
        seen.append((len(seen), dead_zone_px))
        return render_line_overlay(frame, line, dead_zone_px)

    fake = _FakeClock()
    counters = replay(session, state, render=render, clock=fake.clock, sleep=fake.sleep)

    # Vitesse réelle : l'image i est publiée à i / 10 s du début.
    assert fake.slept == pytest.approx(1.9)
    assert len(seen) == 20
    # Zone morte : largeur relevée pendant la session, à l'instant de l'image.
    assert seen[4][1] is None and seen[5][1] == 12.5
    # Compteurs à l'horloge de la vidéo : IN à 0,5 s (image 5), OUT à 1,4 s (image 14).
    assert published[4]["total_in"] == 0 and published[5]["total_in"] == 1
    assert published[13]["total_out"] == 0 and published[14]["total_out"] == 1
    # Compteurs finaux = ceux de la session d'origine (SESSION_END).
    end = EVENTS[-1]
    assert (counters.confirmed, counters.total_in, counters.total_out, counters.total_new) == (
        end["occupancy_confirmed"], end["total_in"], end["total_out"], end["total_new"])
    _, stats = state.stats()
    assert (stats["presentes"], stats["entrees"], stats["sorties"], stats["nouvelles"]) == (3, 1, 1, 1)
    assert stats["relecture"] is True and stats["etat"] == "termine"


def test_relecture_s_arrete_aux_images_de_la_session_et_sans_lecteur_ne_rend_rien(tmp_path: Path):
    video = write_video(tmp_path / "clip.mp4", frames=20)
    session = load_session(_write_session(tmp_path / "s", video, EVENTS, frames_processed=8))
    state = WebState()
    rendered = []
    fake = _FakeClock()
    replay(session, state, render=lambda *a: rendered.append(a), clock=fake.clock, sleep=fake.sleep)
    assert rendered == []                        # aucun navigateur : aucun rendu
    assert fake.slept == pytest.approx(0.7)      # 8 images, pas 20
    assert state.stats()[1]["presentes"] == 3    # fin du journal appliquée


def test_relecture_sans_ligne_et_session_incomplete(tmp_path: Path):
    video = write_video(tmp_path / "clip.mp4", frames=3)
    session = load_session(_write_session(tmp_path / "s", video, EVENTS[:2], line=False, dead_zone=False))
    assert session.line is None and session.dead_zone == []
    state = WebState()
    fake = _FakeClock()
    replay(session, state, render=render_line_overlay, clock=fake.clock, sleep=fake.sleep)
    _, stats = state.stats()
    assert stats["sans_ligne"] is True and stats["entrees"] is None
    with pytest.raises(ReplayError):
        load_session(tmp_path / "absent")
    broken = _write_session(tmp_path / "b", tmp_path / "introuvable.mp4", EVENTS)
    with pytest.raises(ReplayError):
        replay(load_session(broken), WebState(), render=render_line_overlay,
               clock=fake.clock, sleep=fake.sleep)


def test_relecture_d_une_session_du_pipeline_egale_ses_compteurs(tmp_path: Path, monkeypatch):
    """Session produite par ``main`` (faux YOLO), puis relue : mêmes compteurs
    finaux, et la zone morte relevée pendant la session est enregistrée."""
    frames = 40
    video = write_video(tmp_path / "clip.mp4", frames=frames)
    side = FRAME_SIDE
    person = (1, [0.40 * side, 0.60 * side, 0.50 * side, 0.95 * side], 0.9)
    line = ValidatedLine(
        p1=(0.05, 0.5), p2=(0.95, 0.5), inside_side="negative",
        frame_width=side, frame_height=side, display_scale=1.0,
    )
    code = run_main(tmp_path, monkeypatch, line=line, source=str(video), session_id="orig",
                    frames=frames, detections=[person])
    assert code == 0
    root = tmp_path / "orig"
    trace = [json.loads(row) for row in (root / DEAD_ZONE_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert trace and trace[-1]["dead_zone_px"] == pytest.approx(0.15 * 0.35 * side, abs=0.01)

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    state = WebState()
    fake = _FakeClock()
    counters = replay(load_session(root), state, render=render_line_overlay,
                      clock=fake.clock, sleep=fake.sleep)
    end = next(json.loads(row) for row in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(row)["type"] == "SESSION_END")
    assert (counters.confirmed, counters.total_in, counters.total_out, counters.total_new) == (
        end["occupancy_confirmed"], end["total_in"], end["total_out"], end["total_new"])
    assert counters.confirmed == summary["counters"]["occupancy_confirmed"]
    assert counters.confirmed >= 1  # la personne présente est bien comptée


def test_option_replay_lance_la_relecture_sans_pipeline(tmp_path: Path, monkeypatch):
    video = write_video(tmp_path / "clip.mp4", frames=4)
    root = _write_session(tmp_path / "s", video, EVENTS)
    started = {}

    class _Server:
        def shutdown(self):
            started["arret"] = True

    import web_view

    monkeypatch.setattr(web_view, "start_server", lambda state, port: started.setdefault("port", port) and _Server())
    monkeypatch.setattr(entry_point, "load_config", lambda *a: pytest.fail("configuration chargée"))

    def _ctrl_c(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(entry_point.time, "sleep", _ctrl_c)
    assert entry_point.main(["--replay", str(root), "--port", "8765"]) == 0
    assert started == {"port": 8765, "arret": True}
    assert entry_point.main(["--replay", str(tmp_path / "absent")]) == 2
