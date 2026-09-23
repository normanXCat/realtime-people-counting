"""Interface web (``--web``) : calibration dans le navigateur et vue de démonstration.

Vérifie le rendu dédié (seule la ligne), l'objet partagé, les routes (client de
test Flask), la calibration web (points refusés, normalisation, inversion) et
l'intégration dans ``main`` (aucune fenêtre OpenCV, origine « web »).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

flask = pytest.importorskip("flask")

import main as entry_point  # noqa: E402
import web_view  # noqa: E402
from geometry import Side, VirtualLine  # noqa: E402
from main import apply_cli_overrides, build_argument_parser  # noqa: E402
from config import load_config  # noqa: E402
from web_view import (  # noqa: E402
    HOST,
    LINE_COLOR_BGR,
    PHASE_CALIBRATION,
    PHASE_DEMO,
    PHASE_WAITING,
    WebState,
    create_app,
    encode_jpeg,
    render_line_only,
)


def _line() -> VirtualLine:
    return VirtualLine(p1=(0.1, 0.5), p2=(0.9, 0.5), inside_side="negative")


def _frame(value: int = 40, shape=(120, 160, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _calibrating_state(width: int = 160, height: int = 120) -> WebState:
    state = WebState()
    state.open_calibration(
        _frame(shape=(height, width, 3)), inside_side="negative", min_length_ratio=0.05
    )
    return state


# ---------------------------------------------------------------------------
# Rendu dédié : la ligne, et rien d'autre
# ---------------------------------------------------------------------------
def test_rendu_dedie_ne_dessine_que_la_ligne():
    frame = _frame()
    rendered = render_line_only(frame, _line())
    changed = np.any(rendered != frame, axis=2)
    rows = np.nonzero(changed.any(axis=1))[0]
    cols = np.nonzero(changed.any(axis=0))[0]
    # Tous les pixels modifiés sont dans la bande de la ligne (y = 60, x 16..144).
    assert rows.min() >= 60 - 4 and rows.max() <= 60 + 4
    assert cols.min() >= 16 - 4 and cols.max() <= 144 + 4
    assert tuple(int(c) for c in rendered[60, 80]) == LINE_COLOR_BGR
    # L'image source n'est jamais modifiée.
    assert np.all(frame == 40)


def test_rendu_dedie_sans_ligne_laisse_l_image_intacte():
    frame = _frame()
    rendered = render_line_only(frame, None)
    assert rendered is not frame
    assert np.array_equal(rendered, frame)


# ---------------------------------------------------------------------------
# Objet partagé
# ---------------------------------------------------------------------------
def test_objet_partage_publie_image_et_quatre_compteurs():
    state = WebState()
    state.start_demo(no_line=False)
    seq0, values0 = state.stats()
    assert {k: values0[k] for k in ("presentes", "entrees", "sorties", "nouvelles")} == {
        "presentes": 0, "entrees": 0, "sorties": 0, "nouvelles": 0,
    }
    assert values0["phase"] == PHASE_DEMO and values0["etat"] == "en_cours"
    frame = _frame()
    state.publish(frame, confirmed=5, total_in=4, total_out=1, total_new=2)
    seq1, values1 = state.stats()
    assert seq1 == seq0 + 1
    assert (values1["presentes"], values1["entrees"], values1["sorties"], values1["nouvelles"]) == (5, 4, 1, 2)
    frame_seq, published = state.frame()
    assert frame_seq == 1 and published is frame


def test_objet_partage_ne_change_de_version_que_si_une_valeur_change():
    state = WebState()
    state.publish(_frame(), confirmed=1, total_in=1, total_out=0, total_new=0)
    seq, _ = state.stats()
    state.publish(_frame(), confirmed=1, total_in=1, total_out=0, total_new=0)
    assert state.stats()[0] == seq  # mêmes valeurs : aucune version nouvelle
    assert state.frame()[0] == 2    # mais l'image, elle, est renouvelée


def test_objet_partage_sans_ligne_n_affiche_ni_entrees_ni_sorties():
    state = WebState()
    state.start_demo(no_line=True)
    state.publish(_frame(), confirmed=3, total_in=7, total_out=7, total_new=3)
    _, values = state.stats()
    assert values["sans_ligne"] is True
    assert values["entrees"] is None and values["sorties"] is None
    assert values["presentes"] == 3 and values["nouvelles"] == 3


def test_objet_partage_fin_garde_image_et_valeurs():
    state = WebState()
    frame = _frame()
    state.publish(frame, confirmed=12, total_in=12, total_out=0, total_new=0)
    state.finish()
    _, values = state.stats()
    assert values["etat"] == "termine" and values["presentes"] == 12
    assert state.frame()[1] is frame


def test_objet_partage_attente_reveillee_par_une_publication():
    state = WebState()
    seq, _ = state.stats()
    result = {}

    def waiter():
        result["stats"] = state.wait_stats(seq, timeout=5.0)

    thread = threading.Thread(target=waiter)
    thread.start()
    state.publish(_frame(), confirmed=2, total_in=2, total_out=0, total_new=0)
    thread.join(timeout=5.0)
    assert result["stats"][1]["presentes"] == 2


# ---------------------------------------------------------------------------
# Calibration web : validation, normalisation, inversion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "p1, p2, attendu",
    [
        ((0.5, 0.5), (0.5, 0.5), "confondus"),
        ((0.50, 0.50), (0.52, 0.50), "trop courte"),
        ((-0.1, 0.5), (0.9, 0.5), "hors de l'image"),
    ],
)
def test_calibration_web_refuse_les_lignes_invalides(p1, p2, attendu):
    state = _calibrating_state()
    accepted, message = state.submit_calibration(
        {"mode": "ligne", "p1": list(p1), "p2": list(p2), "cote_interieur": "negative"}
    )
    assert not accepted and attendu in message


def test_calibration_web_refuse_point_mal_forme_et_mode_inconnu():
    state = _calibrating_state()
    assert not state.submit_calibration(
        {"mode": "ligne", "p1": [0.1], "p2": [0.9, 0.5], "cote_interieur": "negative"}
    )[0]
    assert not state.submit_calibration(
        {"mode": "ligne", "p1": [0.1, 0.5], "p2": [0.9, 0.5], "cote_interieur": "haut"}
    )[0]
    assert not state.submit_calibration({"mode": "autre"})[0]
    assert not state.submit_calibration("pas un objet")[0]


def test_calibration_web_normalisee_sur_la_taille_reelle_de_l_image():
    state = _calibrating_state(width=1280, height=720)
    accepted, _ = state.submit_calibration(
        {"mode": "ligne", "p1": [0.383, 1.0], "p2": [0.56, 0.55],
         "cote_interieur": "negative", "largeur_affichee": 640}
    )
    assert accepted
    mode, line = state.wait_calibration()
    assert mode == "ligne"
    assert (line.frame_width, line.frame_height) == (1280, 720)
    assert line.p1 == (0.383, 1.0) and line.p2 == (0.56, 0.55)
    assert line.clicks_px[0] == pytest.approx((0.383 * 1280, 720.0))
    assert line.display_scale == pytest.approx(0.5)
    assert line.inside_side == "negative"


@pytest.mark.parametrize("cote", ["negative", "positive"])
def test_calibration_web_inversion_et_cote_teinte_coherents(cote):
    """Le côté teinté par la page (formule de app.js) est l'intérieur du pipeline."""
    state = _calibrating_state()
    assert state.submit_calibration(
        {"mode": "ligne", "p1": [0.2, 0.3], "p2": [0.8, 0.6], "cote_interieur": cote}
    )[0]
    _, validated = state.wait_calibration()
    assert validated.inside_side == cote
    line = VirtualLine(p1=validated.p1, p2=validated.p2, inside_side=cote)
    # directionInterieure (app.js) : signe(côté) * (dy, -dx), en pixels affichés.
    width, height = 160, 120
    ax, ay = 0.2 * width, 0.3 * height
    bx, by = 0.8 * width, 0.6 * height
    sign = 1 if cote == "positive" else -1
    dx, dy = bx - ax, by - ay
    mid = ((ax + bx) / 2, (ay + by) / 2)
    teinte = (mid[0] + sign * dy * 0.2, mid[1] - sign * dx * 0.2)
    oppose = (mid[0] - sign * dy * 0.2, mid[1] + sign * dx * 0.2)
    assert line.side(teinte, width, height) is Side.INTERIEURE
    assert line.side(oppose, width, height) is Side.EXTERIEURE


def test_calibration_web_sans_ligne_et_validation_unique():
    state = _calibrating_state()
    assert state.submit_calibration({"mode": "sans_ligne"})[0]
    assert state.wait_calibration() == ("sans_ligne", None)
    accepted, message = state.submit_calibration({"mode": "sans_ligne"})
    assert not accepted and "déjà" in message


def test_calibration_web_refusee_hors_phase_de_calibration():
    state = WebState()
    assert not state.submit_calibration({"mode": "sans_ligne"})[0]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@pytest.fixture
def client_and_state():
    state = WebState()
    app = create_app(state, keepalive_seconds=0.05, video_resend_seconds=0.05)
    return app.test_client(), state


def test_route_page(client_and_state):
    client, _ = client_and_state
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for label in (
        "Personnes présentes", "Entrées", "Sorties", "Nouvelles présences",
        "Tracer une ligne", "Compter sans ligne", "Inverser le côté", "Recommencer",
        "Valider et lancer",
    ):
        assert label in html
    # Entièrement hors ligne : aucune ressource externe.
    assert "http://" not in html and "https://" not in html
    for asset in ("/static/style.css", "/static/app.js"):
        served = client.get(asset)
        assert served.status_code == 200
        assert b"http://" not in served.data and b"https://" not in served.data


def test_routes_de_calibration(client_and_state):
    client, state = client_and_state
    assert client.get("/calibration").get_json()["phase"] == PHASE_WAITING
    assert client.get("/calibration/image").status_code == 404

    state.open_calibration(_frame(shape=(72, 128, 3)), inside_side="negative", min_length_ratio=0.05)
    info = client.get("/calibration").get_json()
    assert info == {
        "phase": PHASE_CALIBRATION, "largeur": 128, "hauteur": 72,
        "cote_interieur": "negative", "longueur_min": 0.05,
    }
    image = client.get("/calibration/image")
    assert image.status_code == 200 and image.mimetype == "image/jpeg"

    assert client.post("/calibration", data="mode=sans_ligne").status_code == 415
    refused = client.post("/calibration", json={"mode": "ligne", "p1": [0.5, 0.5],
                                                "p2": [0.5, 0.5], "cote_interieur": "negative"})
    assert refused.status_code == 400 and "confondus" in refused.get_json()["message"]
    accepted = client.post("/calibration", json={"mode": "ligne", "p1": [0.1, 0.5],
                                                 "p2": [0.9, 0.5], "cote_interieur": "positive"})
    assert accepted.status_code == 200 and accepted.get_json()["accepte"] is True
    assert state.wait_calibration()[1].inside_side == "positive"


def test_route_video_flux_mjpeg(client_and_state):
    client, state = client_and_state
    state.publish(_frame(), confirmed=0, total_in=0, total_out=0, total_new=0)
    response = client.get("/video", buffered=False)
    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    chunks = iter(response.response)
    assert next(chunks) == b"--frame\r\n"
    part = next(chunks)
    assert part.startswith(b"Content-Type: image/jpeg")
    # Délimitation émise juste après l'image : la dernière image s'affiche aussi.
    assert part.endswith(encode_jpeg(state.frame()[1]) + b"\r\n--frame\r\n")
    # Sans image nouvelle (fin de vidéo), la dernière est renvoyée.
    assert next(chunks) == part
    response.close()


def test_route_stats_sse_n_envoie_que_les_changements(client_and_state):
    client, state = client_and_state
    state.start_demo(no_line=False)
    response = client.get("/stats", buffered=False)
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    chunks = iter(response.response)

    def next_data():
        for chunk in chunks:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            if text.startswith("data: "):
                return json.loads(text[len("data: "):])
        raise AssertionError("flux terminé")

    assert next_data()["presentes"] == 0
    state.publish(_frame(), confirmed=0, total_in=0, total_out=0, total_new=0)  # rien ne change
    state.publish(_frame(), confirmed=4, total_in=3, total_out=0, total_new=1)
    values = next_data()
    assert (values["presentes"], values["entrees"], values["sorties"], values["nouvelles"]) == (4, 3, 0, 1)
    state.finish()
    assert next_data()["etat"] == "termine"
    response.close()


# ---------------------------------------------------------------------------
# Options et intégration dans main
# ---------------------------------------------------------------------------
def test_option_web_port_et_ecoute_locale_uniquement():
    args = build_argument_parser().parse_args(["--web"])
    assert args.web is True and args.port == 8000
    assert not hasattr(args, "host"), "aucune option d'exposition réseau"
    assert HOST == "127.0.0.1"
    assert build_argument_parser().parse_args([]).web is False


def test_web_supprime_toute_fenetre_de_traitement_opencv(monkeypatch):
    monkeypatch.setattr(entry_point, "display_available", lambda: True)
    args = build_argument_parser().parse_args(["--web"])
    assert apply_cli_overrides(load_config(), args).display.enabled is False


class _FakeServer:
    def shutdown(self) -> None:
        return None


def _run_web_main(tmp_path: Path, monkeypatch, extra, submit=None) -> tuple[int, Path]:
    """`main --web` avec un faux pipeline, un faux serveur et un Ctrl+C final simulé."""
    from test_main_e2e import install_fake_ultralytics

    install_fake_ultralytics(monkeypatch, frames=6)
    states: list = []
    monkeypatch.setattr(web_view, "start_server", lambda state, port: states.append(state) or _FakeServer())
    monkeypatch.setattr(entry_point, "read_first_frame", lambda source: _frame(shape=(480, 640, 3)))
    # Aucune fenêtre OpenCV ne doit être sollicitée en mode web.
    monkeypatch.setattr(entry_point, "select_line", lambda *a, **k: pytest.fail("fenêtre OpenCV ouverte"))
    monkeypatch.setattr(entry_point.cv2, "imshow", lambda *a, **k: pytest.fail("fenêtre OpenCV ouverte"))
    if submit is not None:
        original = WebState.wait_calibration

        def _wait(self, poll_seconds=0.5):
            assert self.calibration_info()["phase"] == PHASE_CALIBRATION
            assert self.submit_calibration(submit)[0]
            return original(self, poll_seconds)

        monkeypatch.setattr(WebState, "wait_calibration", _wait)

    def _ctrl_c(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(entry_point.time, "sleep", _ctrl_c)
    code = entry_point.main(
        ["--source", "clip.mp4", "--output-root", str(tmp_path), "--session-id", "web", "--web", *extra]
    )
    assert states and states[0].stats()[1]["etat"] == "termine"
    return code, tmp_path / "web"


def _events(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def test_main_web_calibration_dans_le_navigateur(tmp_path: Path, monkeypatch):
    code, root = _run_web_main(
        tmp_path, monkeypatch, [],
        submit={"mode": "ligne", "p1": [0.25, 0.4], "p2": [0.75, 0.4], "cote_interieur": "positive"},
    )
    assert code == 0
    event = next(e for e in _events(root) if e["type"] == "LINE_VALIDATED")
    assert event["line_origin"] == "web" and event["inside_side"] == "positive"
    assert event["frame_resolution"] == [640, 480]
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["line_origin"] == "web"
    assert calibration["line"]["p1_normalized"] == [0.25, 0.4]


def test_main_web_compter_sans_ligne_dans_le_navigateur(tmp_path: Path, monkeypatch):
    code, root = _run_web_main(tmp_path, monkeypatch, [], submit={"mode": "sans_ligne"})
    assert code == 0
    event = next(e for e in _events(root) if e["type"] == "LINE_DISABLED")
    assert event["line_origin"] == "web_no_line"
    types = [e["type"] for e in _events(root)]
    assert "IN" not in types and "OUT" not in types


@pytest.mark.parametrize("extra, attendu", [
    (["--line", "0.1,0.5,0.9,0.5"], "LINE_VALIDATED"),
    (["--no-line"], "LINE_DISABLED"),
])
def test_main_web_options_court_circuitent_la_calibration(tmp_path: Path, monkeypatch, extra, attendu):
    monkeypatch.setattr(WebState, "wait_calibration", lambda self, *a: pytest.fail("calibration web sollicitée"))
    code, root = _run_web_main(tmp_path, monkeypatch, extra)
    assert code == 0
    assert attendu in [e["type"] for e in _events(root)]


def test_serveur_refuse_un_port_deja_utilise():
    """Sous Windows, SO_REUSEADDR laisserait un second serveur s'attacher en
    silence au même port : le port est sondé avant le démarrage."""
    import socket as _socket

    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    listener.bind((HOST, 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    try:
        assert web_view.port_in_use(port)
        with pytest.raises(OSError, match="déjà utilisé"):
            web_view.start_server(WebState(), port)
    finally:
        listener.close()


def test_serveur_demarre_sur_un_port_libre():
    import socket as _socket

    with _socket.socket() as probe:
        probe.bind((HOST, 0))
        port = probe.getsockname()[1]
    server = web_view.start_server(WebState(), port)
    try:
        assert web_view.port_in_use(port)
    finally:
        server.shutdown()


def test_echec_d_attache_converti_en_oserror(monkeypatch):
    """Werkzeug quitte (SystemExit) quand l'attache au port échoue : converti en
    OSError pour que main affiche son message et sorte proprement (code 2)."""
    import werkzeug.serving

    def _boom(*_args, **_kwargs):
        raise SystemExit(1)

    monkeypatch.setattr(web_view, "port_in_use", lambda port: False)
    monkeypatch.setattr(werkzeug.serving, "make_server", _boom)
    with pytest.raises(OSError, match="impossible"):
        web_view.start_server(WebState(), 8765)

