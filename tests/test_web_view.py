"""Interface web de démonstration (``--web``) : lecture seule de l'état du pipeline.

Vérifie l'objet partagé, les trois routes (client de test Flask) et le rendu
dédié, qui ne trace que la ligne virtuelle.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from geometry import VirtualLine  # noqa: E402
from main import build_argument_parser  # noqa: E402
from web_view import (  # noqa: E402
    LINE_COLOR_BGR,
    WebState,
    create_app,
    encode_jpeg,
    render_line_only,
)


def _line() -> VirtualLine:
    return VirtualLine(p1=(0.1, 0.5), p2=(0.9, 0.5), inside_side="negative")


def _frame(value: int = 40) -> np.ndarray:
    return np.full((120, 160, 3), value, dtype=np.uint8)


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
    seq0, values0 = state.stats()
    assert values0 == {
        "presentes": 0, "entrees": 0, "sorties": 0, "nouvelles": 0,
        "sans_ligne": False, "etat": "en_cours",
    }
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
    state = WebState(no_line=True)
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
    for label in ("Personnes présentes", "Entrées", "Sorties", "Nouvelles présences"):
        assert label in html
    # Entièrement hors ligne : aucune ressource externe.
    assert "http://" not in html and "https://" not in html
    for asset in ("/static/style.css", "/static/app.js"):
        served = client.get(asset)
        assert served.status_code == 200
        assert b"http://" not in served.data and b"https://" not in served.data


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


def test_option_web_et_ecoute_locale_par_defaut():
    args = build_argument_parser().parse_args(["--web"])
    assert args.web is True and args.port == 8000 and args.host == "127.0.0.1"
    args = build_argument_parser().parse_args([])
    assert args.web is False
