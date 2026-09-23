"""Interface web de démonstration (``--web``) : lecture seule de l'état du pipeline.

Trois éléments, sans aucune influence sur le comptage, le suivi ni les seuils :

- :func:`render_line_only` : rendu **dédié** de la vidéo, qui ne trace que la
  ligne virtuelle (ni boîtes, ni identifiants, ni ancres, ni compteurs). Le
  rendu OpenCV habituel (``OccupancyManager.draw_overlay``) reste inchangé ;
- :class:`WebState` : objet partagé, protégé par un verrou, où la boucle du
  pipeline dépose la dernière image rendue et les quatre compteurs affichés ;
- :func:`create_app` : petit serveur Flask. ``/`` sert la page (``src/web/``),
  ``/video`` un flux MJPEG, ``/stats`` des Server-Sent Events qui ne
  transmettent les valeurs que lorsqu'elles changent.

Le serveur écoute sur 127.0.0.1 par défaut : la vidéo montre des personnes
identifiables, elle n'est exposée sur le réseau que sur option explicite
(``--host``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

#: Dossier de la page (HTML, CSS, JavaScript), sans ressource externe.
WEB_DIR = Path(__file__).resolve().parent / "web"

#: Couleur et épaisseur de la ligne dans le rendu dédié (BGR). Présentation seule.
LINE_COLOR_BGR = (0, 200, 255)
LINE_THICKNESS = 4

#: Qualité JPEG du flux MJPEG. Présentation seule.
JPEG_QUALITY = 80

#: Délai d'attente d'un changement avant un commentaire de maintien SSE (s).
KEEPALIVE_SECONDS = 15.0

#: Sans image nouvelle (fin de vidéo), la dernière est renvoyée à ce rythme (s) :
#: les navigateurs n'affichent une partie MJPEG qu'à l'arrivée de données
#: suivantes, sans quoi la dernière image ne s'afficherait jamais.
VIDEO_RESEND_SECONDS = 1.0


def render_line_only(frame: np.ndarray, line: Any | None) -> np.ndarray:
    """Copie de ``frame`` avec, pour seul ajout, la ligne virtuelle.

    ``line`` expose ``to_pixels(width, height)`` (``VirtualLine``) ; ``None``
    (mode sans ligne) renvoie l'image intacte. L'image source n'est jamais
    modifiée.
    """
    rendered = frame.copy()
    if line is not None:
        height, width = rendered.shape[:2]
        start, end = line.to_pixels(width, height)
        cv2.line(
            rendered,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            LINE_COLOR_BGR,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )
    return rendered


class WebState:
    """Dernière image et compteurs affichés, partagés entre pipeline et serveur."""

    def __init__(self, *, no_line: bool = False) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._stats: dict[str, Any] = {
            "presentes": 0,
            "entrees": None if no_line else 0,
            "sorties": None if no_line else 0,
            "nouvelles": 0,
            "sans_ligne": bool(no_line),
            "etat": "en_cours",
        }
        self._stats_seq = 0

    # -- Côté pipeline -------------------------------------------------------
    def publish(
        self,
        frame: np.ndarray,
        *,
        confirmed: int,
        total_in: int,
        total_out: int,
        total_new: int,
    ) -> None:
        """Dépose l'image rendue et les quatre compteurs (IN/OUT ignorés sans ligne)."""
        with self._cond:
            self._frame = frame
            self._frame_seq += 1
            no_line = self._stats["sans_ligne"]
            self._update_stats(
                presentes=int(confirmed),
                entrees=None if no_line else int(total_in),
                sorties=None if no_line else int(total_out),
                nouvelles=int(total_new),
            )
            self._cond.notify_all()

    def finish(self) -> None:
        """Fin de la vidéo : la dernière image et les valeurs finales restent servies."""
        with self._cond:
            self._update_stats(etat="termine")
            self._cond.notify_all()

    def _update_stats(self, **values: Any) -> None:
        changed = {k: v for k, v in values.items() if self._stats.get(k) != v}
        if changed:
            self._stats.update(changed)
            self._stats_seq += 1

    # -- Côté serveur --------------------------------------------------------
    def stats(self) -> tuple[int, dict[str, Any]]:
        with self._cond:
            return self._stats_seq, dict(self._stats)

    def frame(self) -> tuple[int, np.ndarray | None]:
        with self._cond:
            return self._frame_seq, self._frame

    def wait_stats(self, after_seq: int, timeout: float) -> tuple[int, dict[str, Any]]:
        """Attend une version des compteurs postérieure à ``after_seq`` (ou le délai)."""
        with self._cond:
            self._cond.wait_for(lambda: self._stats_seq != after_seq, timeout=timeout)
            return self._stats_seq, dict(self._stats)

    def wait_frame(self, after_seq: int, timeout: float) -> tuple[int, np.ndarray | None]:
        """Attend une image postérieure à ``after_seq`` (ou le délai)."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_seq != after_seq, timeout=timeout)
            return self._frame_seq, self._frame


def encode_jpeg(frame: np.ndarray) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def create_app(
    state: WebState,
    *,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
    video_resend_seconds: float = VIDEO_RESEND_SECONDS,
):
    """Application Flask en lecture seule sur ``state``."""
    from flask import Flask, Response, send_from_directory  # import tardif : --web seulement

    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name: str):
        return send_from_directory(WEB_DIR, name)

    @app.get("/video")
    def video():
        def stream() -> Iterator[bytes]:
            # Chaque partie est suivie de sa délimitation ; sans image nouvelle,
            # la dernière est renvoyée (voir VIDEO_RESEND_SECONDS).
            yield b"--frame\r\n"
            seq = -1
            part: bytes | None = None
            while True:
                new_seq, frame = state.wait_frame(seq, video_resend_seconds)
                if frame is not None and new_seq != seq:
                    seq = new_seq
                    payload = encode_jpeg(frame)
                    if payload is not None:
                        part = (
                            b"Content-Type: image/jpeg\r\nContent-Length: "
                            + str(len(payload)).encode()
                            + b"\r\n\r\n" + payload + b"\r\n--frame\r\n"
                        )
                if part is not None:
                    yield part

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stats")
    def stats():
        def stream() -> Iterator[str]:
            seq, values = state.stats()
            yield f"data: {json.dumps(values)}\n\n"
            while True:
                new_seq, new_values = state.wait_stats(seq, keepalive_seconds)
                if new_seq == seq:
                    yield ": maintien\n\n"
                    continue
                seq = new_seq
                if new_values != values:
                    values = new_values
                    yield f"data: {json.dumps(values)}\n\n"

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def start_server(state: WebState, host: str, port: int) -> Any:
    """Démarre le serveur dans un thread démon ; renvoie le serveur (``shutdown()``)."""
    from werkzeug.serving import make_server

    server = make_server(host, port, create_app(state), threaded=True)
    thread = threading.Thread(target=server.serve_forever, name="web-view", daemon=True)
    thread.start()
    return server
