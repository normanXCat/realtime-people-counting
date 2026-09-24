"""Interface web (``--web``) : calibration dans le navigateur, puis vue de démonstration.

Aucune influence sur le comptage, le suivi ni les seuils :

- **calibration** (si ni ``--line`` ni ``--no-line``) : la page envoie chaque
  clic et chaque touche (C, G, I, Échap) au serveur, qui les applique à une
  :class:`calibration.LineSelection` — la même que la fenêtre OpenCV — et
  renvoie l'image redessinée par les fonctions OpenCV
  (:func:`calibration.draw_mode_question`, :meth:`LineSelection.draw`). La
  page ne dessine rien. Le pipeline attend la validation
  (:meth:`WebState.wait_calibration`) avant de traiter la première image ;
- **démonstration** : :func:`render_line_overlay` (la ligne, la zone morte et
  la flèche, par :func:`calibration.draw_line_overlay`, le dessin même de la
  fenêtre OpenCV ; ni boîtes, ni identifiants, ni HUD), :class:`WebState`
  (objet partagé sous verrou), flux MJPEG et Server-Sent Events des quatre
  compteurs affichés. La relecture (``--replay``) publie dans le même objet.

Le serveur n'écoute que sur 127.0.0.1 : la vidéo montre des personnes
identifiables.
"""

from __future__ import annotations

import json
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from calibration import (
    LineSelection,
    ValidatedLine,
    display_canvas,
    draw_line_overlay,
    draw_mode_question,
)
from geometry import LineError, validate_line

#: Dossier de la page (HTML, CSS, JavaScript), sans ressource externe.
WEB_DIR = Path(__file__).resolve().parent / "web"

#: Adresse d'écoute, fixe : machine locale uniquement.
HOST = "127.0.0.1"

#: Qualité JPEG du flux MJPEG et de l'image de calibration. Présentation seule.
JPEG_QUALITY = 75

#: Largeur maximale des images du flux MJPEG (réduites avant encodage).
#: Présentation seule : le dessin est fait à pleine résolution, puis réduit.
VIDEO_MAX_WIDTH = 960

#: Délai d'attente d'un changement avant un commentaire de maintien SSE (s).
KEEPALIVE_SECONDS = 15.0

#: Sans image nouvelle (fin de vidéo), la dernière est renvoyée à ce rythme (s) :
#: les navigateurs n'affichent une partie MJPEG qu'à l'arrivée de données
#: suivantes, sans quoi la dernière image ne s'afficherait jamais.
VIDEO_RESEND_SECONDS = 1.0

#: Phases de la page.
PHASE_WAITING = "attente"          # serveur prêt, première image pas encore lue
PHASE_CALIBRATION = "calibration"  # choix du mode / tracé de la ligne
PHASE_DEMO = "demonstration"       # traitement en cours ou terminé

#: Étapes affichées entre la validation et la première image traitée.
STEP_LOADING = "chargement_modeles"
STEP_STARTING = "demarrage"

#: Modes reçus du navigateur.
MODE_LINE = "ligne"
MODE_NO_LINE = "sans_ligne"


def render_line_overlay(
    frame: np.ndarray, line: Any | None, dead_zone_px: float | None
) -> np.ndarray:
    """Copie de ``frame`` avec la ligne, la zone morte et la flèche, rien d'autre.

    Dessin délégué à :func:`calibration.draw_line_overlay`, celui de la fenêtre
    OpenCV. ``line`` à ``None`` (mode sans ligne) renvoie l'image intacte.
    L'image source n'est jamais modifiée.
    """
    rendered = frame.copy()
    if line is not None:
        draw_line_overlay(rendered, line, dead_zone_px)
    return rendered


def encode_jpeg(frame: np.ndarray, max_width: int | None = None) -> bytes | None:
    """JPEG (qualité :data:`JPEG_QUALITY`), réduit à ``max_width`` si plus large."""
    width = frame.shape[1]
    if max_width is not None and width > max_width:
        height = int(round(frame.shape[0] * max_width / width))
        # INTER_LINEAR : 3,3 ms par image de fort_occ4 contre 5,5 ms en INTER_AREA,
        # même poids (51 Kio), à peine plus que l'ancien encodage pleine taille.
        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_LINEAR)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def _point(raw: Any, name: str) -> tuple[float, float]:
    try:
        x, y = (float(value) for value in raw)
    except (TypeError, ValueError):
        raise LineError(f"Point {name} manquant ou mal formé : deux nombres attendus.") from None
    return (x, y)


class WebState:
    """État partagé entre le pipeline (écrivain) et le serveur (lecteur)."""

    def __init__(self, *, no_line: bool = False) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._phase = PHASE_WAITING
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        #: JPEG de l'image courante, encodé une seule fois (au premier lecteur).
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._viewers = 0
        self._stats: dict[str, Any] = {}
        self._stats_seq = 0
        self._set_mode_stats(no_line)
        # Calibration web.
        self._calibration_jpeg: bytes | None = None
        self._calibration_render: np.ndarray | None = None
        self._calibration_version = 0
        self._canvas: np.ndarray | None = None
        self._selection: LineSelection | None = None
        self._choosing_mode = True
        self._frame_size = (0, 0)
        self._inside_side = "negative"
        self._min_length_ratio = 0.05
        self._calibration_result: tuple[str, ValidatedLine | None] | None = None

    # -- Calibration (navigateur) ---------------------------------------------
    def open_calibration(
        self, frame: np.ndarray, *, inside_side: str, min_length_ratio: float
    ) -> None:
        """Première image affichée dans la page, en attente du choix de l'opérateur.

        Même état initial que :func:`calibration.select_line` : image à la taille
        de la fenêtre de sélection, question « ligne ou pas » posée d'abord.
        """
        canvas = display_canvas(frame)
        selection = LineSelection(
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
            display_width=int(canvas.shape[1]),
            display_height=int(canvas.shape[0]),
            inside_side=inside_side,
            min_length_ratio=min_length_ratio,
        )
        selection.set_message("Cliquer l'extremite 1 de la ligne.", error=False)
        with self._cond:
            self._canvas = canvas
            self._selection = selection
            self._choosing_mode = True
            self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            self._inside_side = inside_side
            self._min_length_ratio = float(min_length_ratio)
            self._render_calibration()
            self._phase = PHASE_CALIBRATION
            self._update_stats(phase=PHASE_CALIBRATION)
            self._cond.notify_all()

    def _render_calibration(self) -> None:
        """Redessine l'aperçu par les fonctions OpenCV (appelé sous verrou)."""
        if self._canvas is None or self._selection is None:
            return
        if self._choosing_mode:
            image = draw_mode_question(self._canvas.copy())
        else:
            image = self._selection.draw(self._canvas.copy())
        self._calibration_render = image
        self._calibration_jpeg = encode_jpeg(image)
        self._calibration_version += 1

    def calibration_info(self) -> dict[str, Any]:
        with self._cond:
            width, height = self._frame_size
            selection = self._selection
            return {
                "phase": self._phase,
                "largeur": width,
                "hauteur": height,
                "cote_interieur": selection.inside_side if selection else self._inside_side,
                "longueur_min": self._min_length_ratio,
                "version": self._calibration_version,
                "question": self._choosing_mode,
                "points": len(selection.clicks_px) if selection else 0,
            }

    def calibration_image(self) -> bytes | None:
        with self._cond:
            return self._calibration_jpeg

    def calibration_render(self) -> np.ndarray | None:
        """Dernier aperçu de calibration, avant encodage (tests)."""
        with self._cond:
            return self._calibration_render

    def calibration_action(self, data: Any) -> tuple[bool, str]:
        """Applique un clic ou une touche reçus du navigateur. Renvoie ``(accepté, message)``.

        Mêmes effets que dans la fenêtre OpenCV (:func:`calibration.select_line`) :
        « ligne » (O), « sans_ligne » (N), « clic » (coordonnées normalisées dans
        l'image affichée), « valider » (C), « recommencer » (G), « inverser » (I).
        « retour » (Échap) revient à la question « ligne ou pas », ligne effacée —
        dans la fenêtre OpenCV, Échap annule le programme.
        """
        if not isinstance(data, dict):
            return False, "Requête invalide."
        action = data.get("action")
        with self._cond:
            selection = self._selection
            if (
                self._phase != PHASE_CALIBRATION
                or self._calibration_result is not None
                or selection is None
            ):
                return False, "La calibration n'est pas attendue (déjà validée ou pas encore prête)."
            accepted, message = True, ""
            if action == "ligne":
                self._choosing_mode = False
            elif action == "sans_ligne":
                self._calibration_result = (MODE_NO_LINE, None)
                message = "Calibration validée : traitement lancé."
            elif action == "retour":
                selection.reset()
                selection.set_message("Cliquer l'extremite 1 de la ligne.", error=False)
                self._choosing_mode = True
            elif self._choosing_mode:
                return False, "Choisissez d'abord : avec ou sans ligne."
            elif action == "clic":
                try:
                    x, y = _point((data.get("x"), data.get("y")), "cliqué")
                except LineError as error:
                    return False, str(error)
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    return False, "Point hors de l'image : cliquez à l'intérieur de l'image."
                accepted = selection.click(x * selection.display_width, y * selection.display_height)
                message = selection.message
            elif action == "valider":
                try:
                    self._calibration_result = (MODE_LINE, selection.confirm())
                    message = "Calibration validée : traitement lancé."
                except LineError as error:
                    selection.set_message(f"Ligne refusee : {error}", error=True)
                    accepted, message = False, _message_fr(str(error), self._min_length_ratio)
            elif action == "recommencer":
                selection.reset()
                message = selection.message
            elif action == "inverser":
                selection.invert_inside_side()
                message = selection.message
            else:
                return False, "Action inconnue."
            self._render_calibration()
            self._cond.notify_all()
            return accepted, message

    def submit_calibration(self, data: Any) -> tuple[bool, str]:
        """Valide une ligne transmise directement (sans clics). Renvoie ``(accepté, message)``.

        Mêmes règles que la calibration OpenCV (:func:`geometry.validate_line`) :
        coordonnées normalisées, points distincts, longueur minimale.
        """
        with self._cond:
            if self._phase != PHASE_CALIBRATION or self._calibration_result is not None:
                return False, "La calibration n'est pas attendue (déjà validée ou pas encore prête)."
            width, height = self._frame_size
            min_ratio = self._min_length_ratio
        if not isinstance(data, dict):
            return False, "Requête invalide."
        mode = data.get("mode")
        if mode == MODE_NO_LINE:
            result: tuple[str, ValidatedLine | None] = (MODE_NO_LINE, None)
        elif mode == MODE_LINE:
            inside_side = data.get("cote_interieur")
            if inside_side not in ("positive", "negative"):
                return False, "Côté intérieur inconnu."
            try:
                p1 = _point(data.get("p1"), "1")
                p2 = _point(data.get("p2"), "2")
                validate_line(p1, p2, min_ratio)
            except LineError as error:
                return False, _message_fr(str(error), min_ratio)
            display_width = data.get("largeur_affichee")
            try:
                display_scale = float(display_width) / width if display_width else 1.0
            except (TypeError, ValueError, ZeroDivisionError):
                display_scale = 1.0
            result = (
                MODE_LINE,
                ValidatedLine(
                    p1=p1,
                    p2=p2,
                    inside_side=inside_side,
                    frame_width=width,
                    frame_height=height,
                    display_scale=display_scale,
                    clicks_px=((p1[0] * width, p1[1] * height), (p2[0] * width, p2[1] * height)),
                    confirmed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
        else:
            return False, "Mode inconnu : « ligne » ou « sans_ligne » attendu."
        with self._cond:
            if self._calibration_result is not None:
                return False, "La calibration a déjà été validée."
            self._calibration_result = result
            self._cond.notify_all()
        return True, "Calibration validée : traitement lancé."

    def wait_calibration(self, poll_seconds: float = 0.5) -> tuple[str, ValidatedLine | None]:
        """Bloque jusqu'à la validation (attente fractionnée : Ctrl+C reste possible)."""
        while True:
            with self._cond:
                if self._calibration_result is not None:
                    return self._calibration_result
                self._cond.wait(timeout=poll_seconds)

    # -- Démonstration (pipeline) ---------------------------------------------
    def start_demo(self, *, no_line: bool, replay: bool = False) -> None:
        """Passage à la vue de démonstration, une fois le mode connu."""
        with self._cond:
            self._phase = PHASE_DEMO
            self._set_mode_stats(no_line)
            self._stats["relecture"] = bool(replay)
            self._cond.notify_all()

    def set_step(self, step: str | None) -> None:
        """Étape affichée avant la première image (chargement, démarrage)."""
        with self._cond:
            self._update_stats(etape=step)
            self._cond.notify_all()

    def _set_mode_stats(self, no_line: bool) -> None:
        self._stats = {
            "presentes": self._stats.get("presentes", 0),
            "entrees": None if no_line else self._stats.get("entrees") or 0,
            "sorties": None if no_line else self._stats.get("sorties") or 0,
            "nouvelles": self._stats.get("nouvelles", 0),
            "sans_ligne": bool(no_line),
            "etat": self._stats.get("etat", "en_cours"),
            "phase": self._phase,
            "etape": self._stats.get("etape"),
            "relecture": self._stats.get("relecture", False),
        }
        self._stats_seq += 1

    @property
    def has_viewer(self) -> bool:
        """Un navigateur lit-il le flux vidéo ? Sans lecteur, rien n'est rendu."""
        with self._cond:
            return self._viewers > 0

    def add_viewer(self) -> None:
        with self._cond:
            self._viewers += 1

    def remove_viewer(self) -> None:
        with self._cond:
            self._viewers = max(0, self._viewers - 1)

    def publish(
        self,
        frame: np.ndarray | None,
        *,
        confirmed: int,
        total_in: int,
        total_out: int,
        total_new: int,
    ) -> None:
        """Dépose l'image rendue et les quatre compteurs (IN/OUT ignorés sans ligne).

        ``frame`` à ``None`` (aucun lecteur du flux) : compteurs seuls, la
        dernière image reste en place.
        """
        with self._cond:
            if frame is not None:
                self._frame = frame
                self._frame_seq += 1
            no_line = self._stats["sans_ligne"]
            self._update_stats(
                presentes=int(confirmed),
                entrees=None if no_line else int(total_in),
                sorties=None if no_line else int(total_out),
                nouvelles=int(total_new),
                etape=None,
            )
            self._cond.notify_all()

    def finish(self) -> None:
        """Fin de la vidéo : la dernière image et les valeurs finales restent servies."""
        with self._cond:
            self._update_stats(etat="termine", etape=None)
            self._cond.notify_all()

    def _update_stats(self, **values: Any) -> None:
        changed = {k: v for k, v in values.items() if self._stats.get(k) != v}
        if changed:
            self._stats.update(changed)
            self._stats_seq += 1

    # -- Lecture (serveur) -----------------------------------------------------
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
        """Attend une image postérieure à ``after_seq`` (ou le délai).

        Sans aucune image encore publiée, l'attente dure jusqu'au délai : un
        retour immédiat ferait tourner le flux en boucle active et priverait le
        pipeline de l'interpréteur (défaut mesuré : ~10 min avant la 1re image).
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame is not None and self._frame_seq != after_seq,
                timeout=timeout,
            )
            return self._frame_seq, self._frame

    def wait_jpeg(self, after_seq: int, timeout: float) -> tuple[int, bytes | None]:
        """Comme :meth:`wait_frame`, mais renvoie le JPEG de la **dernière** image.

        Encodé une seule fois par image, quel que soit le nombre de lecteurs,
        réduit à :data:`VIDEO_MAX_WIDTH`. Aucune file : une image remplacée
        avant d'être lue n'est jamais encodée.
        """
        seq, frame = self.wait_frame(after_seq, timeout)
        if frame is None:
            return seq, None
        with self._cond:
            if self._jpeg_seq == seq and self._jpeg is not None:
                return seq, self._jpeg
        payload = encode_jpeg(frame, VIDEO_MAX_WIDTH)
        with self._cond:
            if seq >= self._jpeg_seq:
                self._jpeg, self._jpeg_seq = payload, seq
        return seq, payload


def _message_fr(message: str, min_ratio: float) -> str:
    """Message d'erreur de :func:`validate_line`, reformulé pour l'opérateur."""
    if "identiques" in message:
        return "Les deux points sont confondus : cliquez deux points distincts."
    if "longueur normalisée" in message:
        return (
            "Ligne trop courte : tracez une ligne plus longue "
            f"(au moins {min_ratio:.0%} de la diagonale normalisée)."
        )
    if "normalisé" in message:
        return "Point hors de l'image : cliquez à l'intérieur de l'image."
    return message


def create_app(
    state: WebState,
    *,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
    video_resend_seconds: float = VIDEO_RESEND_SECONDS,
):
    """Application Flask ; seules les routes POST /calibration[/action] écrivent (choix de l'opérateur)."""
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name: str):
        return send_from_directory(WEB_DIR, name)

    @app.get("/calibration")
    def calibration_info():
        return jsonify(state.calibration_info())

    @app.get("/calibration/image")
    def calibration_image():
        payload = state.calibration_image()
        if payload is None:
            return Response("Image de calibration indisponible.", status=404)
        return Response(payload, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/calibration/action")
    def calibration_action():
        # JSON exigé, comme POST /calibration.
        if not request.is_json:
            return jsonify({"accepte": False, "message": "Requête JSON attendue."}), 415
        accepted, message = state.calibration_action(request.get_json(silent=True))
        return jsonify({"accepte": accepted, "message": message, **state.calibration_info()})

    @app.post("/calibration")
    def calibration_submit():
        # JSON exigé : une page d'une autre origine ne peut pas l'envoyer sans
        # requête préalable CORS, que ce serveur n'autorise pas.
        if not request.is_json:
            return jsonify({"accepte": False, "message": "Requête JSON attendue."}), 415
        accepted, message = state.submit_calibration(request.get_json(silent=True))
        return jsonify({"accepte": accepted, "message": message}), (200 if accepted else 400)

    @app.get("/video")
    def video():
        def stream() -> Iterator[bytes]:
            # Chaque partie est suivie de sa délimitation ; sans image nouvelle,
            # la dernière est renvoyée (voir VIDEO_RESEND_SECONDS). Le pipeline
            # ne rend les images que tant qu'un lecteur est connecté.
            state.add_viewer()
            try:
                yield b"--frame\r\n"
                seq = 0
                part: bytes | None = None
                while True:
                    new_seq, payload = state.wait_jpeg(seq, video_resend_seconds)
                    if payload is not None and new_seq != seq:
                        seq = new_seq
                        part = (
                            b"Content-Type: image/jpeg\r\nContent-Length: "
                            + str(len(payload)).encode()
                            + b"\r\n\r\n" + payload + b"\r\n--frame\r\n"
                        )
                    if part is not None:
                        yield part
            finally:
                state.remove_viewer()

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


def port_in_use(port: int) -> bool:
    """Un serveur répond-il déjà sur ``HOST:port`` ?"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((HOST, port)) == 0


def start_server(state: WebState, port: int) -> Any:
    """Démarre le serveur (127.0.0.1) dans un thread démon ; renvoie le serveur.

    Raises:
        OSError: port déjà utilisé. Sous Windows, ``SO_REUSEADDR`` (activé par
            Werkzeug) laisse un second serveur s'attacher silencieusement au même
            port : le navigateur atteindrait alors l'autre instance. Le port est
            donc sondé avant.
    """
    from werkzeug.serving import make_server

    if port_in_use(port):
        raise OSError(f"le port {port} est déjà utilisé par un autre serveur (choisir --port)")
    try:
        server = make_server(HOST, port, create_app(state), threaded=True)
    except SystemExit as error:  # Werkzeug quitte au lieu de lever sur échec d'attache
        raise OSError(f"attache au port {port} impossible") from error
    thread = threading.Thread(target=server.serve_forever, name="web-view", daemon=True)
    thread.start()
    return server
