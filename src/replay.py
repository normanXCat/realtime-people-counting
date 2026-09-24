"""Relecture d'une session déjà calculée (``--replay <dossier de session>``).

Aucun calcul de détection ni de suivi : la vidéo source est relue à sa vitesse
réelle, avec la ligne et la flèche de ``calibration.json`` et la zone morte à
la largeur relevée pendant la session (:data:`DEAD_ZONE_FILENAME`), dessinées
par :func:`calibration.draw_line_overlay` — le dessin de la fenêtre OpenCV et
de la vue web. Les quatre compteurs suivent ``events.jsonl`` selon ses
horodatages (horloge de la vidéo : ``timestamp_s`` de l'image ``i`` vaut
``i / fps``, comme dans le pipeline).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2

from geometry import VirtualLine

#: Largeur de la zone morte relevée pendant la session (une ligne à chaque
#: changement), écrite par ``main`` à côté de ``events.jsonl``.
DEAD_ZONE_FILENAME = "dead_zone.jsonl"

#: Événements horodatés à l'horloge murale, pas à celle de la vidéo : appliqués
#: à la fin de la relecture.
WALL_CLOCK_EVENTS = frozenset({"SOURCE_END_OF_STREAM", "SOURCE_ERROR", "SESSION_END"})


class ReplayError(RuntimeError):
    """Session illisible ou incomplète : la relecture ne démarre pas."""


@dataclass
class ReplayCounters:
    """Les quatre compteurs affichés, reconstruits à partir du journal."""

    initial: int = 0
    total_in: int = 0
    total_out: int = 0
    total_new: int = 0

    @property
    def confirmed(self) -> int:
        """``initial + IN + NEW - OUT`` (``occupancy_operational``)."""
        return self.initial + self.total_in + self.total_new - self.total_out

    def apply(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "WARMUP_END":
            self.initial = int(event.get("initial_occupancy", self.initial))
        elif kind == "IN":
            self.total_in += 1
        elif kind == "OUT":
            self.total_out += 1
        elif kind == "NEW":
            self.total_new += 1
        if kind in ("OCCUPANCY_SNAPSHOT", "SESSION_END") and "total_in" in event:
            # Valeurs publiées par le pipeline : elles font foi.
            self.total_in = int(event["total_in"])
            self.total_out = int(event["total_out"])
            self.total_new = int(event["total_new"])
            confirmed = int(event.get("occupancy_confirmed", self.confirmed))
            self.initial = confirmed - self.total_in - self.total_new + self.total_out


@dataclass
class ReplaySession:
    root: Path
    source: str
    line: VirtualLine | None
    events: list[dict[str, Any]]
    end_events: list[dict[str, Any]]
    dead_zone: list[tuple[float, float | None]] = field(default_factory=list)
    frames_processed: int | None = None

    def dead_zone_at(self, timestamp_s: float) -> float | None:
        """Largeur relevée à ``timestamp_s`` (dernier relevé antérieur)."""
        value = None
        for when, px in self.dead_zone:
            if when > timestamp_s:
                break
            value = px
        return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_session(root: str | Path, source: str | None = None) -> ReplaySession:
    """Lit ``calibration.json``, ``events.jsonl`` et la zone morte d'une session."""
    root = Path(root)
    calibration_path = root / "calibration.json"
    events_path = root / "events.jsonl"
    for required in (calibration_path, events_path):
        if not required.is_file():
            raise ReplayError(f"fichier de session absent : {required}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    raw_line = calibration.get("line")
    line = None
    if raw_line is not None:
        line = VirtualLine(
            p1=tuple(raw_line["p1_normalized"]),
            p2=tuple(raw_line["p2_normalized"]),
            inside_side=raw_line["inside_side"],
        )
    all_events = _read_jsonl(events_path)
    events = sorted(
        (e for e in all_events if e.get("type") not in WALL_CLOCK_EVENTS),
        key=lambda e: float(e.get("timestamp_s", 0.0)),
    )
    end_events = [e for e in all_events if e.get("type") in WALL_CLOCK_EVENTS]
    dead_zone: list[tuple[float, float | None]] = []
    dead_zone_path = root / DEAD_ZONE_FILENAME
    if dead_zone_path.is_file():
        dead_zone = [
            (float(r["timestamp_s"]), r["dead_zone_px"]) for r in _read_jsonl(dead_zone_path)
        ]
    frames_processed = None
    summary_path = root / "summary.json"
    if summary_path.is_file():
        frames_processed = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "frames_processed"
        )
    return ReplaySession(
        root=root,
        source=str(source or calibration.get("source", "")),
        line=line,
        events=events,
        end_events=end_events,
        dead_zone=dead_zone,
        frames_processed=frames_processed,
    )


def replay(
    session: ReplaySession,
    state: Any,
    *,
    render: Callable[[Any, Any, float | None], Any],
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayCounters:
    """Relit la vidéo à sa vitesse réelle et publie image et compteurs dans ``state``.

    ``state`` : :class:`web_view.WebState` ; ``render`` : rendu de la ligne,
    de la zone morte et de la flèche (``web_view.render_line_overlay``).
    Retourne les compteurs finaux, ceux de la session d'origine.
    """
    capture = cv2.VideoCapture(session.source)
    if not capture.isOpened():
        raise ReplayError(f"vidéo source illisible : {session.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise ReplayError(f"cadence de la vidéo source illisible : {session.source}")
    counters = ReplayCounters()
    state.start_demo(no_line=session.line is None, replay=True)
    pending = iter(session.events)
    upcoming = next(pending, None)
    start = clock()
    index = 0
    try:
        while session.frames_processed is None or index < session.frames_processed:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp_s = index / fps
            while upcoming is not None and float(upcoming.get("timestamp_s", 0.0)) <= timestamp_s:
                counters.apply(upcoming)
                upcoming = next(pending, None)
            image = (
                render(frame, session.line, session.dead_zone_at(timestamp_s))
                if state.has_viewer
                else None
            )
            delay = start + timestamp_s - clock()
            if delay > 0:
                sleep(delay)
            state.publish(
                image,
                confirmed=counters.confirmed,
                total_in=counters.total_in,
                total_out=counters.total_out,
                total_new=counters.total_new,
            )
            index += 1
    finally:
        capture.release()
    while upcoming is not None:
        counters.apply(upcoming)
        upcoming = next(pending, None)
    for event in session.end_events:
        counters.apply(event)
    state.publish(
        None,
        confirmed=counters.confirmed,
        total_in=counters.total_in,
        total_out=counters.total_out,
        total_new=counters.total_new,
    )
    state.finish()
    return counters
