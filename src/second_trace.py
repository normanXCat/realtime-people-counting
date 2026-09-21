"""Relevé par seconde vidéo de l'état du pipeline — **outillage de mesure**.

Pourquoi ce module
------------------

Les métriques de vérité terrain (``CLAUDE.md`` §3) comparent le pipeline à une
annotation humaine **par seconde** (``tests/fixtures/ground_truth_*.json``).
Le journal d'événements ne suffit pas : il ne conserve ni la boîte de chaque
personne à un instant donné, ni ``visible_count`` seconde par seconde. Le
harnais de session 0, qui rejouait ``process_frame`` de son côté, a divergé du
système livré (``docs/correctif_occlusion.md`` §13.5) : ce relevé est donc
produit **par le pipeline lui-même**, dans sa boucle, et se contente de **lire**
l'état de :class:`OccupancyManager` après ``process_frame``. Il ne calcule rien,
ne décide rien, ne modifie aucun état.

Ce qu'il écrit, pour la première frame de chaque seconde vidéo (frame
``k × fps_source``, soit la frame que montre ``annotation/t{k+1:02d}.jpg``) :

- ``trace_per_second.jsonl`` : une ligne par seconde — compteurs d'occupation,
  ``visible_count``, ``pistes_vues`` (identifiants BoT-SORT présents dans la
  frame), et pour chaque personne suivie : ``person_id``, identifiant
  technique, état FSM, ``inside_occupancy``, boîte, confiance ;
- ``trace_frames/s{k:02d}.jpg`` : l'image brute avec la boîte de chaque personne
  étiquetée ``id<person_id>``. **Jamais ``P<n>``** : les étiquettes humaines de
  la vérité terrain s'écrivent ``P1`` … ``P13``, et l'overlay du pipeline écrit
  lui aussi ``P<person_id>`` — la confusion serait immédiate.

La seconde est dérivée de l'**index de frame** et du FPS de la source, jamais de
l'horloge du pipeline : la vérité terrain est indexée sur le temps vidéo, quelle
que soit la base de temps interne.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

TRACE_FILENAME = "trace_per_second.jsonl"
FRAMES_DIRNAME = "trace_frames"


class SecondTraceUnavailable(ValueError):
    """Le relevé par seconde n'a pas de sens pour cette source (caméra, FPS inconnu)."""


def source_fps(source: int | str) -> float:
    """FPS nominal d'un **fichier** vidéo. Refuse une caméra : pas de temps vidéo."""
    if isinstance(source, int):
        raise SecondTraceUnavailable(
            "--trace-per-second exige un fichier vidéo : sur caméra, il n'existe "
            "pas de seconde vidéo à rapprocher d'une annotation."
        )
    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0.0:
        raise SecondTraceUnavailable(f"FPS de la source illisible : {source}")
    return fps


def second_of_frame(frame_index: int, fps: float) -> int | None:
    """Seconde vidéo ``k`` si ``frame_index`` (1-based) est la frame ``round(k × fps)``.

    ``frame_index`` est l'index de la boucle de ``main.py``, incrémenté **avant**
    traitement : la première frame vaut 1 et correspond à la frame vidéo 0.
    """
    zero_based = frame_index - 1
    if zero_based < 0:
        return None
    k = int(round(zero_based / fps))
    return k if int(round(k * fps)) == zero_based else None


def _bbox_list(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    return [round(float(v), 1) for v in list(bbox)[:4]]


class SecondTracer:
    """Écrit le relevé par seconde. Lecture seule sur l'état du pipeline."""

    def __init__(self, session_root: Path, fps: float) -> None:
        self.fps = float(fps)
        self.frames_dir = session_root / FRAMES_DIRNAME
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._handle = (session_root / TRACE_FILENAME).open("w", encoding="utf-8")

    def observe(
        self,
        frame_index: int,
        frame: np.ndarray,
        occupancy: Any,
        technical_ids_in_frame: Iterable[int],
    ) -> int | None:
        """Relève la frame si elle ouvre une seconde vidéo. Retourne la seconde ou ``None``."""
        second = second_of_frame(frame_index, self.fps)
        if second is None:
            return None
        persons = []
        for person_id, track in sorted(occupancy.tracks.items()):
            persons.append(
                {
                    "person_id": int(person_id),
                    "technical_track_id": int(track.technical_track_id),
                    "state": track.state.name,
                    "inside_occupancy": bool(track.inside_occupancy),
                    "is_visible": bool(track.is_visible),
                    "seen_this_frame": int(track.last_seen_frame) == int(frame_index),
                    "bbox": _bbox_list(track.bbox),
                    "confidence": round(float(track.confidence), 3),
                }
            )
        technical_ids = sorted({int(t) for t in technical_ids_in_frame})
        row = {
            "second": second,
            "frame_index": int(frame_index),
            "occupancy_operational": int(occupancy.occupancy_operational),
            "occupancy_observed": int(occupancy.occupancy_observed),
            "occupancy_uncertain": int(occupancy.occupancy_uncertain),
            "visible_count": int(occupancy.visible_count),
            "occluded_count": int(occupancy.occluded_count),
            "pistes_vues": len(technical_ids),
            "technical_ids_in_frame": technical_ids,
            "persons": persons,
        }
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()
        cv2.imwrite(
            str(self.frames_dir / f"s{second:02d}.jpg"),
            self._annotate(frame, persons, second),
        )
        return second

    @staticmethod
    def _annotate(frame: np.ndarray, persons: list[dict], second: int) -> np.ndarray:
        image = frame.copy()
        for person in persons:
            if person["bbox"] is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in person["bbox"])
            # Vert : vue dans cette frame ; orange : boîte figée (non vue).
            color = (0, 200, 0) if person["seen_this_frame"] else (0, 140, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"id{person['person_id']} t{person['technical_track_id']}"
            cv2.putText(
                image, label, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )
        cv2.putText(
            image, f"s{second}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return image

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
