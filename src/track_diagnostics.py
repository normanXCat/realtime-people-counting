"""Diagnostics de détection et d'association (perte de piste).

Pourquoi ce module existe
-------------------------

Le chemin nominal du pipeline suppose que chaque frame produit une liste de
détections **avec** identifiant technique BoT-SORT. En pratique, trois
situations très différentes se présentent et elles étaient toutes traitées
comme « pas de détection » :

1. ``boxes`` absent (aucune personne détectée) — c'est une absence réelle ;
2. ``boxes`` présent mais ``boxes.id`` absent : BoT-SORT n'a associé aucune
   piste à des détections qui existent pourtant. Inventer un identifiant
   technique, créer un ``person_id`` ou compter quoi que ce soit dans ce cas
   produirait un double comptage silencieux ;
3. détections présentes mais toutes sous ``tracker.track_high_thresh`` : la
   piste ne tient que par l'association « faible » de BoT-SORT, situation
   typique d'une personne légèrement floue.

Ce module **ne décide rien** : il classe, compte et journalise. Il n'invente
jamais d'identifiant et ne produit ni ``IN``, ni ``OUT``, ni ``NEW``. La
politique d'occupation reste entièrement dans :mod:`occupancy_manager` et
:mod:`fsm`.

Limitation de débit
-------------------

Un événement par frame rendrait ``events.jsonl`` illisible et masquerait les
événements métier. Le **front montant** d'une situation est toujours journalisé,
puis au plus un événement par ``diagnostics.min_interval_seconds`` tant que la
situation dure. Le nombre de frames concernées est publié dans le résumé de
session (``summary.json``) : l'information quantitative n'est donc jamais
perdue, seulement le détail ligne à ligne.
"""

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from config import DiagnosticsConfig
from events import EventSink

_log = logging.getLogger("track_diagnostics")


#: Statuts possibles d'un jeu de boîtes retourné par le tracker.
BOXES_MISSING = "missing"   # ``result.boxes is None``
BOXES_EMPTY = "empty"       # boîtes présentes mais vides
BOXES_PRESENT = "present"   # au moins une boîte

#: Situations reconnues par le diagnostic.
SITUATION_NOMINAL = "nominal"
SITUATION_DETECTION_ABSENT = "detection_absent"
SITUATION_TRACK_ID_ABSENT = "track_id_absent"
SITUATION_LOW_CONFIDENCE = "low_confidence"


def box_count(boxes: Any) -> int:
    """Nombre de boîtes, sans supposer l'interface Ultralytics complète.

    ``boxes`` peut être ``None``, un objet ``Boxes`` d'Ultralytics (itérable,
    avec ``__len__``) ou un double de test qui n'expose que ``xyxy``/``id``/
    ``conf``. Aucune exception ne doit remonter d'ici : un diagnostic qui plante
    serait pire que pas de diagnostic.
    """
    if boxes is None:
        return 0
    try:
        return int(len(boxes))
    except TypeError:
        pass
    for attribute in ("xyxy", "conf", "id"):
        values = getattr(boxes, attribute, None)
        if values is None:
            continue
        tolist = getattr(values, "tolist", None)
        if callable(tolist):
            try:
                return len(tolist())
            except Exception:  # pragma: no cover - double exotique
                continue
    return 0


@dataclass(frozen=True)
class DetectionSnapshot:
    """Ce que la frame courante a réellement fourni à l'association."""

    boxes_status: str = BOXES_EMPTY
    boxes_count: int = 0
    ids_present: bool = False
    confidences: tuple[float, ...] = ()

    @property
    def max_confidence(self) -> float | None:
        return max(self.confidences) if self.confidences else None


class DetectionDiagnostics:
    """Classe une frame et journalise les situations de perte de piste."""

    #: Type d'événement par situation. ``nominal`` n'émet rien.
    EVENT_TYPES: Mapping[str, str] = {
        SITUATION_DETECTION_ABSENT: "DETECTION_ABSENT",
        SITUATION_TRACK_ID_ABSENT: "TRACK_ID_ABSENT",
        SITUATION_LOW_CONFIDENCE: "DETECTION_LOW_CONFIDENCE",
    }

    def __init__(
        self,
        config: DiagnosticsConfig | None = None,
        event_sink: EventSink | None = None,
        track_high_thresh: float = 0.4,
    ) -> None:
        self.config = config or DiagnosticsConfig()
        self.event_sink = event_sink
        self.track_high_thresh = float(track_high_thresh)
        self._last_situation: str | None = None
        self._consecutive = 0
        self._last_emit_s: dict[str, float] = {}
        self._frames_by_situation: dict[str, int] = {}
        self._events_by_situation: dict[str, int] = {}
        self.frames_without_ids = 0
        self.frames_observed = 0
        self.raw_yolo_detections_count = 0
        self.under_threshold_detections_count = 0
        self.rejected_geometry_detections_count = 0

    # -- Classification ----------------------------------------------------
    def classify(self, snapshot: DetectionSnapshot) -> str:
        """Situation de la frame — une seule, par ordre de gravité décroissante."""
        if snapshot.boxes_status != BOXES_PRESENT or snapshot.boxes_count <= 0:
            return SITUATION_DETECTION_ABSENT
        if not snapshot.ids_present:
            return SITUATION_TRACK_ID_ABSENT
        best = snapshot.max_confidence
        if best is not None and best < self.track_high_thresh:
            return SITUATION_LOW_CONFIDENCE
        return SITUATION_NOMINAL

    # -- Boucle ------------------------------------------------------------
    def observe(
        self,
        snapshot: DetectionSnapshot,
        timestamp_s: float,
        frame_index: int,
    ) -> str:
        """Classe la frame, met à jour les compteurs et journalise si nécessaire.

        Returns:
            La situation retenue pour cette frame.
        """
        self.frames_observed += 1
        situation = self.classify(snapshot)
        self._frames_by_situation[situation] = self._frames_by_situation.get(situation, 0) + 1
        if situation == SITUATION_TRACK_ID_ABSENT:
            self.frames_without_ids += 1

        leading_edge = situation != self._last_situation
        if leading_edge:
            self._consecutive = 1
            self._last_situation = situation
            self._last_emit_s.pop(situation, None)
        else:
            self._consecutive += 1

        event_type = self.EVENT_TYPES.get(situation)
        if event_type is None or not self.config.enabled:
            return situation

        last_emit = self._last_emit_s.get(situation)
        if last_emit is not None and (
            timestamp_s - last_emit
        ) < self.config.min_interval_seconds:
            # La situation persiste : elle reste comptée, mais n'est pas
            # ré-écrite à chaque frame (limitation de débit explicite).
            return situation
        self._last_emit_s[situation] = timestamp_s
        self._events_by_situation[situation] = self._events_by_situation.get(situation, 0) + 1
        self._emit(event_type, situation, snapshot, timestamp_s, frame_index)
        return situation

    def _emit(
        self,
        event_type: str,
        situation: str,
        snapshot: DetectionSnapshot,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        if self.event_sink is None:
            return
        fields: dict[str, Any] = {
            "detections": int(snapshot.boxes_count),
            "consecutive_frames": int(self._consecutive),
            "frames_observed": int(self._frames_by_situation.get(situation, 0)),
        }
        if situation == SITUATION_DETECTION_ABSENT:
            fields["reason"] = (
                "boxes_missing"
                if snapshot.boxes_status == BOXES_MISSING
                else "no_boxes_in_frame"
            )
        elif situation == SITUATION_TRACK_ID_ABSENT:
            fields["reason"] = "boxes_present_without_tracker_ids"
        else:
            fields["reason"] = "only_low_confidence_detections"
            fields["max_confidence"] = snapshot.max_confidence
            fields["track_high_thresh"] = self.track_high_thresh
        self.event_sink.emit(event_type, timestamp_s, frame_index, **fields)

    def observe_yolo_detections(
        self,
        boxes: Sequence[Any] | None,
        confidences: Sequence[float] | None,
        confidence_threshold: float,
        frame_index: int,
    ) -> None:
        """Diagnostic avant tout filtrage (spec correction 1).

        Journalise systématiquement (niveau DEBUG) :
        - le nombre brut de détections retournées par YOLO avant application du seuil ;
        - pour chaque détection sous le seuil, son score et sa position.
        """
        if boxes is None or len(boxes) == 0:
            _log.debug("Frame %d: YOLO raw detections count: 0", frame_index)
            return
        total = len(boxes)
        self.raw_yolo_detections_count += total
        _log.debug("Frame %d: YOLO raw detections count: %d", frame_index, total)
        confs = list(confidences) if confidences is not None else [1.0] * total
        for box, conf in zip(boxes, confs):
            c = float(conf)
            if c < confidence_threshold:
                self.under_threshold_detections_count += 1
                pos = [round(float(v), 2) for v in (box[:4] if hasattr(box, "__getitem__") else [])]
                _log.debug(
                    "Frame %d: detection under threshold: score=%.3f, pos=%s",
                    frame_index,
                    c,
                    pos,
                )

    def observe_rejected_geometry(
        self,
        reason: str,
        bbox_wh_ratio: float,
        confidence: float,
        timestamp_s: float,
        frame_index: int,
        bbox: Any = None,
        technical_track_id: int | None = None,
    ) -> None:
        """Journalise une détection écartée par un filtre géométrique (spec correction 1)."""
        self.rejected_geometry_detections_count += 1
        _log.debug(
            "Frame %d: detection rejected by geometric filter: reason=%s, w/h=%.2f, conf=%.3f",
            frame_index,
            reason,
            bbox_wh_ratio,
            confidence,
        )
        if self.event_sink is not None:
            fields: dict[str, Any] = {
                "reason": reason,
                "bbox_wh_ratio": round(float(bbox_wh_ratio), 4),
                "confidence": round(float(confidence), 4),
                "frame": frame_index,
            }
            if bbox is not None:
                fields["bbox"] = [round(float(v), 2) for v in bbox[:4]]
            if technical_track_id is not None:
                fields["technical_track_id"] = int(technical_track_id)
            self.event_sink.emit("DETECTION_REJECTED_GEOMETRY", timestamp_s, frame_index, **fields)

    # -- Résumé ------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Compteurs publiés dans ``summary.json`` (aucune information perdue)."""
        return {
            "enabled": bool(self.config.enabled),
            "min_interval_seconds": float(self.config.min_interval_seconds),
            "frames_observed": self.frames_observed,
            "frames_by_situation": dict(sorted(self._frames_by_situation.items())),
            "events_by_situation": dict(sorted(self._events_by_situation.items())),
            "frames_without_tracker_ids": self.frames_without_ids,
            "raw_yolo_detections": self.raw_yolo_detections_count,
            "under_threshold_detections": self.under_threshold_detections_count,
            "rejected_geometry_detections": self.rejected_geometry_detections_count,
            "track_high_thresh": self.track_high_thresh,
        }

