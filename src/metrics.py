"""Mesure du FPS réel et de la latence par étape (spec 5.3 et 6.5).

Le FPS utilisé par le pipeline provient **exclusivement** de la mesure runtime
(:class:`FpsEstimator`), jamais d'une constante supposée ni d'une réouverture de
la source. C'est lui qui convertit les durées de la configuration (exprimées en
secondes) en frames, et lui qui pilote l'écriture vidéo.

Module sans dépendance externe : testable avec un temps simulé.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

#: Écart relatif au-delà duquel le débit de traitement mesuré est considéré
#: comme divergent du FPS natif déclaré (signal d'un run hors temps réel).
NATIVE_FPS_DIVERGENCE_THRESHOLD = 0.20


def native_video_fps(source: Any) -> float | None:
    """FPS natif déclaré par une source vidéo (``CAP_PROP_FPS``), ou ``None``.

    Sur un fichier traité hors temps réel, c'est le seul référentiel qui garde
    les durées de la configuration alignées sur le **contenu vidéo** plutôt que
    sur le débit CPU. Ne lève jamais : une source caméra (FPS non déclaré) ou
    une erreur d'ouverture renvoie ``None``, et l'appelant se rabat sur
    l'horloge murale.
    """
    try:
        import cv2

        capture = cv2.VideoCapture(source)
    except Exception:
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    except Exception:  # pragma: no cover - backend vidéo en échec
        return None
    finally:
        capture.release()
    return fps if fps > 0 else None

#: Étapes instrumentées séparément (spec 6.5).
LATENCY_STAGES: tuple[str, ...] = (
    "capture",
    "detection",
    "tracking",
    "reid",
    "fsm",
    "render",
    "write",
)


class FpsEstimator:
    """Estime le FPS effectif par moyenne glissante d'intervalles mesurés.

    Le premier tick n'a pas d'intervalle : aucun FPS n'est disponible avant
    deux ticks, et :meth:`require_fps` refuse alors de deviner (règle 0.2 :
    jamais de FPS supposé). Le pipeline ne convertit les durées en frames et
    n'ouvre l'écriture vidéo qu'une fois cette mesure disponible.
    """

    def __init__(self, window: int = 30, native_fps: float | None = None) -> None:
        if window < 2:
            raise ValueError("La fenêtre d'estimation du FPS doit être >= 2")
        self.window = int(window)
        #: FPS natif de la source (``None`` pour une caméra live) : sert de
        #: référence pour détecter un traitement hors temps réel.
        self.native_fps = None if native_fps is None else float(native_fps)
        self._intervals: deque[float] = deque(maxlen=window)
        self._last_timestamp: float | None = None
        self._first_timestamp: float | None = None
        self._ticks = 0

    def tick(self, timestamp_s: float) -> None:
        """Enregistre l'horodatage d'une frame traitée."""
        if self._first_timestamp is None:
            self._first_timestamp = timestamp_s
        if self._last_timestamp is not None:
            delta = timestamp_s - self._last_timestamp
            if delta > 0:
                self._intervals.append(delta)
        self._last_timestamp = timestamp_s
        self._ticks += 1

    @property
    def samples(self) -> int:
        return len(self._intervals)

    @property
    def is_ready(self) -> bool:
        """Vrai dès qu'un intervalle a été mesuré, donc après deux ticks."""
        return len(self._intervals) >= 1

    @property
    def fps(self) -> float | None:
        """FPS moyen glissant, ou ``None`` tant que la mesure n'est pas fiable."""
        if not self._intervals:
            return None
        mean_delta = sum(self._intervals) / len(self._intervals)
        return 1.0 / mean_delta if mean_delta > 0 else None

    @property
    def fps_min(self) -> float | None:
        if not self._intervals:
            return None
        slowest = max(self._intervals)
        return 1.0 / slowest if slowest > 0 else None

    @property
    def fps_median(self) -> float | None:
        if not self._intervals:
            return None
        ordered = sorted(self._intervals)
        middle = ordered[len(ordered) // 2]
        return 1.0 / middle if middle > 0 else None

    @property
    def divergence_ratio(self) -> float | None:
        """Écart relatif entre le FPS mesuré et le FPS natif de référence."""
        measured = self.fps
        if self.native_fps is None or measured is None or self.native_fps <= 0:
            return None
        return abs(measured - self.native_fps) / self.native_fps

    @property
    def diverges_from_native(self) -> bool:
        """Le débit de traitement s'écarte-t-il de plus de 20 % du FPS natif ?"""
        ratio = self.divergence_ratio
        return ratio is not None and ratio > NATIVE_FPS_DIVERGENCE_THRESHOLD

    def divergence_message(self) -> str | None:
        """Message de ``CONFIG_WARNING`` si le run diverge du FPS natif, sinon ``None``."""
        if not self.diverges_from_native or self.native_fps is None:
            return None
        return (
            f"Le débit de traitement mesuré ({(self.fps or 0.0):.2f} ips) diverge "
            f"de plus de {NATIVE_FPS_DIVERGENCE_THRESHOLD * 100:.0f} % du FPS "
            f"natif déclaré ({self.native_fps:.2f} ips) : run hors temps réel non "
            "couvert par les seuils calibrés (track_buffer, grace_period_frames)."
        )

    def require_fps(self) -> float:
        """Retourne le FPS mesuré, ou échoue explicitement s'il n'existe pas."""
        value = self.fps
        if value is None or value <= 0:
            raise RuntimeError(
                "FPS non encore mesurable : au moins deux frames sont nécessaires "
                "(aucune valeur par défaut n'est supposée, règle 0.2)"
            )
        return value

    def elapsed(self) -> float:
        if self._first_timestamp is None or self._last_timestamp is None:
            return 0.0
        return self._last_timestamp - self._first_timestamp


class Timer:
    """Chronomètre réutilisable, utilisé comme gestionnaire de contexte."""

    __slots__ = ("start", "last", "running")

    def __init__(self) -> None:
        self.start = 0.0
        self.last = 0.0
        self.running = False

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.running = True
        return self

    def __exit__(self, *_exc: object) -> None:
        self.last = time.perf_counter() - self.start
        self.running = False

    @property
    def ms(self) -> float:
        return (self.last if not self.running else time.perf_counter() - self.start) * 1000.0


def percentile(values: list[float], fraction: float) -> float:
    """Percentile par interpolation linéaire (équivalent numpy.percentile)."""
    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("La fraction de percentile doit être dans [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass
class LatencyProfiler:
    """Accumule la latence par étape et produit un rapport (spec 6.5)."""

    stages: tuple[str, ...] = LATENCY_STAGES
    samples: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for stage in self.stages:
            self.samples.setdefault(stage, [])

    def record(self, stage: str, duration_ms: float) -> None:
        self.samples.setdefault(stage, []).append(float(duration_ms))

    def record_ms(self, stage: str, milliseconds: float) -> None:
        """Alias explicite : la valeur est déjà en millisecondes."""
        self.record(stage, milliseconds)

    def summary(self) -> dict[str, dict[str, float]]:
        """Rapport par étape : moyenne, médiane, minimum et P95."""
        report: dict[str, dict[str, float]] = {}
        for stage, values in self.samples.items():
            if not values:
                continue
            report[stage] = {
                "mean_ms": round(sum(values) / len(values), 3),
                "median_ms": round(percentile(values, 0.5), 3),
                "min_ms": round(min(values), 3),
                "max_ms": round(max(values), 3),
                "p95_ms": round(percentile(values, 0.95), 3),
                "samples": len(values),
            }
        return report

    def total_mean_ms(self) -> float:
        return sum(entry["mean_ms"] for entry in self.summary().values())


def iter_ticks(timestamps: list[float]) -> Iterator[tuple[int, float]]:
    """Utilitaire de test : énumère (index, horodatage)."""
    for index, timestamp in enumerate(timestamps):
        yield index, timestamp
