"""Géométrie canonique de la ligne virtuelle et de l'ancrage des personnes.

Ce module est la **seule** implémentation des notions géométriques du système
(audit §2.6 et §2.7 : les versions concurrentes de `tracker.py` et
`occupancy_manager.py` ont été supprimées). Règles appliquées :

- **Aucune distance en pixels absolus** (règle 0.3) : la zone morte est une
  fraction de l'échelle locale (:class:`LocalScale`, hauteur médiane des bbox
  observées), jamais une constante en pixels.
- **Jamais de décision silencieuse en cas d'ambiguïté** (règle 0.4) : un point
  exactement sur la ligne est classé ``indeterminate`` tant que la politique
  ``on_line_policy`` ne dit pas explicitement le contraire ; une échelle locale
  indisponible renvoie ``None`` au lieu d'une valeur par défaut arbitraire.
- **Ancre au bord de l'image** (spec 5.4) : si la bounding box touche un bord
  dans la marge configurée, l'ancre est déclarée non fiable et toute décision de
  franchissement doit être suspendue pour cette frame.

Convention de signe (unique et documentée) : pour un segment orienté
``p1 -> p2`` de vecteur ``d = p2 - p1`` et un point ``q``,

    distance_signee(q) = cross(q - p1, d) / ||d||

Le côté *positif* est celui vers lequel pointe la normale ``(dy, -dx)``. Le côté
``inside_side`` de la configuration sélectionne le demi-plan qui correspond à
l'intérieur de la salle (``negative`` par défaut : c'est le cas d'une caméra
placée à l'intérieur, avec la ligne horizontale et la salle sous la ligne).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, Sequence

import numpy as np

Point = tuple[float, float]
BBox = Sequence[float]


class Zone(Enum):
    """Classification spatiale d'un point par rapport à la ligne virtuelle."""

    INTERIEURE = auto()   # côté salle, au-delà de la zone morte
    MORTE = auto()        # bande tampon (hystérésis) autour de la ligne
    EXTERIEURE = auto()   # côté sortie, au-delà de la zone morte


class Side(Enum):
    """Demi-plan du point, **sans** tenir compte de la zone morte."""

    INTERIEURE = auto()
    EXTERIEURE = auto()
    INDETERMINEE = auto()  # exactement sur la ligne et politique « indeterminate »


class AnchorReliability(Enum):
    """Fiabilité de l'ancre pieds pour la frame courante (spec 5.4)."""

    FIABLE = auto()
    BORD_IMAGE = auto()        # bbox tronquée par un bord : ancre fantôme possible
    SCALE_INDISPONIBLE = auto()  # aucune échelle locale : aucune décision possible
    CHUTE_HAUTEUR = auto()     # réduction de hauteur anormale (occlusion basse, personne assise)



class LineError(ValueError):
    """Ligne virtuelle dégénérée ou trop courte (spec 5.7)."""


# ---------------------------------------------------------------------------
# Distance signée et franchissement
# ---------------------------------------------------------------------------
def signed_perpendicular_distance(pt: Point, line_start: Point, line_end: Point) -> float:
    """Distance orthogonale **signée** du point par rapport à la ligne orientée.

    Implémentation canonique unique. Voir la convention de signe en tête de
    module. Retourne ``0.0`` si la ligne est dégénérée (les deux points
    confondus), ce que :func:`validate_line` interdit en amont.
    """
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-12:
        return 0.0
    return float(((pt[0] - line_start[0]) * dy - (pt[1] - line_start[1]) * dx) / norm)


def sign_of_side(inside_side: str) -> int:
    """Signe de la distance signée correspondant au côté « intérieur »."""
    if inside_side == "positive":
        return 1
    if inside_side == "negative":
        return -1
    raise LineError(f"inside_side inconnu : {inside_side!r}")


def side_of(
    distance_signed: float,
    inside_side: str,
    on_line_policy: str = "indeterminate",
    on_line_epsilon: float = 0.0,
) -> Side:
    """Demi-plan d'un point à partir de sa distance signée.

    ``on_line_epsilon`` vaut 0 par défaut : seul un point *exactement* sur la
    ligne est ambigu. La zone morte, qui gère la proximité, est indépendante.
    """
    if abs(distance_signed) <= on_line_epsilon:
        if on_line_policy == "inside":
            return Side.INTERIEURE
        if on_line_policy == "outside":
            return Side.EXTERIEURE
        return Side.INDETERMINEE
    return Side.INTERIEURE if distance_signed * sign_of_side(inside_side) > 0 else Side.EXTERIEURE


def zone_of(
    distance_signed: float,
    dead_zone_px: float | None,
    inside_side: str = "negative",
) -> Zone | None:
    """Zone hystérétique d'un point (``None`` si l'échelle locale est absente).

    La zone morte est symétrique autour de la ligne : ``dead_zone_px`` vaut
    ``dead_zone_ratio * hauteur_mediane_des_bbox`` (jamais une constante).
    """
    if dead_zone_px is None:
        return None
    if abs(distance_signed) <= dead_zone_px:
        return Zone.MORTE
    return (
        Zone.INTERIEURE
        if distance_signed * sign_of_side(inside_side) > 0
        else Zone.EXTERIEURE
    )


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Intersection stricte des segments [p1,p2] et [p3,p4] (implémentation unique)."""

    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    eps = 1e-9
    o1, o2 = orientation(p1, p2, p3), orientation(p1, p2, p4)
    o3, o4 = orientation(p3, p4, p1), orientation(p3, p4, p2)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return (
        (abs(o1) <= eps and on_segment(p1, p3, p2))
        or (abs(o2) <= eps and on_segment(p1, p4, p2))
        or (abs(o3) <= eps and on_segment(p3, p1, p4))
        or (abs(o4) <= eps and on_segment(p3, p2, p4))
    )


def crossing_direction(
    previous: Point,
    current: Point,
    line_start: Point,
    line_end: Point,
    inside_side: str,
    on_line_policy: str = "indeterminate",
) -> str | None:
    """Direction de franchissement entre deux ancres successives.

    Returns:
        ``"in"`` si le segment franchit la ligne en allant vers l'intérieur,
        ``"out"`` s'il va vers l'extérieur, ``None`` si la ligne n'est pas
        franchie ou si le point d'arrivée est ambigu (politique
        ``indeterminate``).
    """
    if not segments_intersect(previous, current, line_start, line_end):
        return None
    after = side_of(
        signed_perpendicular_distance(current, line_start, line_end),
        inside_side,
        on_line_policy,
    )
    if after is Side.INDETERMINEE:
        return None
    return "in" if after is Side.INTERIEURE else "out"


# ---------------------------------------------------------------------------
# Ancre pieds
# ---------------------------------------------------------------------------
def compute_anchor(box: BBox) -> Point:
    """Ancre pieds : centre bas de la bounding box ``(x_center, y_max)``.

    L'option « vue zénithale » (centre de boîte) a été supprimée lors de
    l'audit : elle était stockée mais jamais lue (audit §2.2 h) et aucun
    scénario du protocole de la section 9 ne l'utilise.
    """
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return ((x1 + x2) / 2.0, y2)


def bbox_height(box: BBox) -> float:
    return float(box[3]) - float(box[1])


def bbox_center(box: BBox) -> Point:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


# ---------------------------------------------------------------------------
# Échelle locale (remplace toute constante en pixels)
# ---------------------------------------------------------------------------
class LocalScale:
    """Hauteur médiane des bounding boxes observées récemment.

    C'est l'échelle locale de référence : la zone morte, la tolérance de
    stabilisation et la marge de la contrainte spatio-temporelle s'expriment
    comme des fractions de cette valeur (règle 0.3).
    """

    def __init__(self, window: int = 61, min_samples: int = 1) -> None:
        if window < 1:
            raise ValueError("La fenêtre d'échelle locale doit être >= 1")
        self.window = int(window)
        self.min_samples = int(min_samples)
        self._heights: deque[float] = deque(maxlen=window)

    def observe(self, height: float) -> None:
        if height > 0:
            self._heights.append(float(height))

    def observe_box(self, box: BBox) -> None:
        self.observe(bbox_height(box))

    def observe_many(self, boxes: Iterable[BBox]) -> None:
        for box in boxes:
            self.observe_box(box)

    @property
    def median_height(self) -> float | None:
        if len(self._heights) < self.min_samples:
            return None
        ordered = sorted(self._heights)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    def ratio_to_px(self, ratio: float) -> float | None:
        """Convertit une fraction de hauteur médiane en pixels (``None`` si absente)."""
        median = self.median_height
        return None if median is None else float(ratio) * median

    @property
    def samples(self) -> int:
        return len(self._heights)

    def reset(self) -> None:
        self._heights.clear()


# ---------------------------------------------------------------------------
# Garde-fou bord d'image
# ---------------------------------------------------------------------------
def edge_margin_px(frame_width: int, frame_height: int, margin_ratio: float) -> float:
    """Marge de bord en pixels : fraction de la plus petite dimension de l'image."""
    return float(margin_ratio) * float(min(frame_width, frame_height))


def anchor_reliability(
    box: BBox,
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
) -> AnchorReliability:
    """L'ancre pieds est-elle exploitable pour cette frame ? (spec 5.4)

    Une bounding box qui touche un bord de l'image dans la marge configurée a
    une ancre pieds qui peut se trouver hors du champ : les décisions de
    franchissement doivent être suspendues tant que la boîte reste au bord.
    """
    margin = edge_margin_px(frame_width, frame_height, margin_ratio)
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if (
        x1 <= margin
        or y1 <= margin
        or x2 >= frame_width - margin
        or y2 >= frame_height - margin
    ):
        return AnchorReliability.BORD_IMAGE
    return AnchorReliability.FIABLE


# ---------------------------------------------------------------------------
# Ligne virtuelle calibrée
# ---------------------------------------------------------------------------
def line_length_normalized(p1: Point, p2: Point) -> float:
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def validate_line(p1: Point, p2: Point, min_length_ratio: float = 0.05) -> None:
    """Refuse une ligne dégénérée ou trop courte (spec 5.7).

    Raises:
        LineError: points identiques, longueur sous le seuil, ou coordonnées
            non normalisées.
    """
    for name, point in (("p1", p1), ("p2", p2)):
        if not (0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0):
            raise LineError(
                f"Ligne invalide : {name}={point} doit être normalisé entre 0 et 1"
            )
    length = line_length_normalized(p1, p2)
    if length <= 1e-9:            raise LineError("Ligne invalide : les deux points sont identiques (dégénérée)")
    if length < min_length_ratio:
        raise LineError(
            f"Ligne invalide : longueur normalisée {length:.4f} < seuil minimal "
            f"{min_length_ratio}"
        )


@dataclass(frozen=True)
class VirtualLine:
    """Ligne virtuelle calibrée : deux points normalisés et un côté intérieur."""

    p1: Point
    p2: Point
    inside_side: str = "negative"
    on_line_policy: str = "indeterminate"
    min_length_ratio: float = 0.05
    frame_width: int = 0
    frame_height: int = 0

    def __post_init__(self) -> None:
        validate_line(self.p1, self.p2, self.min_length_ratio)
        sign_of_side(self.inside_side)  # échoue tôt sur une valeur inconnue

    # -- Conversions ------------------------------------------------------
    def to_pixels(self, width: int, height: int) -> tuple[Point, Point]:
        return (
            (self.p1[0] * width, self.p1[1] * height),
            (self.p2[0] * width, self.p2[1] * height),
        )

    def points_px(self) -> tuple[Point, Point]:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise LineError(
                "Dimensions de frame inconnues : la ligne ne peut pas être exprimée "
                "en pixels"
            )
        return self.to_pixels(self.frame_width, self.frame_height)

    def with_frame_size(self, width: int, height: int) -> "VirtualLine":
        return VirtualLine(
            p1=self.p1,
            p2=self.p2,
            inside_side=self.inside_side,
            on_line_policy=self.on_line_policy,
            min_length_ratio=self.min_length_ratio,
            frame_width=width,
            frame_height=height,
        )

    # -- Mesures ----------------------------------------------------------
    def distance(self, point: Point, width: int, height: int) -> float:
        start, end = self.to_pixels(width, height)
        return signed_perpendicular_distance(point, start, end)

    def side(self, point: Point, width: int, height: int) -> Side:
        return side_of(
            self.distance(point, width, height), self.inside_side, self.on_line_policy
        )

    def zone(
        self, point: Point, width: int, height: int, dead_zone_px: float | None
    ) -> Zone | None:
        return zone_of(
            self.distance(point, width, height), dead_zone_px, self.inside_side
        )

    def crossing(
        self, previous: Point, current: Point, width: int, height: int
    ) -> str | None:
        start, end = self.to_pixels(width, height)
        return crossing_direction(
            previous, current, start, end, self.inside_side, self.on_line_policy
        )

    @property
    def length_normalized(self) -> float:
        return line_length_normalized(self.p1, self.p2)


def median_height_of(boxes: Sequence[BBox]) -> float | None:
    """Utilitaire : hauteur médiane d'une liste de bbox (échelle locale ponctuelle)."""
    heights = sorted(bbox_height(box) for box in boxes)
    heights = [h for h in heights if h > 0]
    if not heights:
        return None
    middle = len(heights) // 2
    if len(heights) % 2:
        return heights[middle]
    return (heights[middle - 1] + heights[middle]) / 2.0


def check_detection_geometry(
    bbox: BBox,
    confidence: float = 1.0,
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 2.5,
    min_height_px: float = 10.0,
    min_width_px: float = 10.0,
) -> tuple[bool, str | None, float]:
    """Vérifie si une détection respecte les contraintes géométriques plausibles.

    Permet d'écarter les faux positifs (artefacts, reflets horizontaux extrêmes)
    tout en acceptant explicitement les personnes assises (ratio W/H jusqu'à 2.0-2.5).

    Returns:
        (is_valid, reason, bbox_wh_ratio)
    """
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    ratio = (w / h) if h > 1e-6 else float("inf")

    if h < min_height_px or w < min_width_px:
        return False, "box_too_small", ratio
    if ratio < min_aspect_ratio or ratio > max_aspect_ratio:
        return False, "aspect_ratio_out_of_range", ratio
    return True, None, ratio


def check_anchor_height_drop(
    current_height: float,
    height_history: Sequence[float],
    max_drop_ratio: float = 0.35,
) -> tuple[bool, float, float]:
    """Détecte une chute anormale et brutale de la hauteur de boîte par rapport à l'historique.

    Returns:
        (is_drop, drop_ratio, reference_height)
    """
    if not height_history:
        return False, 0.0, float(current_height)
    valid_history = [float(h) for h in height_history if float(h) > 0]
    if not valid_history:
        return False, 0.0, float(current_height)
    from statistics import median

    ref_height = float(median(valid_history))
    if ref_height <= 0:
        return False, 0.0, float(current_height)
    drop_ratio = (ref_height - float(current_height)) / ref_height
    is_drop = drop_ratio >= max_drop_ratio
    return is_drop, float(drop_ratio), float(ref_height)


def check_box_width_growth(
    current_width: float,
    width_history: Sequence[float],
    max_growth_ratio: float = 1.6,
    current_height: float | None = None,
    height_history: Sequence[float] | None = None,
) -> tuple[bool, float, float]:
    """Détecte une croissance anormale et brutale de la largeur de boîte
    par rapport à l'historique stabilisé — signal possible de fusion de
    deux personnes proches dans une même détection.

    Returns:
        (is_abnormal, current_ratio, threshold)
    """
    if not width_history:
        return False, 1.0, float(max_growth_ratio)
    valid_history = [float(w) for w in width_history if float(w) > 0]
    if not valid_history:
        return False, 1.0, float(max_growth_ratio)
    from statistics import median

    ref_width = float(median(valid_history))
    if ref_width <= 0:
        return False, 1.0, float(max_growth_ratio)
    current_ratio = float(current_width) / ref_width
    is_abnormal = current_ratio >= float(max_growth_ratio)

    if is_abnormal and current_height is not None and height_history is not None:
        valid_heights = [float(h) for h in height_history if float(h) > 0]
        if valid_heights:
            ref_height = float(median(valid_heights))
            if ref_height > 0:
                height_ratio = float(current_height) / ref_height
                # Si la hauteur augmente également de manière proportionnelle (rapprochement caméra),
                # la croissance est cohérente et ne constitue pas une fusion.
                if height_ratio >= float(max_growth_ratio) * 0.8 and (current_ratio / height_ratio) < 1.3:
                    is_abnormal = False

    return is_abnormal, float(current_ratio), float(max_growth_ratio)


# ---------------------------------------------------------------------------
# Assistance tête : extraction des points-clés COCO (additif)
# ---------------------------------------------------------------------------
#: Index COCO des cinq points-clés de la tête exploitables pour l'assistance
#: de présence. Les 12 points restants (épaules, coudes, poignets, hanches,
#: genoux, chevilles) ne sont pas utilisés ici.
COCO_HEAD_KEYPOINTS: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
}


def extract_head_point(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    keypoints_used: tuple[str, ...],
    min_confidence: float,
) -> tuple[tuple[float, float] | None, float | None]:
    """Renvoie le point moyen des points-clés fiables et la confiance minimale.

    Args:
        keypoints_xy: coordonnées ``(17, 2)`` des points-clés COCO.
        keypoints_conf: confiances ``(17,)`` associées.
        keypoints_used: noms des points-clés à considérer (ex. ``("nose", "left_eye", "right_eye")``).
        min_confidence: seuil en dessous duquel un point-clé est ignoré.

    Returns:
        ``((mean_x, mean_y), min_conf)`` si au moins un point-clé est fiable,
        ``(None, None)`` sinon.
    """
    valid_xs: list[float] = []
    valid_ys: list[float] = []
    confs: list[float] = []
    for name in keypoints_used:
        idx = COCO_HEAD_KEYPOINTS.get(name)
        if idx is None:
            continue
        conf = float(keypoints_conf[idx])
        if conf >= min_confidence:
            valid_xs.append(float(keypoints_xy[idx, 0]))
            valid_ys.append(float(keypoints_xy[idx, 1]))
            confs.append(conf)
    if not confs:
        return None, None
    mean_x = sum(valid_xs) / len(valid_xs)
    mean_y = sum(valid_ys) / len(valid_ys)
    return (mean_x, mean_y), min(confs)

