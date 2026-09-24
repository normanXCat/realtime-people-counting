"""Sélection **manuelle obligatoire** de la ligne virtuelle de comptage.

La ligne qui sépare l'intérieur de l'extérieur n'est jamais déduite, proposée ni
remplacée par une valeur par défaut : elle est définie par l'opérateur sur la
première image de la source, puis validée explicitement. Sans cette validation,
le comptage ne démarre pas (exigence « empêcher le lancement du comptage tant
que la ligne n'a pas été validée »).

Séquence d'interaction :

1. la **première image** de la caméra ou du fichier est affichée dans une
   fenêtre OpenCV ;
2. un premier clic place l'extrémité 1, un second l'extrémité 2 ; les deux points
   et la ligne apparaissent immédiatement ;
3. ``C`` confirme la ligne, ``G`` réinitialise la sélection, ``I`` inverse la
   définition intérieur/extérieur, ``Échap`` (ou la fermeture de la fenêtre)
   annule proprement le programme.

Les coordonnées cliquées sont converties en **coordonnées normalisées** entre 0
et 1 : la ligne reste identique quelle que soit la résolution d'affichage ou de
traitement (règle 0.3 : jamais de pixels absolus dans la logique métier). La
ligne validée est ensuite l'unique ligne utilisée pour le dessin, le calcul de
la distance signée, la détection des franchissements et le comptage.

Le module est volontairement séparé de :mod:`geometry` (mesures pures) et de
:mod:`main` (orchestration) : :class:`LineSelection` ne dépend d'aucune fenêtre
et reste donc testable sans écran.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import cv2
import numpy as np

from geometry import LineError, Point, line_length_normalized, sign_of_side, validate_line

#: Touches acceptées dans la fenêtre de sélection (majuscules et minuscules).
KEY_CONFIRM = (ord("c"), ord("C"))
KEY_RESET = (ord("g"), ord("G"))
KEY_INVERT = (ord("i"), ord("I"))
#: Démarrage explicite SANS ligne (vidéo sans porte) : effectif = warm-up + NEW.
KEY_NO_LINE = (ord("n"), ord("N"))
#: Réponse « O » à la question de démarrage : tracer une ligne virtuelle.
KEY_MODE_LINE = (ord("o"), ord("O"))

#: Question posée à l'ouverture de la fenêtre, avant tout clic.
MODE_QUESTION = "O : tracer une ligne virtuelle  —  N : compter sans ligne"

#: Issues de la question de démarrage (voir :func:`mode_for_key`).
MODE_LINE = "ligne"
MODE_NO_LINE = "sans_ligne"
MODE_CANCEL = "annuler"
KEY_CANCEL = 27  # Échap

#: Durée d'attente entre deux itérations de la boucle d'événements (ms).
POLL_INTERVAL_MS = 25

#: Résolution maximale d'affichage de la première image (côté le plus long).
MAX_DISPLAY_SIDE = 1080

#: Nom de fenêtre par défaut, en ASCII strict (voir :func:`sanitize_window_name`).
DEFAULT_WINDOW_NAME = "Selection de la ligne de comptage"


class CalibrationError(RuntimeError):
    """Échec de la sélection de ligne : le comptage ne doit pas démarrer."""


class CalibrationCancelled(CalibrationError):
    """Sélection annulée par l'opérateur (Échap ou fermeture de la fenêtre)."""


class CalibrationUnavailable(CalibrationError):
    """Aucune fenêtre ne peut être affichée : la ligne ne peut pas être définie."""


class SourceUnreadable(CalibrationError):
    """La première image de la source n'a pas pu être lue (source inouvrable)."""


def _clean_ascii(text: str) -> str:
    """Remplace les caractères accentués et typographiques pour cv2.putText (font Hershey)."""
    table = str.maketrans({
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "à": "a", "â": "a", "ä": "a", "À": "A", "Â": "A",
        "ô": "o", "ö": "o", "Ô": "O", "Ö": "O",
        "î": "i", "ï": "i", "Î": "I", "Ï": "I",
        "ù": "u", "û": "u", "ü": "u", "Ù": "U", "Û": "U",
        "ç": "c", "Ç": "C",
        "«": '"', "»": '"', "“": '"', "”": '"',
        "‘": "'", "’": "'", "—": "-", "–": "-", "…": "...",
    })
    return text.translate(table)


def sanitize_window_name(name: str) -> str:
    """Retourne un nom de fenêtre **ASCII**, seul utilisable par le backend Qt.

    Le backend Qt d'OpenCV 5.0 enregistre la fenêtre sous son nom et ne la
    retrouve plus dès que ce nom contient un caractère non ASCII : tout appel à
    ``cv2.setMouseCallback`` échoue alors en « (-27:Null pointer) NULL window
    handler in function 'setMouseCallbackImpl' », et la sélection manuelle de la
    ligne — obligatoire pour tout le comptage — devient impossible. Un simple
    tiret cadratin (« — ») ou un « é » dans le nom du titre suffisait à bloquer
    le démarrage.

    La translittération (accents, tirets typographiques) conserve le sens du
    titre ; les caractères non translittérables deviennent « ? ». Un nom vide
    retombe sur :data:`DEFAULT_WINDOW_NAME`, jamais sur une fenêtre sans nom.
    """
    if not isinstance(name, str):
        raise CalibrationError(
            f"Nom de fenêtre invalide : {name!r} (chaîne de caractères attendue)"
        )
    ascii_name = _clean_ascii(name).encode("ascii", "replace").decode("ascii")
    ascii_name = " ".join(ascii_name.split())
    return ascii_name or DEFAULT_WINDOW_NAME


def display_available() -> bool:
    """Un affichage graphique est-il disponible sur cette machine ?

    Implémentation unique : ``main`` s'appuie sur cette fonction pour décider du
    mode headless, et :func:`select_line` pour refuser un démarrage sans ligne.
    """
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


@dataclass(frozen=True)
class ValidatedLine:
    """Ligne effectivement validée par l'opérateur, avec sa traçabilité.

    Attributes:
        p1: extrémité 1, coordonnées **normalisées** (0-1).
        p2: extrémité 2, coordonnées **normalisées** (0-1).
        inside_side: côté intérieur retenu par l'opérateur (``positive`` ou
            ``negative``, au sens de :func:`geometry.sign_of_side`).
        frame_width/frame_height: résolution de la première image de la source.
        display_scale: facteur appliqué pour l'affichage (1.0 si aucun).
        clicks_px: clics bruts dans la fenêtre, pour audit.
    """

    p1: Point
    p2: Point
    inside_side: str
    frame_width: int
    frame_height: int
    display_scale: float
    clicks_px: tuple[Point, Point] = ((0.0, 0.0), (0.0, 0.0))
    confirmed_at: str = ""

    @property
    def length_ratio(self) -> float:
        """Longueur de la ligne en unités normalisées."""
        return line_length_normalized(self.p1, self.p2)

    def to_pixels(self, width: int, height: int) -> tuple[Point, Point]:
        return ((self.p1[0] * width, self.p1[1] * height), (self.p2[0] * width, self.p2[1] * height))

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        yield self.p1
        yield self.p2

    def __getitem__(self, index: int) -> Point:
        if index == 0:
            return self.p1
        if index == 1:
            return self.p2
        raise IndexError("ValidatedLine contient exactement deux points (p1, p2)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "p1_normalized": [round(self.p1[0], 6), round(self.p1[1], 6)],
            "p2_normalized": [round(self.p2[0], 6), round(self.p2[1], 6)],
            "inside_side": self.inside_side,
            "length_normalized": round(self.length_ratio, 6),
            "frame_resolution": [self.frame_width, self.frame_height],
            "display_scale": round(self.display_scale, 6),
            "clicks_px": [[round(x, 2), round(y, 2)] for x, y in self.clicks_px],
            "confirmed_at_utc": self.confirmed_at,
        }


@dataclass
class LineSelection:
    """État de la sélection : deux points normalisés et une orientation.

    Cette classe ne connaît ni fenêtre ni souris : elle reçoit des clics déjà
    exprimés dans le repère de l'image affichée et les normalise. Toute décision
    y est explicite — ``confirm`` refuse une ligne invalide en levant
    :class:`geometry.LineError` avec la raison exacte, sans jamais corriger en
    silence ni retomber sur une ligne par défaut.
    """

    frame_width: int
    frame_height: int
    display_width: int
    display_height: int
    inside_side: str = "negative"
    min_length_ratio: float = 0.05
    message: str = ""
    message_is_error: bool = False

    #: Clics bruts dans la fenêtre d'affichage (audit et affichage des points).
    clicks_px: list[Point] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise CalibrationError(
                f"Résolution de source invalide : {self.frame_width}x{self.frame_height}"
            )
        if self.display_width <= 0 or self.display_height <= 0:
            raise CalibrationError(
                "Résolution d'affichage invalide : les clics ne peuvent pas être normalisés"
            )
        sign_of_side(self.inside_side)  # échoue tôt sur une valeur inconnue

    # -- État -------------------------------------------------------------
    @property
    def points(self) -> tuple[Point, ...]:
        """Points cliqués, en coordonnées **normalisées** (0-1)."""
        return tuple(
            (x / self.display_width, y / self.display_height) for x, y in self.clicks_px
        )

    @property
    def ready(self) -> bool:
        """Deux points sont-ils sélectionnés ?"""
        return len(self.clicks_px) == 2

    @property
    def pending_points(self) -> int:
        """Nombre de clics encore attendus."""
        return max(0, 2 - len(self.clicks_px))

    def to_display_pixels(self, width: int, height: int) -> tuple[Point, ...]:
        return tuple((x / self.display_width * width, y / self.display_height * height)
                     for x, y in self.clicks_px)

    # -- Entrées opérateur ------------------------------------------------
    def click(self, x_px: float, y_px: float) -> bool:
        """Enregistre un clic. Retourne ``False`` si la sélection est déjà complète."""
        if self.ready:
            self.set_message(
                "Deux points sont déjà sélectionnés : « G » pour recommencer, « C » pour valider.",
                error=True,
            )
            return False
        self.clicks_px.append((float(x_px), float(y_px)))
        if self.ready:
            self.set_message("Ligne tracée : « C » pour valider, « G » pour recommencer.", error=False)
        else:
            self.set_message("Extrémité 1 placée : cliquer la seconde extrémité.", error=False)
        return True

    def reset(self) -> None:
        """Efface les points et recommence la sélection (« G »)."""
        self.clicks_px.clear()
        self.set_message("Sélection réinitialisée : cliquer les deux extrémités.", error=False)

    def invert_inside_side(self) -> str:
        """Inverse intérieur/extérieur (« I ») et retourne le nouveau côté.

        Le sens du franchissement IN/OUT suit cette inversion : le côté marqué
        « INTERIEUR » à l'écran est exactement celui compté comme intérieur.
        """
        self.inside_side = "positive" if self.inside_side == "negative" else "negative"
        self.set_message(
            f"Sens inverse : le cote « {self.inside_side} » est desormais l'INTERIEUR.",
            error=False,
        )
        return self.inside_side

    def set_message(self, text: str, error: bool) -> None:
        """Publie un message explicite à l'écran (jamais de refus silencieux)."""
        self.message = text
        self.message_is_error = error

    # -- Validation -------------------------------------------------------
    def confirm(self) -> ValidatedLine:
        """Valide la sélection et retourne la ligne à utiliser pour le comptage.

        Raises:
            LineError: sélection incomplète, points identiques ou ligne trop
                courte (spec 5.7). La fenêtre reste ouverte dans ce cas : le
                refus est annoncé à l'opérateur, jamais silencieux.
        """
        if not self.ready:
            raise LineError(
                f"Sélection incomplète : {self.pending_points} point(s) encore attendu(s) "
                "avant de pouvoir valider."
            )
        p1, p2 = self.points
        validate_line(p1, p2, self.min_length_ratio)
        return ValidatedLine(
            p1=p1,
            p2=p2,
            inside_side=self.inside_side,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            display_scale=self.display_width / self.frame_width,
            clicks_px=(self.clicks_px[0], self.clicks_px[1]),
            confirmed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # -- Rendu ------------------------------------------------------------
    def draw(self, canvas: Any) -> Any:
        """Dessine points, ligne, sens IN/OUT et aide, puis retourne l'image."""
        height, width = canvas.shape[:2]
        points = self.to_display_pixels(width, height)

        # La ligne est tracée d'abord : les deux points restent visibles par-dessus
        # (l'opérateur doit toujours voir ce qu'il a cliqué).
        if self.ready:
            self._draw_line_and_sides(canvas, points)

        for index, point in enumerate(points):
            centre = (int(round(point[0])), int(round(point[1])))
            cv2.circle(canvas, centre, 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas, str(index + 1), (centre[0] + 12, centre[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

        self._draw_help(canvas, width, height)
        return canvas

    def _draw_line_and_sides(self, canvas: Any, points: Sequence[Point]) -> None:
        start = (int(round(points[0][0])), int(round(points[0][1])))
        end = (int(round(points[1][0])), int(round(points[1][1])))
        cv2.line(canvas, start, end, (0, 255, 0), 3, cv2.LINE_AA)
        draw_inside_side(canvas, start, end, self.inside_side)

    def _draw_help(self, canvas: Any, width: int, height: int) -> None:
        lines = [
            "Ligne de comptage : cliquer les DEUX extremites (point 1 puis point 2)",
            "C : valider la ligne   |   G : recommencer   |   I : inverser IN/OUT   |   Echap : annuler",
            "N : demarrer SANS ligne (aucune porte : effectif = warm-up + NEW, jamais IN/OUT)",
            (
                "Aucune ligne par defaut : le comptage ne demarre qu'apres validation."
                if not self.ready
                else "Ligne prete : appuyer sur C pour lancer YOLO11 + BoT-SORT + comptage."
            ),
        ]
        if self.message:
            lines.append(self.message)
        colour = (0, 0, 255) if self.message_is_error else (255, 255, 255)
        y = 30
        for index, text in enumerate(lines):
            safe_text = _clean_ascii(text)
            active = (0, 0, 255) if (index == len(lines) - 1 and self.message_is_error) else colour
            cv2.putText(
                canvas, safe_text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                canvas, safe_text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, active, 2, cv2.LINE_AA
            )
            y += 26


def _draw_label(canvas: Any, point: Point, text: str, colour: tuple[int, int, int]) -> None:
    safe_text = _clean_ascii(text)
    height, width = canvas.shape[:2]
    size, _ = cv2.getTextSize(safe_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = max(5, min(int(point[0]) - size[0] // 2, width - size[0] - 5))
    y = max(size[1] + 5, min(int(point[1]), height - 5))
    cv2.putText(
        canvas, safe_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        canvas, safe_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA
    )


def draw_inside_side(
    canvas: Any, start: tuple[int, int], end: tuple[int, int], inside_side: str
) -> None:
    """Flèche vers l'intérieur et libellés INTERIEUR / EXTERIEUR (pixels entiers)."""
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-9:
        return  # ligne dégénérée : la validation la refuse, aucun sens à dessiner
    # Normale unitaire : signe négatif de la distance signée dans ce sens,
    # d'où le facteur `-sign_of_side` pour viser l'intérieur (voir geometry).
    normal = (-dy / norm, dx / norm)
    inside_sign = sign_of_side(inside_side)
    mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    offset = 0.12 * min(canvas.shape[:2])

    inside_point = (
        int(round(mid[0] - inside_sign * normal[0] * offset)),
        int(round(mid[1] - inside_sign * normal[1] * offset)),
    )
    outside_point = (
        int(round(mid[0] + inside_sign * normal[0] * offset)),
        int(round(mid[1] + inside_sign * normal[1] * offset)),
    )
    _draw_label(canvas, inside_point, f"INTERIEUR (IN, cote {inside_side})", (0, 255, 0))
    _draw_label(canvas, outside_point, "EXTERIEUR (OUT)", (0, 165, 255))
    cv2.arrowedLine(
        canvas,
        (int(round(mid[0])), int(round(mid[1]))),
        inside_point,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )


def draw_dead_zone(
    frame: Any, start: tuple[float, float], end: tuple[float, float], dead_zone_px: float | None
) -> None:
    """Bande translucide de la zone morte, de demi-largeur ``dead_zone_px``."""
    if not dead_zone_px:
        return
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(float(np.hypot(dx, dy)), 1.0)
    nx, ny = dy / length, -dx / length
    points = np.array(
        [
            [start[0] + nx * dead_zone_px, start[1] + ny * dead_zone_px],
            [end[0] + nx * dead_zone_px, end[1] + ny * dead_zone_px],
            [end[0] - nx * dead_zone_px, end[1] - ny * dead_zone_px],
            [start[0] - nx * dead_zone_px, start[1] - ny * dead_zone_px],
        ],
        dtype=np.int32,
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], (0, 255, 255))
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def draw_line_overlay(frame: Any, line: Any, dead_zone_px: float | None) -> Any:
    """Ligne, zone morte et flèche intérieur/extérieur : dessin **unique**.

    Seul dessin de ces trois éléments, partagé par la fenêtre de traitement
    OpenCV (``OccupancyManager.draw_overlay``), la vue web et la relecture :
    les trois affichages sont donc identiques au pixel près. ``line`` expose
    ``to_pixels(width, height)`` et ``inside_side`` (``VirtualLine`` ou
    :class:`ValidatedLine`). Dessine en place et retourne ``frame``.
    """
    height, width = frame.shape[:2]
    start, end = line.to_pixels(width, height)
    cv2.line(
        frame,
        (int(start[0]), int(start[1])),
        (int(end[0]), int(end[1])),
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    draw_dead_zone(frame, start, end, dead_zone_px)
    draw_inside_side(
        frame,
        (int(round(start[0])), int(round(start[1]))),
        (int(round(end[0])), int(round(end[1]))),
        line.inside_side,
    )
    return frame


def mode_for_key(key: int) -> str | None:
    """Réponse à la question de démarrage : « O », « N », Échap, ou ``None`` (attente)."""
    if key in KEY_MODE_LINE:
        return MODE_LINE
    if key in KEY_NO_LINE:
        return MODE_NO_LINE
    if key == KEY_CANCEL:
        return MODE_CANCEL
    return None


def draw_mode_question(canvas: Any) -> Any:
    """Affiche la question de démarrage sur la première image, puis retourne l'image."""
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    top = max(0, height // 2 - 70)
    cv2.rectangle(overlay, (0, top), (width, min(height, top + 140)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    lines = (
        ("Mode de comptage ?", 0.7, (200, 200, 200)),
        (MODE_QUESTION, 0.9, (255, 255, 255)),
        ("Echap : quitter", 0.6, (200, 200, 200)),
    )
    y = top + 35
    for text, scale, colour in lines:
        safe_text = _clean_ascii(text)
        (text_width, _), _ = cv2.getTextSize(safe_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = max(10, (width - text_width) // 2)
        cv2.putText(canvas, safe_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
        y += 45
    return canvas


def read_first_frame(source: int | str) -> Any:
    """Lit la **première** image de la source (exigence 1 de la sélection).

    Raises:
        SourceUnreadable: source inouvrable ou première image illisible. Le cas
            est distingué d'une erreur d'affichage (spec 5.5).
    """
    capture = cv2.VideoCapture(source)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise SourceUnreadable(
            f"Impossible de lire la première image de la source : {source}"
        )
    return frame


def display_canvas(frame: Any, max_display_side: int = MAX_DISPLAY_SIDE) -> Any:
    """Première image à la taille de la fenêtre de sélection (côté le plus long borné)."""
    frame_height, frame_width = frame.shape[:2]
    scale = min(1.0, float(max_display_side) / max(frame_width, frame_height))
    if scale < 1.0:
        return cv2.resize(
            frame,
            (int(frame_width * scale), int(frame_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame.copy()


def select_line(
    source: int | str,
    *,
    inside_side: str = "negative",
    min_length_ratio: float = 0.05,
    window_name: str = DEFAULT_WINDOW_NAME,
    max_display_side: int = MAX_DISPLAY_SIDE,
) -> ValidatedLine | None:
    """Affiche la première image et exige une ligne validée par l'opérateur.

    Returns:
        La :class:`ValidatedLine` sélectionnée, seule ligne utilisée ensuite par
        le pipeline (dessin, distance signée, franchissements, comptage) ; ou
        ``None`` si l'opérateur choisit explicitement le mode sans ligne (« N »),
        à la question de démarrage ou pendant le tracé.

    Raises:
        SourceUnreadable: première image illisible.
        CalibrationUnavailable: aucune fenêtre ne peut être ouverte (le
            démarrage est refusé plutôt que de fabriquer une ligne par défaut).
        CalibrationCancelled: Échap ou fermeture de la fenêtre.
    """
    # Nom de fenêtre assaini **avant** tout appel OpenCV : sans cela, un titre
    # accentué ferait échouer l'installation du callback souris (voir
    # :func:`sanitize_window_name`).
    window_name = sanitize_window_name(window_name)

    frame = read_first_frame(source)
    frame_height, frame_width = frame.shape[:2]

    if not display_available():
        raise CalibrationUnavailable(
            "Aucun affichage graphique disponible : la ligne virtuelle doit être "
            "définie manuellement, le comptage ne peut donc pas démarrer."
        )

    canvas = display_canvas(frame, max_display_side)
    display_height, display_width = canvas.shape[:2]

    selection = LineSelection(
        frame_width=frame_width,
        frame_height=frame_height,
        display_width=display_width,
        display_height=display_height,
        inside_side=inside_side,
        min_length_ratio=min_length_ratio,
    )
    selection.set_message("Cliquer l'extremite 1 de la ligne.", error=False)
    #: Question de démarrage (« O » ligne / « N » sans ligne) posée avant tout
    #: clic ; la calibration elle-même est ensuite inchangée.
    choosing_mode = True

    def render() -> Any:
        if choosing_mode:
            return draw_mode_question(canvas.copy())
        return selection.draw(canvas.copy())

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and not choosing_mode:
            selection.click(x, y)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Le backend Qt (défaut de nombreuses distributions, et seul disponible
        # sur certaines installations) ne crée le widget natif de la fenêtre
        # qu'au premier rendu réel : installer le callback souris avant ce
        # premier rendu échoue en « NULL window handler », ce qui rendait toute
        # calibration impossible. On force donc un affichage avant le callback.
        cv2.imshow(window_name, render())
        cv2.waitKey(1)
        cv2.setMouseCallback(window_name, on_mouse)
    except cv2.error as error:  # plateforme graphique indisponible en pratique
        cv2.destroyAllWindows()
        raise CalibrationUnavailable(
            f"Ouverture de la fenêtre de sélection impossible : {error}"
        ) from error

    try:
        while True:
            cv2.imshow(window_name, render())
            key = cv2.waitKey(POLL_INTERVAL_MS) & 0xFF
            if choosing_mode:
                mode = mode_for_key(key)
                if mode == MODE_LINE:
                    choosing_mode = False
                elif mode == MODE_NO_LINE:
                    return None
                elif mode == MODE_CANCEL:
                    raise CalibrationCancelled(
                        "Choix du mode annulé par l'opérateur (Échap) : comptage non lancé."
                    )
            elif key in KEY_CONFIRM:
                try:
                    return selection.confirm()
                except LineError as error:
                    selection.set_message(f"Ligne refusee : {error}", error=True)
            elif key in KEY_RESET:
                selection.reset()
            elif key in KEY_INVERT:
                selection.invert_inside_side()
            elif key in KEY_NO_LINE:
                return None
            elif key == KEY_CANCEL:
                raise CalibrationCancelled(
                    "Sélection de la ligne annulée par l'opérateur (Échap)."
                )
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise CalibrationCancelled(
                    "Fenêtre de sélection fermée : ligne non validée, comptage non lancé."
                )
    finally:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)


#: Alias canonique conforme aux attentes du point d'entrée.
calibrate = select_line
