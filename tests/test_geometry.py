"""Niveau 1 — Géométrie : ligne horizontale, inclinée, inversée, point sur la
ligne, zone morte **relative**, garde-fou bord d'image, ligne dégénérée.

Couvre les règles 0.3 (aucun pixel absolu) et 0.4 (jamais de décision
silencieuse) ainsi que la spec 5.4 et 5.7.
"""

from __future__ import annotations

import pytest

from geometry import (
    AnchorReliability,
    LocalScale,
    LineError,
    Side,
    VirtualLine,
    Zone,
    anchor_reliability,
    bbox_height,
    compute_anchor,
    crossing_direction,
    edge_margin_px,
    line_length_normalized,
    median_height_of,
    segments_intersect,
    side_of,
    signed_perpendicular_distance,
    validate_line,
    zone_of,
)

LINE_H = ((0.0, 0.5), (1.0, 0.5))          # horizontale médiane
LINE_DIAG = ((0.0, 0.0), (1.0, 1.0))        # diagonale descendante
W, H = 800, 600


# ---------------------------------------------------------------------------
# Distance signée
# ---------------------------------------------------------------------------
def test_horizontal_line_signs():
    start, end = (0.0, 300.0), (800.0, 300.0)
    # inside_side = negative -> l'intérieur est sous la ligne (y plus grand)
    assert signed_perpendicular_distance((400.0, 400.0), start, end) == pytest.approx(-100.0)
    assert signed_perpendicular_distance((400.0, 200.0), start, end) == pytest.approx(100.0)
    assert signed_perpendicular_distance((400.0, 300.0), start, end) == pytest.approx(0.0)
    # La norme est normalisée : déplacer les extrémités ne change pas la distance.
    assert signed_perpendicular_distance((400.0, 400.0), (100.0, 300.0), (700.0, 300.0)) == pytest.approx(-100.0)


def test_inverted_points_flip_the_sign():
    start, end = (0.0, 300.0), (800.0, 300.0)
    point = (400.0, 400.0)
    forward = signed_perpendicular_distance(point, start, end)
    backward = signed_perpendicular_distance(point, end, start)
    assert forward == pytest.approx(-backward)
    assert side_of(forward, "negative") is Side.INTERIEURE
    assert side_of(backward, "negative") is Side.EXTERIEURE


def test_oblique_line_distance_is_orthogonal():
    start, end = (0.0, 0.0), (100.0, 100.0)
    # Distance orthogonale de (100, 0) à la diagonale y = x : 100 / sqrt(2).
    assert abs(signed_perpendicular_distance((100.0, 0.0), start, end)) == pytest.approx(
        100.0 / (2 ** 0.5)
    )


def test_point_exactly_on_line_is_indeterminate():
    start, end = (0.0, 300.0), (800.0, 300.0)
    distance = signed_perpendicular_distance((400.0, 300.0), start, end)
    assert side_of(distance, "negative") is Side.INDETERMINEE
    # Politique explicite : le côté peut être forcé, jamais deviné.
    assert side_of(distance, "negative", "inside") is Side.INTERIEURE
    assert side_of(distance, "negative", "outside") is Side.EXTERIEURE


def test_degenerate_line_distance_is_zero():
    assert signed_perpendicular_distance((10.0, 10.0), (5.0, 5.0), (5.0, 5.0)) == 0.0


# ---------------------------------------------------------------------------
# Zone morte relative
# ---------------------------------------------------------------------------
def test_zone_depends_on_relative_dead_zone_only():
    assert zone_of(-100.0, 30.0, "negative") is Zone.INTERIEURE
    assert zone_of(-10.0, 30.0, "negative") is Zone.MORTE
    assert zone_of(100.0, 30.0, "negative") is Zone.EXTERIEURE
    assert zone_of(0.0, 30.0, "negative") is Zone.MORTE
    # Sans échelle locale, aucune zone n'est décidable (règle 0.3).
    assert zone_of(-100.0, None, "negative") is None


def test_dead_zone_scales_with_local_scale():
    scale = LocalScale(window=10)
    scale.observe(200.0)
    dead_zone_small = scale.ratio_to_px(0.15)
    scale.observe(400.0)
    dead_zone_large = scale.ratio_to_px(0.15)
    assert dead_zone_small == pytest.approx(30.0)
    # La médiane de [200, 400] vaut 300 : la zone morte suit l'échelle observée
    # et n'est jamais une constante en pixels.
    assert dead_zone_large == pytest.approx(45.0)


def test_local_scale_median_and_guard():
    scale = LocalScale(window=5, min_samples=1)
    assert scale.median_height is None
    assert scale.ratio_to_px(0.1) is None
    for height in (100.0, 200.0, 300.0):
        scale.observe(height)
    assert scale.median_height == pytest.approx(200.0)
    scale.observe(0.0)  # hauteur nulle ignorée
    assert scale.samples == 3
    scale.reset()
    assert scale.median_height is None


def test_local_scale_respects_window():
    scale = LocalScale(window=3)
    for height in (100.0, 100.0, 100.0, 900.0):
        scale.observe(height)
    # La fenêtre glissante a oublié les deux premières mesures.
    assert scale.samples == 3
    assert scale.median_height == pytest.approx(100.0)


def test_median_height_of_boxes():
    assert median_height_of([]) is None
    assert median_height_of([[0, 0, 10, 20]]) == pytest.approx(20.0)
    # Hauteurs 10 et 30 -> médiane 20
    assert median_height_of([[0, 0, 10, 10], [0, 0, 10, 30]]) == pytest.approx(20.0)
    assert median_height_of([[0, 0, 10, 10], [0, 0, 10, 30], [0, 0, 10, 50]]) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Ancre et garde-fou bord d'image
# ---------------------------------------------------------------------------
def test_compute_anchor_is_bottom_center():
    assert compute_anchor([10.0, 20.0, 30.0, 80.0]) == (20.0, 80.0)
    assert bbox_height([10.0, 20.0, 30.0, 80.0]) == 60.0


def test_edge_margin_is_relative_to_frame():
    assert edge_margin_px(1000, 500, 0.02) == pytest.approx(10.0)
    assert edge_margin_px(100, 200, 0.05) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "box,expected",
    [
        ([100.0, 100.0, 200.0, 300.0], AnchorReliability.FIABLE),
        ([2.0, 100.0, 200.0, 300.0], AnchorReliability.BORD_IMAGE),      # bord gauche
        ([100.0, 1.0, 200.0, 300.0], AnchorReliability.BORD_IMAGE),      # bord haut
        ([600.0, 100.0, 799.0, 300.0], AnchorReliability.BORD_IMAGE),    # bord droit
        ([100.0, 100.0, 200.0, 599.0], AnchorReliability.BORD_IMAGE),    # bord bas
    ],
)
def test_anchor_reliability_at_frame_borders(box, expected):
    assert anchor_reliability(box, W, H, 0.02) is expected


# ---------------------------------------------------------------------------
# Franchissement
# ---------------------------------------------------------------------------
def test_segments_intersect():
    assert segments_intersect((5.0, 0.0), (5.0, 10.0), (0.0, 5.0), (10.0, 5.0))
    assert not segments_intersect((0.0, 0.0), (2.0, 2.0), (5.0, 0.0), (5.0, 5.0))


def test_crossing_direction_inside_and_outside():
    start, end = (0.0, 300.0), (800.0, 300.0)
    inside = crossing_direction((400.0, 200.0), (400.0, 400.0), start, end, "negative")
    outside = crossing_direction((400.0, 400.0), (400.0, 200.0), start, end, "negative")
    assert inside == "in"
    assert outside == "out"


def test_no_crossing_when_segment_does_not_touch_line():
    start, end = (0.0, 300.0), (800.0, 300.0)
    assert crossing_direction((400.0, 100.0), (400.0, 200.0), start, end, "negative") is None
    assert crossing_direction((400.0, 400.0), (400.0, 500.0), start, end, "negative") is None


def test_no_crossing_outside_line_extent():
    start, end = (100.0, 300.0), (200.0, 300.0)
    # La trajectoire franchit l'axe de la ligne mais hors de son étendue.
    assert crossing_direction((400.0, 200.0), (400.0, 400.0), start, end, "negative") is None


def test_landing_exactly_on_line_is_indeterminate():
    start, end = (0.0, 300.0), (800.0, 300.0)
    assert crossing_direction((400.0, 200.0), (400.0, 300.0), start, end, "negative") is None


def test_crossing_direction_fast_jump():
    start, end = (0.0, 300.0), (800.0, 300.0)
    # Saut complet d'un côté à l'autre entre deux frames.
    assert crossing_direction((400.0, 100.0), (400.0, 500.0), start, end, "negative") == "in"
    assert crossing_direction((400.0, 500.0), (400.0, 100.0), start, end, "negative") == "out"


# ---------------------------------------------------------------------------
# Ligne virtuelle calibrée
# ---------------------------------------------------------------------------
def test_line_length_normalized():
    assert line_length_normalized((0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert line_length_normalized((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0)


def test_validate_line_rejects_degenerate_and_short():
    validate_line((0.0, 0.5), (1.0, 0.5), 0.05)
    with pytest.raises(LineError):
        validate_line((0.5, 0.5), (0.5, 0.5), 0.05)
    with pytest.raises(LineError):
        validate_line((0.5, 0.5), (0.52, 0.5), 0.05)
    with pytest.raises(LineError):
        validate_line((0.0, 1.2), (1.0, 0.5), 0.05)


def test_virtual_line_rejects_unknown_inside_side():
    with pytest.raises(LineError):
        VirtualLine(p1=(0.0, 0.5), p2=(1.0, 0.5), inside_side="milieu")


def test_virtual_line_measurements():
    line = VirtualLine(p1=LINE_H[0], p2=LINE_H[1], inside_side="negative")
    assert line.distance((400.0, 400.0), W, H) == pytest.approx(-100.0)
    assert line.side((400.0, 400.0), W, H) is Side.INTERIEURE
    assert line.zone((400.0, 400.0), W, H, 30.0) is Zone.INTERIEURE
    assert line.zone((400.0, 400.0), W, H, None) is None
    assert line.crossing((400.0, 200.0), (400.0, 400.0), W, H) == "in"
    assert line.length_normalized == pytest.approx(1.0)


def test_virtual_line_with_frame_size_keeps_normalized_points():
    line = VirtualLine(p1=LINE_DIAG[0], p2=LINE_DIAG[1], inside_side="positive")
    sized = line.with_frame_size(W, H)
    assert sized.p1 == line.p1 and sized.p2 == line.p2
    assert sized.points_px() == ((0.0, 0.0), (800.0, 600.0))


def test_oblique_line_zones():
    line = VirtualLine(p1=(0.0, 0.0), p2=(1.0, 1.0), inside_side="negative")
    # Côté négatif de la diagonale : sous la ligne, soit (400, 540) en pixels.
    assert line.side((400.0, 540.0), W, H) is Side.INTERIEURE
    assert line.side((400.0, 100.0), W, H) is Side.EXTERIEURE
    # À 10 px de la diagonale (distance -192 -> point proche de la ligne en y).
    assert line.zone((400.0, 305.0), W, H, 40.0) is Zone.MORTE
