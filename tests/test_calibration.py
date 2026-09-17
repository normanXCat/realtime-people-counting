"""Niveau 1 — Sélection **manuelle obligatoire** de la ligne virtuelle.

Vérifie tout ce qui est testable sans fenêtre : normalisation des clics (la ligne
doit être identique quelle que soit la résolution), machine d'états de la
sélection (deux points, remise à zéro, inversion intérieur/extérieur), refus
explicite d'une ligne incomplète, dégénérée ou trop courte, et rendu (points,
ligne, côtés IN/OUT annoncés à l'écran).
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration import (
    CalibrationError,
    LineSelection,
    ValidatedLine,
    display_available,
)
from geometry import LineError

WIDTH, HEIGHT = 1920, 1080
DISPLAY = (960, 540)  # fenêtre réduite : facteur 0,5


def selection(**kwargs) -> LineSelection:
    params = {
        "frame_width": WIDTH,
        "frame_height": HEIGHT,
        "display_width": DISPLAY[0],
        "display_height": DISPLAY[1],
    }
    params.update(kwargs)
    return LineSelection(**params)


def canvas(width: int = 600, height: int = 600) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def clicked(item: LineSelection, p1, p2) -> LineSelection:
    item.click(*p1)
    item.click(*p2)
    return item


def render_selection(**kwargs) -> LineSelection:
    """Sélection dont la fenêtre a exactement la taille d'un canevas 600×600.

    Les clics s'y projettent 1:1, ce qui rend les vérifications de rendu exactes.
    """
    params = {"frame_width": 600, "frame_height": 600, "display_width": 600, "display_height": 600}
    params.update(kwargs)
    return LineSelection(**params)


def green_near(image: np.ndarray, row: int, column: int, radius: int = 3) -> bool:
    """Un tracé vert existe-t-il au voisinage de (row, column) ? (anti-aliasing toléré)"""
    window = image[
        max(0, row - radius): row + radius + 1,
        max(0, column - radius): column + radius + 1,
    ]
    return bool(
        ((window[:, :, 1] > 150) & (window[:, :, 0] < 120) & (window[:, :, 2] < 120)).any()
    )


def assert_points(actual, expected) -> None:
    """Comparaison approximative de points (pytest.approx refuse l'imbriqué)."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got == pytest.approx(want)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frame_width,display_width", [(0, 960), (1920, 0)])
def test_resolution_invalide_refusee(frame_width, display_width):
    with pytest.raises(CalibrationError):
        selection(frame_width=frame_width, display_width=display_width)


def test_orientation_inconnue_refusee():
    with pytest.raises(LineError):
        selection(inside_side="milieu")


def test_display_available_suit_la_presence_d_un_ecran(monkeypatch):
    import os
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert display_available() is False

    monkeypatch.setenv("DISPLAY", ":0")
    assert display_available() is True


# ---------------------------------------------------------------------------
# Enregistrement des clics
# ---------------------------------------------------------------------------
def test_deux_clics_completent_la_selection():
    item = selection()
    assert item.pending_points == 2
    assert item.click(480, 270) is True
    assert item.pending_points == 1
    assert item.ready is False
    assert item.click(960, 270) is True
    assert item.ready is True
    assert item.pending_points == 0


def test_les_points_sont_normalises_entre_0_et_1():
    item = clicked(selection(), (480, 270), (960, 405))
    assert_points(item.points, ((0.5, 0.5), (1.0, 0.75)))


@pytest.mark.parametrize(
    "frame,display,click",
    [
        ((1920, 1080), (960, 540), (240, 135)),   # demi-résolution
        ((1280, 720), (1280, 720), (320, 180)),   # affichage natif
        ((3840, 2160), (960, 540), (240, 135)),   # très haute résolution
    ],
)
def test_meme_clic_relatif_donne_la_meme_ligne(frame, display, click):
    """La ligne ne dépend pas de la résolution (exigence de normalisation)."""
    item = clicked(
        selection(
            frame_width=frame[0], frame_height=frame[1],
            display_width=display[0], display_height=display[1],
        ),
        click,
        (display[0], display[1]),
    )
    assert_points(item.points, ((0.25, 0.25), (1.0, 1.0)))


def test_un_troisieme_clic_est_refuse_et_annonce():
    item = clicked(selection(), (100, 100), (500, 100))
    assert item.click(300, 300) is False
    assert len(item.points) == 2
    assert item.message_is_error is True
    assert "G" in item.message  # la remise à zéro est indiquée à l'opérateur


def test_reinitialisation_efface_les_points():
    item = clicked(selection(), (100, 100), (500, 100))
    item.reset()
    assert item.points == ()
    assert item.ready is False
    assert item.message_is_error is False


def test_inversion_interieur_exterieur():
    item = selection(inside_side="negative")
    assert item.invert_inside_side() == "positive"
    assert item.invert_inside_side() == "negative"
    assert "negative" in item.message


# ---------------------------------------------------------------------------
# Validation (exigence 6 : points distincts et longueur valide)
# ---------------------------------------------------------------------------
def test_validation_incomplete_refusee():
    item = selection()
    item.click(100, 100)
    with pytest.raises(LineError) as error:
        item.confirm()
    assert "incomplète" in str(error.value)
    assert item.ready is False  # la sélection reste ouverte pour l'opérateur


def test_points_identiques_refuses():
    item = clicked(selection(), (500, 300), (500, 300))
    with pytest.raises(LineError) as error:
        item.confirm()
    assert "identiques" in str(error.value)


def test_ligne_trop_courte_refusee():
    item = clicked(selection(), (300, 300), (310, 300))
    with pytest.raises(LineError) as error:
        item.confirm()
    assert "longueur" in str(error.value)


def test_seuil_de_longueur_issu_de_la_configuration():
    """Une ligne refusée avec le seuil livré peut passer avec un seuil plus permissif."""
    item = clicked(selection(), (300, 300), (330, 300))
    with pytest.raises(LineError):
        item.confirm()  # 30/960 = 0,031 < 0,05
    permissive = clicked(selection(min_length_ratio=0.01), (300, 300), (330, 300))
    assert permissive.confirm().length_ratio == pytest.approx(30 / 960)


def test_confirmation_retourne_la_ligne_validee():
    item = clicked(selection(inside_side="positive"), (480, 540), (960, 540))
    validated = item.confirm()

    assert isinstance(validated, ValidatedLine)
    assert validated.p1 == pytest.approx((0.5, 1.0))
    assert validated.p2 == pytest.approx((1.0, 1.0))
    assert validated.inside_side == "positive"
    assert (validated.frame_width, validated.frame_height) == (WIDTH, HEIGHT)
    assert validated.display_scale == pytest.approx(0.5)
    assert validated.clicks_px == ((480, 540), (960, 540))
    assert validated.confirmed_at  # horodatage pour l'audit
    assert_points(validated.to_pixels(WIDTH, HEIGHT), ((960.0, 1080.0), (1920.0, 1080.0)))


@pytest.mark.parametrize(
    "p1,p2,orientation",
    [
        ((60, 300), (540, 300), "horizontale"),
        ((300, 60), (300, 540), "verticale"),
        ((60, 500), (540, 100), "inclinee"),
        ((540, 500), (60, 100), "inclinee inversee"),
    ],
)
def test_toute_orientation_de_ligne_est_acceptee(p1, p2, orientation):
    """La ligne peut être horizontale, verticale ou inclinée (aucune contrainte d'axe)."""
    item = clicked(render_selection(), p1, p2)

    validated = item.confirm()

    assert validated.p1 == pytest.approx((p1[0] / 600, p1[1] / 600))
    assert validated.p2 == pytest.approx((p2[0] / 600, p2[1] / 600))
    assert validated.length_ratio > 0.5
    # Les deux extrémités sont bien distinctes, quelle que soit l'orientation.
    assert validated.p1 != validated.p2
    image = item.draw(canvas())
    # Le milieu des deux extrémités est sur la ligne : le tracé y est présent
    # quelle que soit l'orientation (tolérance d'anti-aliasing).
    assert green_near(image, 300, 300)


def test_deux_points_identiques_refuses_meme_a_un_pixel_pres():
    """Une ligne de longueur nulle ou quasi nulle est refusée."""
    identical = clicked(render_selection(), (300, 300), (300, 300))
    with pytest.raises(LineError):
        identical.confirm()

    one_pixel = clicked(render_selection(), (300, 300), (301, 300))
    with pytest.raises(LineError) as error:
        one_pixel.confirm()
    assert "longueur" in str(error.value)


def test_dictionnaire_de_tracabilite_complet():
    validated = clicked(selection(), (0, 270), (960, 270)).confirm()
    payload = validated.to_dict()
    assert payload["p1_normalized"] == [0.0, 0.5]
    assert payload["p2_normalized"] == [1.0, 0.5]
    assert payload["inside_side"] == "negative"
    assert payload["length_normalized"] == pytest.approx(1.0)
    assert payload["frame_resolution"] == [WIDTH, HEIGHT]
    assert len(payload["clicks_px"]) == 2


# ---------------------------------------------------------------------------
# Rendu : points, ligne, annonce des côtés IN/OUT (exigence 9)
# ---------------------------------------------------------------------------
def test_rendu_des_points_seuls_ne_plante_pas():
    item = render_selection()
    item.click(100, 100)
    image = item.draw(canvas())
    assert image.shape == (600, 600, 3)
    assert int(image.sum()) > 0  # le point est visible


def test_rendu_de_la_ligne_et_des_deux_points():
    item = clicked(render_selection(), (300, 60), (300, 540))
    image = item.draw(canvas())
    assert image[300, 300][1] > 200  # ligne verte au milieu
    assert image[60, 300][2] > 200   # point 1 (rouge) à l'endroit cliqué
    assert image[540, 300][2] > 200  # point 2 (rouge) à l'endroit cliqué


def test_rendu_annonce_le_cote_interieur():
    """Le côté intérieur est marqué, et l'inversion le déplace (exigence 9)."""
    item = clicked(render_selection(inside_side="negative"), (10, 300), (590, 300))
    image = item.draw(canvas())
    outside_colour = np.array([0, 165, 255])
    above = np.all(image[:300] == outside_colour, axis=-1).sum()
    below = np.all(image[300:] == outside_colour, axis=-1).sum()
    # Intérieur = sous la ligne : l'étiquette "EXTERIEUR" est au-dessus.
    assert above > 0
    assert below == 0

    item.invert_inside_side()
    inverted = item.draw(canvas())
    above_inverted = np.all(inverted[:300] == outside_colour, axis=-1).sum()
    below_inverted = np.all(inverted[300:] == outside_colour, axis=-1).sum()
    assert below_inverted > 0
    assert above_inverted == 0


def test_rendu_affiche_l_aide_et_l_etat_de_la_selection():
    item = render_selection()
    item.click(100, 100)
    image = item.draw(canvas())
    # L'aide (C / G / I / Echap) et l'état sont écrits en clair dans l'image :
    # on vérifie qu'une zone de texte a bien été rendue en haut de la fenêtre.
    assert int(image[:120].sum()) > 0


def test_rendu_ligne_degeneree_ne_plante_pas():
    item = clicked(render_selection(), (300, 300), (300, 300))
    image = item.draw(canvas())
    assert image.shape == (600, 600, 3)


def test_clean_ascii_nettoie_les_accents():
    from calibration import _clean_ascii

    cleaned = _clean_ascii("Sélection incomplète : « C » validé, « G » réinitialisé — été")
    assert "é" not in cleaned
    assert "«" not in cleaned
    assert "Selection incomplete : \" C \" valide, \" G \" reinitialise - ete" == cleaned


def test_sanitize_window_name_produit_un_nom_ascii():
    """Un nom de fenêtre non ASCII empêchait `setMouseCallback` de s'installer."""
    from calibration import DEFAULT_WINDOW_NAME, sanitize_window_name

    name = sanitize_window_name("People counting — ligne de comptage")
    assert name.isascii()
    assert name == "People counting - ligne de comptage"
    # Les accents sont translittérés, pas supprimés : le titre reste lisible.
    assert sanitize_window_name("Sélection de la ligne") == "Selection de la ligne"
    # Un caractère non translittérable ne peut pas casser le backend.
    assert sanitize_window_name("ligne 中文").isascii()
    assert sanitize_window_name("Ligne n°1") == "Ligne n?1"


def test_sanitize_window_name_refuse_le_vide_sans_planter():
    from calibration import DEFAULT_WINDOW_NAME, sanitize_window_name

    assert sanitize_window_name("") == DEFAULT_WINDOW_NAME
    assert sanitize_window_name("   ") == DEFAULT_WINDOW_NAME
    assert sanitize_window_name("\n\t \r") == DEFAULT_WINDOW_NAME


def test_sanitize_window_name_refuse_un_type_invalide():
    from calibration import CalibrationError, sanitize_window_name

    with pytest.raises(CalibrationError):
        sanitize_window_name(42)  # type: ignore[arg-type]
