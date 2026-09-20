"""Drapeau `--line` : ligne virtuelle explicite (lot 0, `docs/lot_0_outillage.md` §0.1).

Trois exigences, chacune vérifiée ici :

1. **Analyse syntaxique** — forme valide, forme malformée, valeurs hors bornes,
   ligne dégénérée, ligne trop courte. Un argument douteux est refusé, jamais
   réinterprété.
2. **Refus en mode non interactif** — `--no-show` sans `--line` échoue
   immédiatement, avec un code de sortie qui lui est propre. Le repli
   automatique « ligne horizontale à 60 % » reste supprimé et ne revient sous
   aucune condition.
3. **Équivalence** — une ligne passée en argument produit exactement la même
   `ValidatedLine` que la même ligne obtenue par des clics, et elle traverse la
   même validation. Il n'existe pas de second chemin de contrôle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration import CalibrationUnavailable, LineSelection  # noqa: E402
from config import load_config  # noqa: E402
from geometry import LineError  # noqa: E402
import main as entry_point  # noqa: E402
from main import (  # noqa: E402
    NonInteractiveLineRequired,
    build_argument_parser,
    line_from_argument,
    parse_line_argument,
    perform_line_selection,
)


@pytest.fixture()
def config():
    return load_config(None)


@pytest.fixture()
def fake_frame(monkeypatch):
    """Neutralise la lecture disque : seule la résolution nous intéresse ici."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(entry_point, "read_first_frame", lambda source: frame)
    return frame


# ---------------------------------------------------------------------------
# 1. Analyse syntaxique
# ---------------------------------------------------------------------------
def test_forme_valide():
    p1, p2 = parse_line_argument("0.05,0.6,0.95,0.6")
    assert p1 == (0.05, 0.6)
    assert p2 == (0.95, 0.6)


def test_espaces_autour_des_valeurs_tolerés():
    """Un copier-coller avec espaces reste lisible sans ambiguïté."""
    assert parse_line_argument(" 0.1 , 0.2 , 0.8 , 0.9 ") == ((0.1, 0.2), (0.8, 0.9))


@pytest.mark.parametrize(
    "raw",
    [
        "0.05,0.6,0.95",          # trois valeurs
        "0.05,0.6,0.95,0.6,0.7",  # cinq valeurs
        "",                       # vide
        "a,b,c,d",                # non numérique
        "0.05;0.6;0.95;0.6",      # mauvais séparateur
    ],
)
def test_forme_malformee_refusee(raw):
    with pytest.raises(LineError):
        parse_line_argument(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "0.05,0.6,1.5,0.6",    # hors bornes, au-dessus
        "-0.1,0.6,0.95,0.6",   # hors bornes, en dessous
        "0.5,0.5,0.5,0.5",     # dégénérée : deux points identiques
        "0.5,0.5,0.51,0.5",    # plus courte que min_length_ratio
    ],
)
def test_valeurs_refusees_par_la_validation_partagee(raw, config, fake_frame):
    """Bornes, dégénérescence et longueur : contrôlées par `validate_line`.

    Le refus ne vient pas d'un contrôle propre à `--line` mais de la fonction
    qui valide **aussi** la ligne cliquée : un seul endroit décide de ce qu'est
    une ligne acceptable.
    """
    with pytest.raises(LineError):
        line_from_argument(config, "clip.mp4", raw)


def test_la_source_est_lue_avant_de_construire_la_ligne(config, monkeypatch):
    """Une source illisible est signalée ici, pas plus tard dans la boucle."""
    from calibration import SourceUnreadable

    def _boom(source):
        raise SourceUnreadable(f"illisible : {source}")

    monkeypatch.setattr(entry_point, "read_first_frame", _boom)
    with pytest.raises(SourceUnreadable):
        line_from_argument(config, "absente.mp4", "0.05,0.6,0.95,0.6")


# ---------------------------------------------------------------------------
# 2. Mode non interactif
# ---------------------------------------------------------------------------
def test_sans_ligne_le_mode_non_interactif_refuse(config):
    with pytest.raises(NonInteractiveLineRequired) as error:
        perform_line_selection(config, "clip.mp4", headless_requested=True)
    assert "--line" in str(error.value)


def test_le_refus_reste_une_calibration_indisponible(config):
    """La nouvelle exception s'insère dans la hiérarchie existante.

    Tout code qui rattrapait `CalibrationUnavailable` continue de fonctionner :
    le lot n'introduit pas un chemin d'erreur parallèle.
    """
    with pytest.raises(CalibrationUnavailable):
        perform_line_selection(config, "clip.mp4", headless_requested=True)


def test_avec_ligne_la_selection_manuelle_n_est_pas_sollicitee(
    config, fake_frame, monkeypatch
):
    appels: list = []
    monkeypatch.setattr(
        entry_point, "select_line", lambda source, **kw: appels.append(source)
    )

    validated = perform_line_selection(
        config, "clip.mp4", headless_requested=True, line_argument="0.05,0.6,0.95,0.6"
    )

    assert appels == []
    assert validated.p1 == (0.05, 0.6)


def test_sans_ligne_ni_no_show_le_mode_interactif_est_inchange(config, monkeypatch):
    """Exigence §0.4 : le comportement historique ne bouge pas d'un iota."""
    captured: dict = {}

    def _fake(source, **kwargs):
        captured.update(kwargs, source=source)
        return "ligne-cliquee"

    monkeypatch.setattr(entry_point, "select_line", _fake)

    assert perform_line_selection(config, "clip.mp4") == "ligne-cliquee"
    assert captured["source"] == "clip.mp4"
    assert captured["inside_side"] == config.line.inside_side
    assert captured["min_length_ratio"] == config.line.min_length_ratio


# ---------------------------------------------------------------------------
# 3. Équivalence entre ligne cliquée et ligne passée en argument
# ---------------------------------------------------------------------------
def test_identite_entre_ligne_cliquee_et_ligne_en_argument(config, fake_frame):
    """Les deux provenances doivent décrire exactement la même géométrie.

    C'est la garantie qui rend le harnais de mesure légitime : mesurer avec
    `--line` revient à mesurer ce qu'un opérateur aurait obtenu en cliquant.
    """
    selection = LineSelection(
        frame_width=1280,
        frame_height=720,
        display_width=1280,
        display_height=720,
        inside_side=config.line.inside_side,
        min_length_ratio=config.line.min_length_ratio,
    )
    selection.click(0.05 * 1280, 0.6 * 720)
    selection.click(0.95 * 1280, 0.6 * 720)
    cliquee = selection.confirm()

    argument = line_from_argument(config, "clip.mp4", "0.05,0.6,0.95,0.6")

    assert argument.p1 == pytest.approx(cliquee.p1)
    assert argument.p2 == pytest.approx(cliquee.p2)
    assert argument.inside_side == cliquee.inside_side
    assert argument.frame_width == cliquee.frame_width
    assert argument.frame_height == cliquee.frame_height
    assert argument.length_ratio == pytest.approx(cliquee.length_ratio)


def test_l_orientation_de_la_configuration_est_respectee(config, fake_frame):
    """`--inside-side` reste maître de l'orientation, `--line` ne la fixe pas."""
    from config import override_config

    positive = override_config(config, {"line": {"inside_side": "positive"}})
    assert (
        line_from_argument(positive, "clip.mp4", "0.05,0.6,0.95,0.6").inside_side
        == "positive"
    )


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------
def test_le_drapeau_existe_et_n_a_aucune_valeur_par_defaut():
    """Une ligne par défaut serait une erreur silencieuse : il n'y en a pas."""
    args = build_argument_parser().parse_args(["--source", "clip.mp4"])
    assert args.line is None
