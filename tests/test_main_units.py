"""Tests du point d'entrée `src/main.py` (spec 5.2, 5.5, 5.6, 0.5).

Ils couvrent les parties testables sans caméra : résolution de la source,
surcharges CLI (chacune doit avoir un effet **réel** sur la configuration),
vérification de l'empreinte des poids, l'impossibilité de contourner la
sélection manuelle de la ligne, et le chemin d'erreur d'une source inouvrable —
trois cas distincts exigés par la spec 5.5. Les parcours complets (ligne validée,
annulation, comptage) sont dans `test_main_e2e.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration import CalibrationUnavailable  # noqa: E402
from config import ConfigError, load_config  # noqa: E402
from events import read_events  # noqa: E402
import main as entry_point  # noqa: E402
from main import (  # noqa: E402
    apply_cli_overrides,
    build_argument_parser,
    calibrate,
    dependency_versions,
    perform_line_selection,
    resolve_source,
    sha256_of,
    verify_weights,
)


# ---------------------------------------------------------------------------
# Source (spec 5.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [("0", 0), (" 1 ", 1), ("video.mp4", "video.mp4")])
def test_resolution_de_la_source(raw, expected):
    assert resolve_source(raw) == expected


def test_index_camera_et_writer_utilisent_la_meme_resolution():
    # Une chaîne avec espace ou retour à la ligne ne doit jamais produire deux
    # résolutions différentes entre l'ouverture YOLO et l'écriture vidéo (5.2).
    assert resolve_source("0\n") == 0
    assert resolve_source("0 ") == 0


# ---------------------------------------------------------------------------
# Poids (règle 0.5)
# ---------------------------------------------------------------------------
def test_sha256_et_verification_des_poids(tmp_path: Path):
    weights = tmp_path / "yolo_test.pt"
    weights.write_bytes(b"poids-factices")
    digest = sha256_of(weights)
    assert len(digest) == 64
    assert verify_weights(weights, digest) == digest


def test_poids_absents_ou_hash_incorrect_refuses(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        verify_weights(tmp_path / "absent.pt", "")

    weights = tmp_path / "yolo_test.pt"
    weights.write_bytes(b"poids-factices")
    with pytest.raises(ConfigError, match="Hash"):
        verify_weights(weights, "0" * 64)


def test_environnement_trace_les_dependances():
    versions = dependency_versions()
    assert versions["python"]
    assert "ultralytics" in versions


# ---------------------------------------------------------------------------
# Surcharges CLI : chaque drapeau doit avoir un effet réel (spec 0.1)
# ---------------------------------------------------------------------------
def _overrides(*argv: str):
    args = build_argument_parser().parse_args(list(argv))
    return apply_cli_overrides(load_config(), args)


def test_aucun_argument_cli_mort():
    """Les arguments sans effet de l'audit (§2.1 c) ne doivent plus exister."""
    parser = build_argument_parser()
    accepted = {action.dest for action in parser._actions}
    for dead in ("line2_p1", "line2_p2", "gate_width", "zenithal"):
        assert dead not in accepted
    # La ligne n'est plus fournissable en argument : elle est obligatoirement
    # sélectionnée à la souris (deux clics + « C »).
    for removed in ("line_p1", "line_p2", "calibrate"):
        assert removed not in accepted


def test_aucune_trace_de_double_ligne_ni_de_calibration_a_quatre_points():
    """Régression : une seule ligne, deux points, aucun résidu L1/L2."""
    forbidden = (
        "calibrate_two_lines",   # double calibration (L1 + L2)
        "line2",                 # second jeu de points
        "gate_width",            # largeur de porte entre les deux lignes
        "line_pos_ratio",        # ancienne ligne horizontale par pourcentage
        "EN_APPROCHE_L1",
        "EN_APPROCHE_L2",
    )
    for path in sorted((REPO_ROOT / "src").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} contient encore {token!r}"


def test_surcharges_de_detection():
    config = _overrides("--conf", "0.42", "--iou", "0.6", "--imgsz", "640")
    assert config.model.confidence == pytest.approx(0.42)
    assert config.model.nms_iou == pytest.approx(0.6)
    assert config.model.image_size == 640


def test_surcharge_d_orientation_de_ligne_et_de_source():
    """`--inside-side` ne fixe que l'orientation INITIALE proposée dans la fenêtre."""
    config = _overrides("--inside-side", "positive", "--source", "clip.mp4")
    assert config.line.inside_side == "positive"
    assert config.source.default == "clip.mp4"


def test_mode_headless_desactive_l_affichage():
    config = _overrides("--no-show")
    assert config.display.enabled is False


def test_mode_comptage_est_le_seul_mode_officiel():
    """Étape « l'utilisateur choisit le mode comptage » : explicite et validée."""
    parser = build_argument_parser()
    assert parser.parse_args([]).mode == "comptage"
    assert parser.parse_args(["--mode", "comptage"]).mode == "comptage"

    for unknown in ("calibration", "tracking", "detection", "comptage_double"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", unknown])


def test_drapeaux_d_ablation_du_protocole():
    baseline = _overrides("--no-long-term-reid")
    assert baseline.reid.long_term.enabled is False

    sans_marge = _overrides("--no-safety-margin")
    assert sans_marge.reid.long_term.safety_margin == 0.0

    sans_contrainte = _overrides("--no-spatial-constraint")
    assert sans_contrainte.reid.long_term.spatial_margin_ratio > 1000

    externe = _overrides("--external-reid")
    assert externe.reid.external_reid.enabled is True

    stabilise = _overrides("--use-bbox-locker", "--use-anchor-stabilizer")
    assert stabilise.stabilization.use_bbox_locker is True
    assert stabilise.stabilization.use_anchor_stabilizer is True


@pytest.mark.parametrize("obsolete", ["--line-p1", "--line-p2", "--calibrate"])
def test_arguments_de_ligne_automatique_refuses(obsolete):
    """Régression : aucun argument ne peut contourner la sélection manuelle."""
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([obsolete])


def test_orientation_de_ligne_invalide_refusee():
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--inside-side", "gauche"])


def test_sortie_jamais_dans_le_repertoire_courant_par_defaut():
    config = load_config()
    assert config.output.session_root not in (Path(""), Path("."))
    assert str(config.output.session_root) == "results"


# ---------------------------------------------------------------------------
# Ligne virtuelle : sélection manuelle obligatoire
# ---------------------------------------------------------------------------
def test_la_fenetre_de_selection_recoit_ses_parametres_de_la_configuration(monkeypatch):
    """La sélection reçoit la proposition d'orientation et le seuil de la config."""
    captured: dict = {}

    def _fake(source, **kwargs):
        captured.update(kwargs, source=source)
        return "ligne-validee"

    monkeypatch.setattr(entry_point, "select_line", _fake)
    config = _overrides("--inside-side", "positive")

    assert perform_line_selection(config, "clip.mp4") == "ligne-validee"
    assert captured["source"] == "clip.mp4"
    assert captured["inside_side"] == "positive"
    assert captured["min_length_ratio"] == config.line.min_length_ratio
    assert "ligne" in captured["window_name"]


def test_no_show_sans_ligne_refuse_meme_avec_un_affichage(monkeypatch):
    """`--no-show` sans `--line` refuse, même si un écran est disponible.

    Attente **modifiée par le lot 0**. Auparavant, `--no-show` ouvrait tout de
    même la fenêtre de sélection : la prévisualisation était supprimée, pas la
    validation de la ligne. Le lot 0 (`docs/lot_0_outillage.md` §0.1) fait de
    `--no-show` un mode non interactif réel, où `--line` est obligatoire.

    Ce que le test continue de garantir est inchangé et reste l'essentiel :
    **aucune ligne n'est jamais devinée**. Seule la façon de le refuser change.
    """
    appels: list = []

    def _fake(source, **kwargs):
        appels.append(source)
        return "ligne-validee"

    monkeypatch.setattr(entry_point, "display_available", lambda: True)
    monkeypatch.setattr(entry_point, "select_line", _fake)
    config = _overrides("--no-show")

    with pytest.raises(entry_point.NonInteractiveLineRequired):
        perform_line_selection(config, "clip.mp4", headless_requested=True)
    assert appels == [], "la fenêtre de sélection ne doit pas être sollicitée"


def test_no_show_sans_interface_graphique_annonce_l_erreur(monkeypatch):
    """Sans écran, `--no-show` doit expliquer qu'aucune ligne n'est définissable.

    Attente **ajustée par le lot 0** : le refus reste un `CalibrationUnavailable`
    (la hiérarchie d'exceptions est préservée, `NonInteractiveLineRequired` en
    hérite), mais le diagnostic pointe désormais la cause première — il manque
    `--line` — plutôt que l'absence d'écran, qui n'est plus l'obstacle puisque
    `--line` permet de s'en passer. Les assertions portent donc sur le nouveau
    message ; l'exigence testée (refus explicite, jamais silencieux) est la même.

    Le cas « pas d'écran **sans** `--no-show` » reste couvert par
    `tests/test_main_e2e.py::test_sans_affichage_refuse_de_demarrer`.
    """
    monkeypatch.setattr(entry_point, "display_available", lambda: False)
    config = _overrides("--no-show")

    with pytest.raises(CalibrationUnavailable) as error:
        perform_line_selection(config, "clip.mp4", headless_requested=True)

    message = str(error.value)
    assert "--line" in message
    assert "non interactif" in message
    assert "devin" in message, "le refus doit dire qu'aucune ligne n'est devinée"


def test_calibrate_interface_deux_points(monkeypatch):
    """Vérifie que `calibrate()` transmet exactement deux points normalisés."""
    from calibration import ValidatedLine

    fake_line = ValidatedLine(
        p1=(0.1, 0.4),
        p2=(0.9, 0.4),
        inside_side="negative",
        frame_width=1280,
        frame_height=720,
        display_scale=1.0,
    )
    monkeypatch.setattr(entry_point, "display_available", lambda: True)
    monkeypatch.setattr(entry_point, "select_line", lambda *args, **kwargs: fake_line)

    # Appel direct de calibrate()
    res = calibrate("clip.mp4")
    p1, p2 = res
    assert p1 == (0.1, 0.4)
    assert p2 == (0.9, 0.4)
    assert len(res) == 2
    assert res.inside_side == "negative"


# ---------------------------------------------------------------------------
# Source inouvrable : arrêt en erreur, distinct d'une fin de flux (spec 5.5)
# ---------------------------------------------------------------------------
def test_source_inouvrable_produit_une_erreur_explicite(tmp_path: Path):
    """Le pipeline doit se terminer en erreur, journaliser et ne rien inventer."""
    session_id = "test-source-absente"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "main.py"),
            "--source", str(tmp_path / "jamais.mp4"),
            "--output-root", str(tmp_path / "results"),
            "--session-id", session_id,
            "--no-show",
            # Ligne explicite : sans elle, le mode non interactif refuse de
            # démarrer AVANT d'avoir tenté d'ouvrir la source (lot 0), et le
            # chemin réellement visé par ce test — « source illisible » — ne
            # serait jamais atteint.
            "--line", "0.05,0.6,0.95,0.6",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 5
    events_path = tmp_path / "results" / session_id / "events.jsonl"
    assert events_path.exists()

    events = read_events(events_path)
    types = [event["type"] for event in events]
    assert types == ["SESSION_START", "SOURCE_ERROR", "SESSION_END"]
    # La sélection de ligne échoue d'abord sur la lecture de la première image.
    assert "jamais.mp4" in events[1]["error"]
    # Une source défaillante ne doit produire aucun franchissement.
    assert not [event for event in events if event["type"] in ("IN", "OUT", "NEW")]
