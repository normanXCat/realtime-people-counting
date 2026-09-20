"""Parcours complets de `src/main.py` avec détection et sélection simulées.

Ces tests vérifient de bout en bout, sans caméra, sans affichage et sans modèle
réel :

- la ligne retenue est **exactement** celle validée par l'opérateur, et c'est
  elle qui sert au dessin, à la distance signée, aux franchissements et au
  comptage (exigences 8 et 9) ;
- une annulation (Échap) ou une machine sans affichage arrête le programme
  **sans lancer le moindre comptage** (exigence 10) ;
- aucune ligne par défaut n'est jamais utilisée (exigence 5) ;
- une source inouvrable reste un cas d'erreur distinct (spec 5.5).

`ultralytics` est remplacé par un double déterministe : le pipeline complet
(journal, session, rendu, occupation) est exercé sans téléchargement ni GPU.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import main as entry_point  # noqa: E402
from calibration import CalibrationCancelled, ValidatedLine  # noqa: E402
from events import read_events  # noqa: E402

FRAME_SIDE = 600


def validated_line(
    p1=(0.2, 0.4), p2=(0.8, 0.4), inside_side="negative"
) -> ValidatedLine:
    """Ligne telle que `calibration.select_line` la retourne après deux clics + « C »."""
    return ValidatedLine(
        p1=p1,
        p2=p2,
        inside_side=inside_side,
        frame_width=FRAME_SIDE,
        frame_height=FRAME_SIDE,
        display_scale=1.0,
        clicks_px=((p1[0] * FRAME_SIDE, p1[1] * FRAME_SIDE), (p2[0] * FRAME_SIDE, p2[1] * FRAME_SIDE)),
        confirmed_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Double déterministe d'ultralytics
# ---------------------------------------------------------------------------
class _FakeTensor:
    """Suffisant pour les appels de `main` (``.int().cpu().tolist()``, ``.cpu().numpy()``)."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def int(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values

    def tolist(self):
        return [int(value) for value in self._values]


class _FakeBoxes:
    def __init__(self, detections):
        self.id = _FakeTensor([item[0] for item in detections])
        self.xyxy = _FakeTensor([item[1] for item in detections])
        self.conf = _FakeTensor([item[2] for item in detections])


class _FakeResult:
    def __init__(self, frame, detections):
        self.orig_img = frame.copy()
        self.speed = {"inference": 1.0, "postprocess": 0.5}
        self.boxes = _FakeBoxes(detections) if detections else None


def install_fake_ultralytics(monkeypatch, frames: int = 8, detections=()) -> np.ndarray:
    """Installe un faux module `ultralytics` rendant `frames` images synthétiques."""
    module = types.ModuleType("ultralytics")
    image = np.zeros((FRAME_SIDE, FRAME_SIDE, 3), dtype=np.uint8)

    class _FakeModel:
        def __init__(self, path):
            self.path = path

        def track(self, **_kwargs):
            for _ in range(frames):
                yield _FakeResult(image, detections)

    module.YOLO = _FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return image


def run_main(
    tmp_path: Path,
    monkeypatch,
    *,
    line: ValidatedLine | None = None,
    source: str = "clip.mp4",
    session_id: str = "session-e2e",
    frames: int = 8,
    detections=(),
    extra=(),
) -> int:
    """Exécute `main` avec une sélection de ligne simulée et une détection double."""
    if line is not None:
        monkeypatch.setattr(
            entry_point, "perform_line_selection", lambda config, src, **kwargs: line
        )
    install_fake_ultralytics(monkeypatch, frames=frames, detections=detections)
    return entry_point.main(
        [
            "--source", source,
            "--output-root", str(tmp_path),
            "--session-id", session_id,
            "--no-show",
            *extra,
        ]
    )


def session_dir(tmp_path: Path, session_id: str = "session-e2e") -> Path:
    return tmp_path / session_id


def write_video(path: Path, frames: int = 3) -> Path:
    """Vidéo minimale lisible, pour les tests qui ouvrent réellement la source."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (FRAME_SIDE, FRAME_SIDE)
    )
    assert writer.isOpened(), "codec mp4v indisponible dans cet environnement"
    for index in range(frames):
        writer.write(np.full((FRAME_SIDE, FRAME_SIDE, 3), 40 + index, dtype=np.uint8))
    writer.release()
    return path


def capture_display(monkeypatch) -> list[np.ndarray]:
    """Active le rendu en interceptant les images passées à `cv2.imshow`."""
    captured: list[np.ndarray] = []
    monkeypatch.setattr(entry_point, "display_available", lambda: True)
    monkeypatch.setattr(entry_point.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        entry_point.cv2, "imshow", lambda _window, image: captured.append(image.copy())
    )
    monkeypatch.setattr(entry_point.cv2, "waitKey", lambda *args, **kwargs: -1)
    monkeypatch.setattr(entry_point.cv2, "destroyAllWindows", lambda *args, **kwargs: None)
    return captured


def run_main_with_display(tmp_path: Path, monkeypatch, line: ValidatedLine, session_id: str) -> int:
    monkeypatch.setattr(
        entry_point, "perform_line_selection", lambda config, source, **kwargs: line
    )
    install_fake_ultralytics(monkeypatch)
    return entry_point.main(
        [
            "--source", "clip.mp4",
            "--output-root", str(tmp_path),
            "--session-id", session_id,
        ]
    )


def has_green(image: np.ndarray, row: int, column: int, radius: int = 3) -> bool:
    """Un pixel franchement vert existe-t-il autour de (row, column) ?"""
    window = image[
        max(0, row - radius): row + radius + 1,
        max(0, column - radius): column + radius + 1,
    ]
    green = (window[:, :, 1] > 200) & (window[:, :, 0] < 80) & (window[:, :, 2] < 80)
    return bool(green.any())


# ---------------------------------------------------------------------------
# La ligne validée est celle utilisée partout
# ---------------------------------------------------------------------------
def test_la_ligne_validee_est_utilisee_pour_le_comptage(tmp_path: Path, monkeypatch):
    """L'orientation choisie dans la fenêtre l'emporte sur la proposition de config."""
    line = validated_line(p1=(0.25, 0.40), p2=(0.75, 0.40), inside_side="positive")

    code = run_main(tmp_path, monkeypatch, line=line)

    assert code == 0
    root = session_dir(tmp_path)

    events = read_events(root / "events.jsonl")
    types = [event["type"] for event in events]
    assert types[0] == "SESSION_START"
    assert events[0]["mode"] == "comptage"
    assert events[0]["line_policy"] == "manual_required"
    assert "LINE_VALIDATED" in types
    validated_event = next(event for event in events if event["type"] == "LINE_VALIDATED")
    assert validated_event["p1"] == pytest.approx([0.25, 0.40])
    assert validated_event["p2"] == pytest.approx([0.75, 0.40])
    assert validated_event["inside_side"] == "positive"
    assert validated_event["frame_resolution"] == [FRAME_SIDE, FRAME_SIDE]

    # La ligne sélectionnée est archivée avec sa traçabilité complète.
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["line"]["p1_normalized"] == [0.25, 0.4]
    assert calibration["line"]["inside_side"] == "positive"
    assert calibration["line"]["clicks_px"] == [[150.0, 240.0], [450.0, 240.0]]

    # La configuration résolue de la session fige l'orientation réellement retenue.
    resolved = (root / "config_resolved.yaml").read_text(encoding="utf-8")
    assert "inside_side: positive" in resolved


def test_le_dessin_reprend_exactement_la_ligne_validee(tmp_path: Path, monkeypatch):
    """Aucune ligne horizontale par défaut ne doit subsister dans le rendu."""
    captured = capture_display(monkeypatch)
    line = validated_line(p1=(0.25, 0.40), p2=(0.75, 0.40), inside_side="negative")

    code = run_main_with_display(tmp_path, monkeypatch, line, "session-dessin")

    assert code == 0
    assert captured, "aucune image n'a été rendue"
    image = captured[-1]
    height, width = image.shape[:2]

    # La ligne validée est tracée à y = 0,40 × hauteur, entre x = 0,25 et 0,75.
    assert has_green(image, int(0.40 * height), int(0.50 * width))
    # L'ancienne ligne par défaut (horizontale à 60 %) n'est plus jamais utilisée.
    assert not has_green(image, int(0.60 * height), int(0.50 * width))
    assert not has_green(image, int(0.60 * height), int(0.30 * width))


def test_ligne_inclinee_utilisee_telle_quelle(tmp_path: Path, monkeypatch):
    """Une ligne inclinée validée est tracée et utilisée telle quelle."""
    captured = capture_display(monkeypatch)
    line = validated_line(p1=(0.2, 0.2), p2=(0.8, 0.8), inside_side="negative")

    code = run_main_with_display(tmp_path, monkeypatch, line, "session-inclinee")

    assert code == 0
    image = captured[-1]
    height, width = image.shape[:2]
    # Le milieu du segment validé est le centre de l'image...
    assert has_green(image, int(0.50 * height), int(0.50 * width))
    # ... et les coins hors de la ligne ne le sont pas.
    assert not has_green(image, int(0.25 * height), int(0.75 * width))
    assert not has_green(image, int(0.75 * height), int(0.25 * width))


# ---------------------------------------------------------------------------
# Pas de ligne validée => pas de comptage
# ---------------------------------------------------------------------------
def test_annulation_par_l_operateur_ne_lance_aucun_comptage(tmp_path: Path, monkeypatch):
    def _annule(config, source, **_kwargs):
        raise CalibrationCancelled("Sélection de la ligne annulée par l'opérateur (Échap).")

    monkeypatch.setattr(entry_point, "perform_line_selection", _annule)
    install_fake_ultralytics(monkeypatch)

    code = entry_point.main(
        [
            "--source", "clip.mp4",
            "--output-root", str(tmp_path),
            "--session-id", "session-annulee",
            "--no-show",
        ]
    )

    assert code == 4
    root = session_dir(tmp_path, "session-annulee")
    types = [event["type"] for event in read_events(root / "events.jsonl")]
    assert types == ["SESSION_START", "LINE_CALIBRATION_CANCELLED", "SESSION_END"]
    assert not (root / "calibration.json").exists()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["aborted"] == "LINE_CALIBRATION_CANCELLED"
    assert summary["counters"] is None


def test_no_show_sans_ligne_refuse_de_demarrer(tmp_path: Path, monkeypatch, capsys):
    """`--no-show` sans `--line` : erreur claire, aucun comptage.

    Attente **ajustée par le lot 0** (`docs/lot_0_outillage.md` §0.1). Avant, le
    refus venait de l'absence d'interface graphique et sortait en code 4. Le
    mode non interactif exige désormais `--line`, et ce refus-là porte un code
    dédié (6) pour ne pas être confondu avec « aucun écran disponible » : les
    deux situations se corrigent différemment.

    Ce qui est garanti reste identique : aucun comptage, aucune ligne devinée,
    aucun `calibration.json`, et un refus journalisé.
    """
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("calibration.display_available", lambda: False)
    monkeypatch.setattr(entry_point, "display_available", lambda: False)
    video = write_video(tmp_path / "clip.mp4")

    code = entry_point.main(
        [
            "--source", str(video),
            "--output-root", str(tmp_path / "results"),
            "--session-id", "session-headless",
            "--no-show",
        ]
    )

    assert code == 6
    root = tmp_path / "results" / "session-headless"
    types = [event["type"] for event in read_events(root / "events.jsonl")]
    assert types == ["SESSION_START", "LINE_CALIBRATION_UNAVAILABLE", "SESSION_END"]
    assert not (root / "calibration.json").exists()
    stderr = capsys.readouterr().err
    assert "--line" in stderr
    assert "non interactif" in stderr


def test_sans_affichage_refuse_de_demarrer(tmp_path: Path, monkeypatch):
    """Sans écran (même sans `--no-show`), la ligne ne peut pas être définie."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("calibration.display_available", lambda: False)
    monkeypatch.setattr(entry_point, "display_available", lambda: False)
    video = write_video(tmp_path / "clip.mp4")

    code = entry_point.main(
        [
            "--source", str(video),
            "--output-root", str(tmp_path / "results"),
            "--session-id", "session-ecran-absent",
        ]
    )

    assert code == 4
    root = tmp_path / "results" / "session-ecran-absent"
    types = [event["type"] for event in read_events(root / "events.jsonl")]
    assert types == ["SESSION_START", "LINE_CALIBRATION_UNAVAILABLE", "SESSION_END"]
    assert not (root / "calibration.json").exists()


def test_source_inouvrable_pendant_la_selection(tmp_path: Path):
    """Source inouvrable : cas d'erreur distinct d'une annulation (spec 5.5)."""
    code = entry_point.main(
        [
            "--source", str(tmp_path / "jamais.mp4"),
            "--output-root", str(tmp_path / "results"),
            "--session-id", "session-source-absente",
            "--no-show",
            # Ligne explicite : sans elle, le mode non interactif refuse avant
            # même d'ouvrir la source (lot 0), et le chemin visé ici — source
            # illisible, code 5 — ne serait pas atteint.
            "--line", "0.05,0.6,0.95,0.6",
        ]
    )

    assert code == 5
    root = tmp_path / "results" / "session-source-absente"
    types = [event["type"] for event in read_events(root / "events.jsonl")]
    assert types == ["SESSION_START", "SOURCE_ERROR", "SESSION_END"]
