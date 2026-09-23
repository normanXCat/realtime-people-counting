"""Niveau 1 — Boucle de la fenêtre de sélection, avec OpenCV simulé.

`calibration.select_line` ouvre une vraie fenêtre OpenCV : ce test remplace
uniquement les primitives d'affichage (fenêtre, `waitKey`, souris) par un double
scripté, ce qui permet de vérifier **le comportement réel de la boucle** :
deux clics puis « C » valident, « C » trop tôt est refusé sans valider, « G »
efface les points, « I » inverse les côtés, « Échap » et la fermeture de la
fenêtre annulent, et la boucle ne tourne jamais indéfiniment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import calibration  # noqa: E402
import cv2  # noqa: E402
from calibration import (  # noqa: E402
    MODE_CANCEL,
    MODE_LINE,
    MODE_NO_LINE,
    MODE_QUESTION,
    CalibrationCancelled,
    CalibrationUnavailable,
    mode_for_key,
    sanitize_window_name,
    select_line,
)

FRAME = np.full((600, 600, 3), 30, dtype=np.uint8)
KEY_CONFIRM = ord("c")
KEY_RESET = ord("g")
KEY_INVERT = ord("i")
KEY_ESCAPE = 27
KEY_MODE_LINE = ord("o")
KEY_MODE_NO_LINE = ord("n")


class FakeWindow:
    """Double de la fenêtre OpenCV : clics injectés avant chaque touche lue."""

    def __init__(self, sequence, visible: bool = True) -> None:
        #: suite de (clics, touche) jouée par la boucle, dans l'ordre
        self.sequence = [tuple(item) for item in sequence]
        self.visible = visible
        self.callback = None
        self.rendered: list[np.ndarray] = []
        self.idle_polls = 0
        #: ordre réel des appels OpenCV, pour vérifier l'installation du souris
        self.calls: list[str] = []
        #: noms de fenêtre réellement transmis à OpenCV, par appel
        self.window_names: dict[str, list[str]] = {}

    def _record(self, call: str, window: str) -> None:
        self.calls.append(call)
        self.window_names.setdefault(call, []).append(window)

    def named_window(self, window, *_args, **_kwargs) -> None:
        self._record("namedWindow", window)

    def set_mouse_callback(self, window, callback) -> None:
        self._record("setMouseCallback", window)
        self.callback = callback

    def imshow(self, window, image) -> None:
        self._record("imshow", window)
        self.rendered.append(image.copy())

    def wait_key(self, _delay: int) -> int:
        self.calls.append("waitKey")
        if self.callback is None:
            # Aucun callback souris installé : aucun clic ne peut encore arriver,
            # la séquence scriptée n'est pas consommée (comme en vrai).
            return -1
        if not self.sequence:
            # Protège contre une boucle qui ne se terminerait jamais : la suite
            # doit toujours finir par une touche de sortie (« c » ou Échap).
            self.idle_polls += 1
            if self.idle_polls > 200:
                raise AssertionError("la boucle de sélection ne se termine pas")
            return -1
        clicks, key = self.sequence.pop(0)
        for x, y in clicks:
            self.callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return key

    def get_window_property(self, *_args, **_kwargs) -> float:
        return 1.0 if self.visible else 0.0

    def destroy_window(self, *_args, **_kwargs) -> None:
        return None


@pytest.fixture
def fake_window(monkeypatch):
    def _install(
        sequence,
        visible: bool = True,
        frame: np.ndarray | None = None,
        answer: int | None = KEY_MODE_LINE,
    ) -> FakeWindow:
        # La fenêtre s'ouvre désormais sur la question de démarrage (§26) :
        # ``answer`` y répond d'abord (« O » par défaut, calibration inchangée
        # ensuite) ; ``None`` laisse la séquence répondre elle-même.
        if answer is not None:
            sequence = [([], answer), *sequence]
        window = FakeWindow(sequence, visible=visible)
        monkeypatch.setattr(calibration, "display_available", lambda: True)
        monkeypatch.setattr(calibration, "read_first_frame", lambda source: (frame if frame is not None else FRAME).copy())
        monkeypatch.setattr(calibration.cv2, "namedWindow", window.named_window)
        monkeypatch.setattr(calibration.cv2, "setMouseCallback", window.set_mouse_callback)
        monkeypatch.setattr(calibration.cv2, "imshow", window.imshow)
        monkeypatch.setattr(calibration.cv2, "waitKey", window.wait_key)
        monkeypatch.setattr(calibration.cv2, "getWindowProperty", window.get_window_property)
        monkeypatch.setattr(calibration.cv2, "destroyWindow", window.destroy_window)
        monkeypatch.setattr(calibration.cv2, "destroyAllWindows", lambda *args, **kwargs: None)
        return window

    return _install


def test_deux_clics_puis_c_valident_la_ligne(fake_window):
    window = fake_window([([(100, 300), (500, 300)], KEY_CONFIRM)])

    line = select_line("clip.mp4")

    assert line.p1 == pytest.approx((100 / 600, 0.5))
    assert line.p2 == pytest.approx((500 / 600, 0.5))
    assert line.inside_side == "negative"  # proposition de la configuration
    assert (line.frame_width, line.frame_height) == (600, 600)
    assert line.clicks_px == ((100, 300), (500, 300))
    assert line.display_scale == 1.0
    assert window.rendered, "la première image doit être affichée"


def test_c_trop_tot_est_refuse_puis_la_suite_est_validee(fake_window):
    """Un « C » prématuré ne valide rien : la boucle continue et l'opérateur voit l'erreur."""
    fake_window(
        [
            ([(100, 300)], KEY_CONFIRM),           # un seul point : refusé
            ([(500, 300)], KEY_CONFIRM),           # deuxième point : validé
        ]
    )

    line = select_line("clip.mp4")
    assert line.p2 == pytest.approx((500 / 600, 0.5))


def test_g_efface_les_points_precedents(fake_window):
    fake_window(
        [
            ([(100, 100)], KEY_RESET),              # point erroné puis « G »
            ([(100, 300), (500, 300)], KEY_CONFIRM),
        ]
    )

    line = select_line("clip.mp4")
    assert line.clicks_px == ((100, 300), (500, 300))


def test_i_inverse_le_cote_interieur(fake_window):
    fake_window([([(100, 300), (500, 300)], KEY_INVERT), ([], KEY_CONFIRM)])

    line = select_line("clip.mp4", inside_side="negative")
    assert line.inside_side == "positive"


def test_echap_annule_la_selection(fake_window):
    fake_window([([(100, 300)], KEY_ESCAPE)])

    with pytest.raises(CalibrationCancelled):
        select_line("clip.mp4")


def test_fermeture_de_la_fenetre_annule_la_selection(fake_window):
    fake_window([([], -1)], visible=False)

    with pytest.raises(CalibrationCancelled):
        select_line("clip.mp4")


def test_ouverture_de_fenetre_impossible_refusee(monkeypatch):
    """Un backend graphique absent refuse le démarrage au lieu de planter."""
    monkeypatch.setattr(calibration, "display_available", lambda: True)
    monkeypatch.setattr(calibration, "read_first_frame", lambda source: FRAME.copy())

    def _boom(*_args, **_kwargs):
        raise cv2.error("aucun backend graphique")

    monkeypatch.setattr(calibration.cv2, "namedWindow", _boom)
    monkeypatch.setattr(calibration.cv2, "destroyAllWindows", lambda *args, **kwargs: None)
    with pytest.raises(CalibrationUnavailable):
        select_line("clip.mp4")


def test_le_callback_souris_est_installe_apres_le_premier_rendu(fake_window):
    """Régression : sous Qt, `setMouseCallback` avant tout rendu échoue
    en « NULL window handler », ce qui rendait la calibration impossible.

    La fenêtre a donc un premier rendu réel (`imshow` + `waitKey`) avant que le
    callback souris soit installé.
    """
    window = fake_window([([(100, 300), (500, 300)], KEY_CONFIRM)])

    select_line("clip.mp4")

    assert window.calls.index("imshow") < window.calls.index("setMouseCallback"), (
        f"ordre réel des appels : {window.calls}"
    )
    assert "waitKey" in window.calls[: window.calls.index("setMouseCallback") + 1]
    assert window.calls.index("namedWindow") < window.calls.index("imshow")


def test_un_titre_non_ascii_est_assaini_avant_opencl(fake_window):
    """Régression du blocage réel : « NULL window handler in setMouseCallbackImpl ».

    Le nom de fenêtre par défaut de `main.py` contenait un tiret cadratin. Le
    backend Qt d'OpenCV 5.0 ne retrouve alors plus la fenêtre et **refuse**
    d'installer le callback souris : la sélection de ligne devenait impossible,
    donc le comptage ne démarrait jamais. Le nom transmis à OpenCV doit donc
    être ASCII, et identique pour tous les appels.
    """
    window = fake_window([([(100, 300), (500, 300)], KEY_CONFIRM)])

    line = select_line("clip.mp4", window_name="People counting — ligne de comptage")

    assert line.p1 == pytest.approx((100 / 600, 0.5))
    names = {name for values in window.window_names.values() for name in values}
    assert names, "aucun nom de fenêtre transmis à OpenCV"
    for name in names:
        assert name.isascii(), f"nom non ASCII transmis à OpenCV : {name!r}"
        assert "—" not in name
    assert len(names) == 1, f"le nom doit être unique et stable : {names}"


@pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="aucun serveur X : le comportement Qt ne peut pas être vérifié ici",
)
def test_le_backend_graphique_reel_accepte_le_nom_assaini():
    """Vérification directe auprès du backend réellement installé.

    Sur une machine avec écran, `cv2.setMouseCallback` doit accepter le nom
    assaini (et échouer avec un nom non ASCII : c'est le blocage observé en
    production). Sur une machine sans écran, le test est ignoré ; il n'est donc
    jamais bloquant pour la suite, mais il verrouille la régression là où le
    backend Qt est effectivement présent.
    """
    frame = np.full((120, 160, 3), 30, dtype=np.uint8)
    name = sanitize_window_name("People counting — ligne de comptage")
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    except cv2.error as error:  # aucun backend graphique utilisable ici
        pytest.skip(f"backend graphique indisponible : {error}")
    try:
        cv2.imshow(name, frame)
        cv2.waitKey(1)
        cv2.setMouseCallback(name, lambda *args: None)
    finally:
        cv2.destroyAllWindows()


def test_le_callback_souris_qui_echoue_refuse_la_selection(monkeypatch):
    """Si le backend refuse tout de même le callback, le démarrage est refusé
    proprement (message explicite) au lieu de planter."""
    monkeypatch.setattr(calibration, "display_available", lambda: True)
    monkeypatch.setattr(calibration, "read_first_frame", lambda source: FRAME.copy())
    monkeypatch.setattr(calibration.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(calibration.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(calibration.cv2, "waitKey", lambda *args, **kwargs: -1)
    monkeypatch.setattr(calibration.cv2, "destroyAllWindows", lambda *args, **kwargs: None)

    def _boom(*_args, **_kwargs):
        raise cv2.error(
            "NULL window handler in function 'setMouseCallbackImpl'"
        )

    monkeypatch.setattr(calibration.cv2, "setMouseCallback", _boom)

    with pytest.raises(CalibrationUnavailable) as excinfo:
        select_line("clip.mp4")
    assert "setMouseCallback" in str(excinfo.value)


def test_fenetre_reduite_pour_une_tres_grande_image(fake_window):
    """Une image plus grande que la fenêtre est réduite : les clics restent normalisés."""
    large = np.full((2160, 3840, 3), 30, dtype=np.uint8)
    fake_window([([(540, 303), (1000, 303)], KEY_CONFIRM)], frame=large)

    line = select_line("clip.mp4")

    assert line.frame_width == 3840 and line.frame_height == 2160
    assert line.display_scale == pytest.approx(1080 / 3840)
    # 540 / 1080 = 0,5 de la largeur affichée, donc 0,5 de la largeur réelle.
    assert line.p1 == pytest.approx((0.5, 303 / 607), abs=1e-3)


def test_affichage_indisponible_refuse_la_selection(monkeypatch):
    monkeypatch.setattr(calibration, "display_available", lambda: False)
    monkeypatch.setattr(calibration, "read_first_frame", lambda source: FRAME.copy())

    with pytest.raises(CalibrationUnavailable):
        select_line("clip.mp4")


def test_premiere_image_illisible_signalee_comme_source(monkeypatch):
    """L'erreur de source est distinguée d'une erreur d'affichage (spec 5.5)."""
    monkeypatch.setattr(calibration, "display_available", lambda: True)
    monkeypatch.setattr(calibration.cv2, "namedWindow", lambda *args, **kwargs: None)

    with pytest.raises(calibration.SourceUnreadable):
        select_line("chemin/inexistant.mp4")


# ---------------------------------------------------------------------------
# Question de démarrage : « O » ligne, « N » sans ligne (§26)
# ---------------------------------------------------------------------------
def test_reponses_a_la_question_de_demarrage():
    assert mode_for_key(ord("o")) == mode_for_key(ord("O")) == MODE_LINE
    assert mode_for_key(ord("n")) == mode_for_key(ord("N")) == MODE_NO_LINE
    assert mode_for_key(KEY_ESCAPE) == MODE_CANCEL
    assert mode_for_key(ord("c")) is None and mode_for_key(-1 & 0xFF) is None
    assert "O : tracer une ligne virtuelle" in MODE_QUESTION
    assert "N : compter sans ligne" in MODE_QUESTION


def test_n_a_la_question_choisit_le_mode_sans_ligne(fake_window):
    fake_window([([], KEY_MODE_NO_LINE)], answer=None)
    assert select_line("clip.mp4") is None


def test_la_question_est_affichee_avant_la_calibration(fake_window):
    """La première image affichée porte la question, pas l'aide de calibration."""
    window = fake_window([([(100, 300), (500, 300)], KEY_CONFIRM)])
    select_line("clip.mp4")
    question, calibration_view = window.rendered[0], window.rendered[-1]
    assert not np.array_equal(question, calibration_view)
    # Bandeau sombre au centre de l'image pendant la question.
    assert question[300, 5].mean() < FRAME[300, 5].mean()


def test_les_clics_sont_ignores_tant_que_le_mode_n_est_pas_choisi(fake_window):
    fake_window(
        [
            ([(10, 10), (590, 10)], ord("x")),       # clics avant « O » : ignorés
            ([], KEY_MODE_LINE),
            ([(100, 300), (500, 300)], KEY_CONFIRM),
        ],
        answer=None,
    )
    line = select_line("clip.mp4")
    assert line.clicks_px == ((100, 300), (500, 300))


def test_echap_a_la_question_quitte_sans_compter(fake_window):
    fake_window([([], KEY_ESCAPE)], answer=None)
    with pytest.raises(CalibrationCancelled):
        select_line("clip.mp4")
