"""
Module de calibration manuelle de la ligne virtuelle de comptage.

Affiche la première frame de la source vidéo et permet à l'opérateur de
cliquer deux points pour définir la ligne de franchissement.

Contrôles clavier :
    - Clic gauche : poser un point (maximum 2).
    - Touche 'g'  : réinitialiser les points (recommencer la sélection).
    - Touche 'c'  : valider la calibration (nécessite exactement 2 points).

Les coordonnées sont retournées **normalisées** (x/largeur, y/hauteur),
entre 0.0 et 1.0, afin de rester valides quelle que soit la résolution.
"""

import cv2
import numpy as np
from typing import Tuple, List


def calibrate_line(source) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Ouvre une fenêtre de calibration interactive sur la première frame de la
    source vidéo et laisse l'utilisateur cliquer deux points pour définir
    la ligne de franchissement.

    Args:
        source: Index de caméra (int) ou chemin/URL d'un fichier vidéo (str).

    Returns:
        Tuple de deux tuples (x_norm, y_norm) représentant les extrémités
        de la ligne en coordonnées normalisées [0.0, 1.0].

    Raises:
        RuntimeError: Si la première frame ne peut pas être lue.
    """
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(
            f"Impossible de lire la première frame de la source : {source}"
        )

    height, width = frame.shape[:2]
    points: List[Tuple[int, int]] = []
    clone = frame.copy()

    # ---- Callback souris ----
    def _mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                # Dessiner le point rouge
                cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
                cv2.circle(clone, (x, y), 8, (255, 255, 255), 1)
                # Dessiner la ligne provisoire dès que le second point est posé
                if len(points) == 2:
                    cv2.line(clone, points[0], points[1], (0, 255, 255), 2)
                cv2.imshow(win_name, clone)

    win_name = "Calibration - Cliquez 2 points | g: reset | c: valider"
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # Redimensionner la fenêtre si la vidéo dépasse 1280x720 tout en conservant le ratio
        max_w, max_h = 1280, 720
        scale = min(max_w / width, max_h / height, 1.0)
        disp_w, disp_h = int(width * scale), int(height * scale)
        cv2.resizeWindow(win_name, disp_w, disp_h)

        cv2.imshow(win_name, clone)
        cv2.setMouseCallback(win_name, _mouse_callback)
    except Exception as err:
        print(f"[Calibration] Impossible d'ouvrir la fenêtre graphique de calibration ({err}).")
        print("[Calibration] Utilisation de la ligne par défaut (horizontale à 60 % de la hauteur).")
        return ((0.0, 0.6), (1.0, 0.6))

    while True:
        # Redessiner le canvas clone
        draw_canvas = clone.copy()
        _draw_instructions(draw_canvas)
        cv2.imshow(win_name, draw_canvas)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("c"):
            # Valider uniquement si exactement 2 points
            if len(points) == 2:
                break
            else:
                print(
                    "[Calibration] Validation impossible : il manque "
                    f"{2 - len(points)} point(s). Cliquez d'abord 2 points."
                )

        elif key == ord("g"):
            # Réinitialiser la sélection
            points.clear()
            clone = frame.copy()
            print("[Calibration] Points réinitialisés.")

        elif key == 27:  # ESC - quitter avec ligne par défaut
            print(
                "[Calibration] Annulé (ESC). "
                "Utilisation de la ligne par défaut (horizontale à 60 % de la hauteur)."
            )
            points = [(0, int(height * 0.6)), (width, int(height * 0.6))]
            break

    cv2.destroyWindow(win_name)

    # Normalisation des coordonnées
    p1_norm = (points[0][0] / width, points[0][1] / height)
    p2_norm = (points[1][0] / width, points[1][1] / height)

    print(
        f"[Calibration] Ligne validée : P1=({p1_norm[0]:.4f}, {p1_norm[1]:.4f}) "
        f"P2=({p2_norm[0]:.4f}, {p2_norm[1]:.4f})"
    )
    return (p1_norm, p2_norm)


def _draw_instructions(frame: np.ndarray) -> None:
    """Affiche un bandeau d'instructions en haut de la frame."""
    cv2.putText(
        frame,
        "Cliquez 2 points | g: reinitialiser | c: valider",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
