"""
Module de rendu graphique et d'overlays OpenCV.

Ce module contient la classe `Visualizer` regroupant les utilitaires de dessin
d'OpenCV pour incruster les trajectoires de suivi (trails), la ligne de franchissement
virtuelle (2 points arbitraires) et le HUD (Heads-Up Display) de statistiques sur le flux vidéo.
"""

import cv2
import numpy as np
from typing import Dict, Deque, Tuple, Union


class Visualizer:
    """
    Fournit des méthodes statiques utilitaires pour dessiner des overlays graphiques
    sur les images OpenCV (HUD, lignes de franchissement arbitraires, trajectoires).
    """

    @staticmethod
    def draw_crossing_line(
        frame: np.ndarray,
        line_p1: Union[Tuple[float, float], Tuple[int, int]],
        line_p2: Union[Tuple[float, float], Tuple[int, int]],
    ) -> None:
        """
        Dessine le segment de ligne virtuelle de franchissement défini par 2 points et son libellé.

        Args:
            frame (np.ndarray): L'image OpenCV (matrice NumPy).
            line_p1 (Tuple): Point 1 (normalisé [0.0, 1.0] ou pixels).
            line_p2 (Tuple): Point 2 (normalisé [0.0, 1.0] ou pixels).
        """
        height, width = frame.shape[:2]

        # Convertir en pixels si les coordonnées sont normalisées (float <= 1.0)
        if isinstance(line_p1[0], float) and line_p1[0] <= 1.0:
            p1_px = (int(line_p1[0] * width), int(line_p1[1] * height))
        else:
            p1_px = (int(line_p1[0]), int(line_p1[1]))

        if isinstance(line_p2[0], float) and line_p2[0] <= 1.0:
            p2_px = (int(line_p2[0] * width), int(line_p2[1] * height))
        else:
            p2_px = (int(line_p2[0]), int(line_p2[1]))

        # Dessin du segment de ligne jaune
        cv2.line(frame, p1_px, p2_px, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, p1_px, 5, (0, 165, 255), -1)
        cv2.circle(frame, p2_px, 5, (0, 165, 255), -1)

        # Libellé au milieu du segment
        mid_x = int((p1_px[0] + p2_px[0]) / 2)
        mid_y = int((p1_px[1] + p2_px[1]) / 2)
        cv2.putText(
            frame,
            "LIGNE DE COMPTAGE",
            (max(10, mid_x - 70), max(20, mid_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_trails(
        frame: np.ndarray, track_history: Dict[int, Deque[Tuple[int, int]]]
    ) -> None:
        """
        Dessine les trajectoires de suivi historiques sous forme de trainée verte s'estompant.

        Trace des lignes successives d'épaisseur décroissante reliant les centroids passés,
        et ajoute un point de repère bleu turquoise sur le dernier centroid actif.

        Args:
            frame (np.ndarray): L'image (matrice NumPy) OpenCV.
            track_history (Dict[int, Deque[Tuple[int, int]]]): Dictionnaire associant
                chaque ID actif à la queue de ses positions récentes.
        """
        for tid, path in track_history.items():
            if len(path) > 1:
                for i in range(1, len(path)):
                    thickness: int = int(np.sqrt(len(path) / float(i + 1)) * 2)
                    cv2.line(frame, path[i - 1], path[i], (0, 255, 0), max(1, thickness))
                cv2.circle(frame, path[-1], 4, (255, 255, 0), -1)

    @staticmethod
    def draw_hud(
        frame: np.ndarray, current_count: int, entries: int, exits: int
    ) -> None:
        """
        Dessine un HUD semi-transparent compact et adaptatif.

        Superpose un petit rectangle noir transparent en haut à gauche et y inscrit le nombre de
        présents en direct, le total des entrées, et le total des sorties.

        Args:
            frame (np.ndarray): L'image OpenCV.
            current_count (int): Nombre de personnes actuellement dans la frame.
            entries (int): Nombre cumulé d'entrées.
            exits (int): Nombre cumulé de sorties.
        """
        height, width = frame.shape[:2]
        # Échelle adaptative calculée sur la largeur de l'image
        scale = max(0.45, min(1.0, width / 1280.0))

        box_w = int(230 * scale)
        box_h = int(72 * scale)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (15, 15, 15), -1)
        alpha: float = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (120, 120, 120), 1)

        f_scale = 0.42 * scale
        th = max(1, int(1 * scale))

        # Titre compact
        cv2.putText(
            frame,
            "TRACKING (ByteTrack + ReID)",
            (16, 10 + int(18 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale * 0.85,
            (0, 255, 255),
            th,
            cv2.LINE_AA,
        )
        # Présents
        cv2.putText(
            frame,
            f"Presents : {current_count}",
            (16, 10 + int(38 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale,
            (255, 255, 255),
            th,
            cv2.LINE_AA,
        )
        # Entrées / Sorties
        cv2.putText(
            frame,
            f"In: {entries}  |  Out: {exits}",
            (16, 10 + int(58 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale,
            (0, 255, 0),
            th,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_boxes(
        frame: np.ndarray,
        xyxy_list: np.ndarray,
        track_ids: list,
        confidences: np.ndarray = None,
    ) -> None:
        """
        Dessine des bounding boxes et des étiquettes d'ID compactes.

        Args:
            frame (np.ndarray): L'image OpenCV.
            xyxy_list (np.ndarray): Matrice des coordonnées [x1, y1, x2, y2].
            track_ids (list): Liste des identifiants.
            confidences (np.ndarray, optional): Scores de confiance correspondants.
        """
        if xyxy_list is None or track_ids is None:
            return

        for i, (bbox, tid) in enumerate(zip(xyxy_list, track_ids)):
            x1, y1, x2, y2 = map(int, bbox[:4])
            color = (0, 255, 0)  # Vert

            # Dessin de la bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Libellé compact : ID et score optionnel
            label = f"#{tid}"
            if confidences is not None and i < len(confidences):
                label += f" {confidences[i]:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
            )
            lbl_y1 = max(0, y1 - text_h - 4)
            lbl_y2 = max(text_h + 4, y1)
            cv2.rectangle(frame, (x1, lbl_y1), (x1 + text_w + 4, lbl_y2), color, -1)

            cv2.putText(
                frame,
                label,
                (x1 + 2, lbl_y2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
