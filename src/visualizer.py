"""
Module de rendu graphique et d'overlays OpenCV.

Ce module contient la classe `Visualizer` regroupant les utilitaires de dessin
d'OpenCV pour incruster les trajectoires de suivi (trails), la ligne de franchissement
virtuelle et le HUD (Heads-Up Display) de statistiques sur le flux vidéo.
"""

import cv2
import numpy as np
from typing import Dict, Deque, Tuple

class Visualizer:
    """
    Fournit des méthodes statiques utilitaires pour dessiner des overlays graphiques
    sur les images OpenCV (HUD, lignes de franchissement, trajectoires).
    """
    
    @staticmethod
    def draw_crossing_line(frame: np.ndarray, line_pos_ratio: float) -> None:
        """
        Dessine la ligne virtuelle horizontale de franchissement et son libellé.
        
        Args:
            frame (np.ndarray): L'image (matrice NumPy) OpenCV sur laquelle dessiner.
            line_pos_ratio (float): Ratio vertical de la position de la ligne (0.0 à 1.0).
        """
        height, width = frame.shape[:2]
        line_y: int = int(height * line_pos_ratio)
        
        # Dessin de la ligne horizontale jaune
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
        cv2.putText(
            frame, 
            "LIGNE DE COMPTAGE", 
            (15, line_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            1, 
            cv2.LINE_AA
        )

    @staticmethod
    def draw_trails(frame: np.ndarray, track_history: Dict[int, Deque[Tuple[int, int]]]) -> None:
        """
        Dessine les trajectoires de suivi historiques sous forme de trainée verte s'estompant.
        
        Trace des lignes successives d'épaisseur décroissante reliant les centroids passés,
        et ajoute un point de repère bleu turquoise sur le dernier centroid actif.
        
        Args:
            frame (np.ndarray): L'image (matrice NumPy) OpenCV sur laquelle dessiner.
            track_history (Dict[int, Deque[Tuple[int, int]]]): Dictionnaire associant
                chaque ID actif à la queue de ses positions récentes.
        """
        for tid, path in track_history.items():
            if len(path) > 1:
                for i in range(1, len(path)):
                    # Épaisseur proportionnelle à l'ancienneté du point dans la queue
                    thickness: int = int(np.sqrt(len(path) / float(i + 1)) * 2)
                    cv2.line(frame, path[i - 1], path[i], (0, 255, 0), max(1, thickness))
                # Dessin du centre actuel en bleu turquoise
                cv2.circle(frame, path[-1], 4, (255, 255, 0), -1)

    @staticmethod
    def draw_hud(frame: np.ndarray, current_count: int, entries: int, exits: int) -> None:
        """
        Dessine un HUD semi-transparent contenant les statistiques dynamiques de comptage.
        
        Superpose un rectangle noir transparent en haut à gauche et y inscrit le nombre de 
        présents en direct, le total des entrées, et le total des sorties.
        
        Args:
            frame (np.ndarray): L'image (matrice NumPy) OpenCV sur laquelle dessiner.
            current_count (int): Nombre de personnes actuellement dans la frame.
            entries (int): Nombre cumulé d'entrées.
            exits (int): Nombre cumulé de sorties.
        """
        # Création de l'overlay translucide
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 140), (15, 15, 15), -1)
        alpha: float = 0.65  # Facteur d'opacité
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Bordure fine grise
        cv2.rectangle(frame, (10, 10), (330, 140), (120, 120, 120), 1)
        
        # Affichage des textes d'informations
        cv2.putText(frame, "TRACKING & COMPTAGE (ByteTrack)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Presents en direct : {current_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Entrees (In)        : {entries}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sorties (Out)       : {exits}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
