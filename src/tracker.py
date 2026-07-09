"""
Module de gestion du suivi d'objets et détection de franchissement de ligne.

Ce module définit la classe `LineCrossingTracker` qui exploite les identifiants
de tracking fournis par YOLO/ByteTrack pour suivre les trajectoires des objets
et détecter si et quand ils traversent une frontière virtuelle horizontale.
"""

from collections import deque
from typing import Tuple, Set, Dict, Deque
import numpy as np

class LineCrossingTracker:
    """
    Gère le suivi des identifiants (tracking) et détecte le franchissement
    d'une ligne virtuelle pour compter les entrées et les sorties.

    Attributes:
        line_pos_ratio (float): Ratio vertical de la ligne de comptage (0.0 à 1.0).
        max_lost_frames (int): Nombre maximal de frames d'inactivité avant d'oublier un ID.
        history_len (int): Taille maximale de l'historique des positions pour dessiner les trails.
        track_history (Dict[int, Deque[Tuple[int, int]]]): Historique des centres de boîtes.
        track_states (Dict[int, str]): États par rapport à la ligne ('above' ou 'below').
        track_lost_counter (Dict[int, int]): Nombre de frames d'inactivité cumulées par ID.
        entries (int): Nombre total d'entrées détectées (passage du haut vers le bas).
        exits (int): Nombre total de sorties détectées (passage du bas vers le haut).
    """

    def __init__(self, line_pos_ratio: float = 0.6, max_lost_frames: int = 90, history_len: int = 20):
        """
        Initialise le tracker de franchissement de ligne.
        
        Args:
            line_pos_ratio (float, optional): Ratio vertical de la ligne (fraction de la hauteur). Défaut à 0.6.
            max_lost_frames (int, optional): Seuil de frames perdues avant oubli. Défaut à 90 (environ 3 secondes à 30fps).
            history_len (int, optional): Taille max de la queue pour la trajectoire. Défaut à 20.
        """
        self.line_pos_ratio: float = line_pos_ratio
        self.max_lost_frames: int = max_lost_frames
        self.history_len: int = history_len
        
        # États internes de tracking
        self.track_history: Dict[int, Deque[Tuple[int, int]]] = {}
        self.track_states: Dict[int, str] = {}
        self.track_lost_counter: Dict[int, int] = {}
        
        # Compteurs globaux de franchissement
        self.entries: int = 0
        self.exits: int = 0

    def update(self, boxes, frame_height: int) -> Tuple[int, Set[int]]:
        """
        Met à jour l'état du tracker avec les boîtes de détection de la frame courante.
        
        Calcule les centroids des boîtes, met à jour l'historique des positions, 
        compare l'état actuel (au-dessus/en-dessous) à l'état précédent et met à jour
        les compteurs d'entrées/sorties en cas de franchissement.
        
        Args:
            boxes (ultralytics.engine.results.Boxes): Boîtes renvoyées par YOLO avec les IDs.
            frame_height (int): La hauteur de la frame actuelle (pour calculer la ligne).
            
        Returns:
            Tuple[int, Set[int]]:
                - current_count (int) : Nombre d'objets actuellement visibles.
                - active_ids_in_frame (Set[int]) : Ensemble des identifiants actifs de la frame.
        """
        active_ids_in_frame: Set[int] = set()
        current_count: int = 0
        line_y: int = int(frame_height * self.line_pos_ratio)

        # Vérification de la présence de boîtes valides avec des IDs de suivi
        if boxes is not None and boxes.id is not None and boxes.xyxy is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxy_list = boxes.xyxy.cpu().numpy()
            current_count = len(track_ids)

            for track_id, bbox in zip(track_ids, xyxy_list):
                active_ids_in_frame.add(track_id)
                x1, y1, x2, y2 = bbox
                
                # Calcul du centroid (milieu de la boîte)
                cx: int = int((x1 + x2) / 2)
                cy: int = int((y1 + y2) / 2)

                # Mise à jour de l'historique de trajectoire
                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=self.history_len)
                self.track_history[track_id].append((cx, cy))

                # Détermination de l'état actuel de la personne par rapport à la ligne
                curr_state: str = 'above' if cy < line_y else 'below'

                # Enregistrement ou mise à jour de l'état
                if track_id not in self.track_states:
                    # Première fois qu'on voit cet ID : on stocke son état initial
                    self.track_states[track_id] = curr_state
                else:
                    prev_state: str = self.track_states[track_id]
                    if prev_state != curr_state:
                        # Détection du sens de franchissement
                        if prev_state == 'above' and curr_state == 'below':
                            self.entries += 1
                            print(f"[COMPTAGE] Personne #{track_id} est ENTRÉE (Haut -> Bas)")
                        elif prev_state == 'below' and curr_state == 'above':
                            self.exits += 1
                            print(f"[COMPTAGE] Personne #{track_id} est SORTIE (Bas -> Haut)")
                        
                        # Mise à jour de l'état de référence pour éviter les doubles comptages immédiats
                        self.track_states[track_id] = curr_state

        # Nettoyage des IDs obsolètes (inactifs depuis trop longtemps)
        self._cleanup_lost_tracks(active_ids_in_frame)

        return current_count, active_ids_in_frame

    def _cleanup_lost_tracks(self, active_ids_in_frame: Set[int]) -> None:
        """
        Incrémente le compteur de perte pour les identifiants absents de la frame active.
        
        Purge les deques et états des identifiants qui ont dépassé `max_lost_frames`
        pour libérer les ressources mémoire.
        
        Args:
            active_ids_in_frame (Set[int]): Identifiants présents dans la frame en cours.
        """
        lost_ids = list(self.track_history.keys() - active_ids_in_frame)
        for lid in lost_ids:
            self.track_lost_counter[lid] = self.track_lost_counter.get(lid, 0) + 1
            if self.track_lost_counter[lid] > self.max_lost_frames:
                self.track_history.pop(lid, None)
                self.track_states.pop(lid, None)
                self.track_lost_counter.pop(lid, None)

        # Réinitialisation pour les identifiants actifs détectés
        for aid in active_ids_in_frame:
            self.track_lost_counter[aid] = 0
