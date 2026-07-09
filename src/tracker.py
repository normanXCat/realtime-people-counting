from collections import deque
import numpy as np

class LineCrossingTracker:
    """
    Gère le suivi des identifiants (tracking) et détecte le franchissement
    d'une ligne virtuelle pour compter les entrées et les sorties.
    """
    def __init__(self, line_pos_ratio=0.6, max_lost_frames=90, history_len=20):
        """
        Initialise le tracker de franchissement de ligne.
        
        Args:
            line_pos_ratio (float): Ratio vertical de la ligne de comptage (0.0 à 1.0).
            max_lost_frames (int): Nombre maximal de frames d'inactivité avant d'oublier un ID.
            history_len (int): Taille de l'historique des positions pour dessiner les trails.
        """
        self.line_pos_ratio = line_pos_ratio
        self.max_lost_frames = max_lost_frames
        self.history_len = history_len
        
        # États internes de tracking
        self.track_history = {}       # track_id -> deque of (cx, cy)
        self.track_states = {}        # track_id -> 'above' ou 'below'
        self.track_lost_counter = {}  # track_id -> int (compteur d'inactivité)
        
        # Compteurs globaux de franchissement
        self.entries = 0
        self.exits = 0

    def update(self, boxes, frame_height):
        """
        Met à jour l'état du tracker avec les boîtes de détection de la frame courante.
        
        Args:
            boxes: L'attribut result.boxes de YOLO contenant les boîtes et IDs.
            frame_height (int): La hauteur de la frame actuelle (pour calculer la ligne de franchissement).
            
        Returns:
            Tuple (current_count, active_ids_in_frame):
                - current_count (int): Nombre de personnes actuellement dans la frame.
                - active_ids_in_frame (set): Les identifiants actifs dans la frame courante.
        """
        active_ids_in_frame = set()
        current_count = 0
        line_y = int(frame_height * self.line_pos_ratio)

        # Vérification de la présence de boîtes valides avec des IDs de suivi
        if boxes is not None and boxes.id is not None and boxes.xyxy is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxy_list = boxes.xyxy.cpu().numpy()
            current_count = len(track_ids)

            for track_id, bbox in zip(track_ids, xyxy_list):
                active_ids_in_frame.add(track_id)
                x1, y1, x2, y2 = bbox
                
                # Calcul du centroid (milieu de la boîte)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Mise à jour de l'historique de trajectoire
                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=self.history_len)
                self.track_history[track_id].append((cx, cy))

                # Détermination de l'état actuel de la personne par rapport à la ligne
                curr_state = 'above' if cy < line_y else 'below'

                # Enregistrement ou mise à jour de l'état
                if track_id not in self.track_states:
                    self.track_states[track_id] = curr_state
                else:
                    prev_state = self.track_states[track_id]
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

    def _cleanup_lost_tracks(self, active_ids_in_frame):
        """
        Incrémente le compteur de perte pour les identifiants absents de la frame active,
        et purge les structures de données pour libérer de la mémoire si nécessaire.
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
