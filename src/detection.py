import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# Configuration des chemins
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "yolo11s.pt"

def parse_args():
    parser = argparse.ArgumentParser(description="Suivi et comptage de personnes en temps réel avec YOLO11 & ByteTrack.")
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Source vidéo : index de caméra (ex: '0') ou chemin d'un fichier vidéo."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=str(DEFAULT_MODEL_PATH), 
        help="Chemin vers le modèle YOLO (.pt)."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.5, 
        help="Seuil de confiance pour la détection."
    )
    parser.add_argument(
        "--line-pos", 
        type=float, 
        default=0.6, 
        help="Position de la ligne de franchissement (fraction de la hauteur, 0.0 à 1.0)."
    )
    parser.add_argument(
        "--no-show", 
        action="store_true", 
        help="Désactiver l'affichage de la fenêtre OpenCV (utile pour les tests ou exécution en arrière-plan)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validation du modèle
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Modèle non trouvé à l'emplacement : {model_path}")
        print("Veuillez spécifier un modèle existant ou le placer dans le dossier 'models'.")
        return
        
    # Initialisation du modèle YOLO
    model = YOLO(str(model_path))
    
    # Résolution de la source vidéo (entier pour webcam, chaîne pour fichier)
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Dictionnaires pour le suivi des trajectoires et états de franchissement
    track_history = {}       # track_id -> deque de positions (cx, cy)
    track_states = {}        # track_id -> 'above' ou 'below'
    track_lost_counter = {}  # track_id -> nombre de frames perdues
    
    # Compteurs globaux
    entries = 0
    exits = 0
    
    # Configuration de la ligne
    line_pos_ratio = args.line_pos
    
    print(f"Lancement du suivi (ByteTrack) sur la source: {source}")
    print(f"Ligne de comptage configurée à {line_pos_ratio * 100}% de la hauteur.")
    print("Appuyez sur 'q' dans la fenêtre vidéo pour quitter, ou Ctrl+C dans le terminal.")
    
    try:
        # Utilisation de model.track avec bytetrack
        results = model.track(
            source=source,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],  # 0 correspond à 'person' dans COCO
            conf=args.conf,
            show=False,   # On gère l'affichage nous-mêmes
            stream=True
        )
        
        for result in results:
            # Récupérer l'image annotée par YOLO (boîtes englobantes et IDs de suivi de base)
            frame = result.plot(labels=True, conf=True, boxes=True)
            
            # Dimensions de l'image
            height, width = frame.shape[:2]
            line_y = int(height * line_pos_ratio)
            
            # Récupération des détections et des identifiants de suivi
            boxes = result.boxes
            current_count = 0
            active_ids_in_frame = set()
            
            if boxes is not None and boxes.id is not None and boxes.xyxy is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_list = boxes.xyxy.cpu().numpy()
                
                current_count = len(track_ids)
                
                for track_id, bbox in zip(track_ids, xyxy_list):
                    active_ids_in_frame.add(track_id)
                    x1, y1, x2, y2 = bbox
                    
                    # Calcul du centre (centroid) du rectangle de détection
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Initialisation/Mise à jour de l'historique de trajectoire
                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=20)
                    track_history[track_id].append((cx, cy))
                    
                    # Détermination du franchissement de ligne
                    curr_state = 'above' if cy < line_y else 'below'
                    
                    if track_id not in track_states:
                        # Première fois qu'on voit cet ID : on stocke son état initial
                        track_states[track_id] = curr_state
                    else:
                        prev_state = track_states[track_id]
                        if prev_state != curr_state:
                            # Changement d'état détecté = franchissement !
                            if prev_state == 'above' and curr_state == 'below':
                                # Franchissement du haut vers le bas -> Entrée
                                entries += 1
                                print(f"[COMPTAGE] Personne #{track_id} est ENTRÉE (Haut -> Bas)")
                            elif prev_state == 'below' and curr_state == 'above':
                                # Franchissement du bas vers le haut -> Sortie
                                exits += 1
                                print(f"[COMPTAGE] Personne #{track_id} est SORTIE (Bas -> Haut)")
                            
                            # Mettre à jour l'état de référence pour éviter des double-comptages immédiats
                            track_states[track_id] = curr_state
            
            # Nettoyage des IDs obsolètes (non détectés depuis plus de 90 frames / ~3 secondes)
            lost_ids = list(track_history.keys() - active_ids_in_frame)
            for lid in lost_ids:
                track_lost_counter[lid] = track_lost_counter.get(lid, 0) + 1
                if track_lost_counter[lid] > 90:
                    track_history.pop(lid, None)
                    track_states.pop(lid, None)
                    track_lost_counter.pop(lid, None)
            
            # Réinitialisation du compteur de perte pour les IDs détectés
            for aid in active_ids_in_frame:
                track_lost_counter[aid] = 0
            
            # Dessin de la ligne de comptage (Ligne horizontale jaune)
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
            
            # Dessin des trajectoires (trails) de suivi pour chaque personne active
            for tid, path in track_history.items():
                if len(path) > 1:
                    for i in range(1, len(path)):
                        # Épaisseur du tracé s'estompant avec l'ancienneté du point
                        thickness = int(np.sqrt(20 / float(i + 1)) * 2)
                        cv2.line(frame, path[i - 1], path[i], (0, 255, 0), max(1, thickness))
                    # Point bleu turquoise sur le dernier centre
                    cv2.circle(frame, path[-1], 4, (255, 255, 0), -1)
            
            # Dessin de l'incrustation HUD (Heads-Up Display) semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (330, 140), (15, 15, 15), -1)
            alpha = 0.65  # Opacité
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Bordure fine grise pour le HUD
            cv2.rectangle(frame, (10, 10), (330, 140), (120, 120, 120), 1)
            
            # Texte dans le HUD
            cv2.putText(frame, "TRACKING & COMPTAGE (ByteTrack)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Presents en direct : {current_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Entrees (In)        : {entries}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Sorties (Out)       : {exits}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Affichage de la fenêtre (sauf si no-show est activé)
            if not args.no_show:
                cv2.imshow("Real-time People Counting & Tracking", frame)
                
                # Quitter si la touche 'q' est pressée
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Arrêt demandé via la touche 'q'...")
                    break
            else:
                # Mode headless : log minimal périodique
                print(f"[LIVE] Présents: {current_count} | Entrées: {entries} | Sorties: {exits}", end="\r")
                
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur (Ctrl+C)...")
        
    finally:
        # Nettoyage des ressources
        cv2.destroyAllWindows()
        print("\n--- STATISTIQUES FINALES ---")
        print(f"Total Entrées : {entries}")
        print(f"Total Sorties : {exits}")
        print("Application fermée.")

if __name__ == "__main__":
    main()
