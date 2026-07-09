"""
Script principal de détection, suivi (tracking) et comptage de personnes.

Ce script initialise la capture d'images (webcam, fichier vidéo ou flux réseau),
instancie le modèle YOLO (YOLO11 par défaut), charge le tracker ByteTrack
pour suivre les objets détectés, et applique des overlays graphiques
(ligne de franchissement virtuelle, trainées de trajectoires, HUD de statistiques)
pour afficher le comptage des entrées et des sorties en temps réel.
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

# Importations des modules locaux
from tracker import LineCrossingTracker
from visualizer import Visualizer

# Configuration des chemins
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "yolo11s.pt"

def parse_args() -> argparse.Namespace:
    """
    Configure et analyse les arguments passés en ligne de commande.
    
    Permet à l'utilisateur de spécifier la source vidéo, le modèle YOLO,
    la confiance de détection, la hauteur de la ligne virtuelle et le mode headless.
    
    Returns:
        argparse.Namespace: Un objet contenant tous les arguments configurés et analysés.
    """
    parser = argparse.ArgumentParser(
        description="Suivi et comptage de personnes en temps réel avec YOLO11 & ByteTrack."
    )
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

def main() -> None:
    """
    Point d'entrée principal du programme.
    
    Valide les prérequis comme l'existence du modèle, initialise YOLO,
    démarre le tracker et lance la boucle infinie de traitement vidéo.
    Intercepte les signaux d'interruption utilisateur pour un nettoyage propre.
    """
    args = parse_args()
    
    # Validation du modèle
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Modèle non trouvé à l'emplacement : {model_path}")
        print("Veuillez spécifier un modèle existant ou le placer dans le dossier 'models'.")
        return
        
    # Initialisation du modèle YOLO
    model = YOLO(str(model_path))
    
    # Résolution de la source vidéo (entier pour webcam, chaîne pour fichier ou flux)
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialisation du tracker (gestion du tracking et de la ligne de comptage)
    tracker = LineCrossingTracker(line_pos_ratio=args.line_pos)
    
    print(f"Lancement du suivi (ByteTrack) sur la source: {source}")
    print(f"Ligne de comptage configurée à {args.line_pos * 100}% de la hauteur.")
    print("Appuyez sur 'q' dans la fenêtre vidéo pour quitter, ou Ctrl+C dans le terminal.")
    
    try:
        # Démarrage du suivi YOLO avec ByteTrack
        results = model.track(
            source=source,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],  # 0 correspond à 'person' dans COCO
            conf=args.conf,
            show=False,   # Rendu graphique manuel
            stream=True
        )
        
        for result in results:
            # Récupérer la frame avec les détections tracées par YOLO
            frame = result.plot(labels=True, conf=True, boxes=True)
            height, width = frame.shape[:2]
            
            # Mise à jour du tracker et calcul du comptage
            current_count, active_ids = tracker.update(result.boxes, height)
            
            # Dessin de la ligne de franchissement
            Visualizer.draw_crossing_line(frame, tracker.line_pos_ratio)
            
            # Dessin des trails/trajectoires de chaque personne
            Visualizer.draw_trails(frame, tracker.track_history)
            
            # Dessin du HUD d'informations (Entrées, Sorties, Présents)
            Visualizer.draw_hud(frame, current_count, tracker.entries, tracker.exits)
            
            # Affichage de la frame
            if not args.no_show:
                cv2.imshow("Real-time People Counting & Tracking", frame)
                
                # Quitter via la touche 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Arrêt demandé via la touche 'q'...")
                    break
            else:
                # Mode sans interface graphique (headless) : affichage textuel de statut
                print(
                    f"[LIVE] Présents: {current_count} | "
                    f"Entrées: {tracker.entries} | "
                    f"Sorties: {tracker.exits}", 
                    end="\r"
                )
                
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur (Ctrl+C)...")
        
    finally:
        # Nettoyage et affichage du bilan de fin de session
        cv2.destroyAllWindows()
        print("\n--- STATISTIQUES FINALES ---")
        print(f"Total Entrées : {tracker.entries}")
        print(f"Total Sorties : {tracker.exits}")
        print("Application fermée.")

if __name__ == "__main__":
    main()
