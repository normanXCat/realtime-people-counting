"""
Script principal de détection, suivi (tracking) et comptage de personnes.

Ce script initialise la capture d'images (webcam, fichier vidéo ou flux réseau),
déclenche la calibration manuelle (2 points) si nécessaire, instancie le modèle YOLO (YOLO11),
exécute le tracking hybride (ByteTrack + réassociation Re-ID par apparence en cas d'occlusion prolongée),
et applique des overlays graphiques (ligne de franchissement virtuelle, trainées de trajectoires, HUD de statistiques).
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Importations des modules locaux
from calibration import calibrate_line
from reid import ReIDGallery
from tracker import LineCrossingTracker
from visualizer import Visualizer

# Configuration des chemins
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "yolo11n.pt"
CUSTOM_BYTETRACK_CONFIG = ROOT_DIR / "configs" / "bytetrack_custom.yaml"


def parse_args() -> argparse.Namespace:
    """
    Configure et analyse les arguments passés en ligne de commande.

    Returns:
        argparse.Namespace: Objet contenant les arguments CLI configurés.
    """
    parser = argparse.ArgumentParser(
        description="Suivi et comptage de personnes en temps réel avec YOLO11, ByteTrack & Re-ID."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source vidéo : index de caméra (ex: '0') ou chemin d'un fichier vidéo.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Chemin vers le modèle YOLO (.pt). Exemple : yolo11n.pt (rapide), yolo11s.pt, yolo11m.pt (plus précis).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Seuil de confiance de détection. 0.5 permet d'éliminer les ombres et faux positifs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Résolution d'inférence (640 par défaut, 960 ou 1280 pour détecter les personnes éloignées/petites).",
    )
    parser.add_argument(
        "--line-p1",
        type=str,
        default=None,
        help="Coordonnées (x,y) normalisées [0.0-1.0] du point 1 de la ligne (ex: '0.1,0.5'). Saute la calibration manuelle si fourni avec --line-p2.",
    )
    parser.add_argument(
        "--line-p2",
        type=str,
        default=None,
        help="Coordonnées (x,y) normalisées [0.0-1.0] du point 2 de la ligne (ex: '0.9,0.5'). Saute la calibration manuelle si fourni avec --line-p1.",
    )
    parser.add_argument(
        "--occlusion-threshold",
        type=int,
        default=1,
        help="Nombre de frames d'absence avant qu'un ID soit archivé en galerie Re-ID (1 pour archivage immédiat).",
    )
    parser.add_argument(
        "--gallery-ttl",
        type=int,
        default=180,
        help="Durée de vie maximale (en frames) d'une identité dans la galerie Re-ID (180 = ~6 sec).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.45,
        help="Seuil de similarité cosinus (0.45 par défaut) pour remapper un nouvel ID vers un ancien ID perdu.",
    )
    parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help="Appliquer un prétraitement CLAHE pour améliorer le contraste sur les scènes sombres ou en contre-jour.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Largeur maximale de la fenêtre d'affichage en pixels (1280 par défaut).",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=720,
        help="Hauteur maximale de la fenêtre d'affichage en pixels (720 par défaut).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Désactiver l'affichage de la fenêtre OpenCV (mode headless).",
    )
    return parser.parse_args()


def parse_point_arg(point_str: str) -> tuple[float, float]:
    """Convertit une chaîne 'x,y' en tuple de floats (x, y)."""
    try:
        parts = [float(p.strip()) for p in point_str.split(",")]
        if len(parts) != 2:
            raise ValueError
        return (parts[0], parts[1])
    except Exception:
        raise ValueError(
            f"Format de point invalide : '{point_str}'. Format attendu : 'x,y' (ex: '0.1,0.5')"
        )


def main() -> None:
    """Point d'entrée principal du programme."""
    args = parse_args()

    # Validation et téléchargement automatique du modèle YOLO
    model_path = Path(args.model)
    if not model_path.exists():
        standard_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        if model_path.name in standard_models:
            print(f"Modèle {model_path.name} non trouvé localement. Téléchargement en cours depuis Ultralytics...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_path.name}"
            try:
                import urllib.request
                opener = urllib.request.build_opener()
                opener.addheaders = [("User-agent", "Mozilla/5.0")]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, model_path)
                print(f"Modèle téléchargé avec succès : {model_path}")
            except Exception as e:
                print(f"Erreur lors du téléchargement automatique : {e}")
                sys.exit(1)
        else:
            print(f"Modèle non trouvé à l'emplacement : {model_path}")
            sys.exit(1)

    model = YOLO(str(model_path))

    # Résolution de la source vidéo
    source = args.source
    if source.isdigit():
        source = int(source)

    # 1. Calibration de la ligne virtuelle (clic 2 points)
    if args.line_p1 is not None and args.line_p2 is not None:
        line_p1 = parse_point_arg(args.line_p1)
        line_p2 = parse_point_arg(args.line_p2)
        print(f"[Calibration] Coordonnées fournies en CLI : P1={line_p1}, P2={line_p2}")
    else:
        if not args.no_show:
            print("[Calibration] Lancement de la calibration manuelle (2 points)...")
            line_p1, line_p2 = calibrate_line(source)
        else:
            print("[Calibration] Mode Headless : Utilisation de la ligne par défaut (60% hauteur).")
            line_p1, line_p2 = (0.0, 0.6), (1.0, 0.6)

    # 2. Initialisation du tracker et de la galerie Re-ID
    tracker = LineCrossingTracker(line_p1=line_p1, line_p2=line_p2)
    reid_gallery = ReIDGallery(
        occlusion_threshold=args.occlusion_threshold,
        gallery_ttl=args.gallery_ttl,
        similarity_threshold=args.similarity_threshold,
    )

    # Utilisation du fichier de config ByteTrack personnalisé si disponible
    tracker_config = "bytetrack.yaml"
    if CUSTOM_BYTETRACK_CONFIG.exists():
        tracker_config = str(CUSTOM_BYTETRACK_CONFIG)

    # Prétraitement CLAHE optionnel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.enhance_contrast else None

    print(f"Lancement du suivi (ByteTrack + Re-ID) sur la source : {source}")
    print(f"Résolution d'inférence : {args.imgsz}x{args.imgsz}")
    print("Appuyez sur 'q' dans la fenêtre vidéo pour quitter, ou Ctrl+C dans le terminal.")

    window_initialized = False

    try:
        results = model.track(
            source=source,
            tracker=tracker_config,
            persist=True,
            classes=[0],  # 0 correspond à 'person' dans COCO
            conf=args.conf,
            imgsz=args.imgsz,
            show=False,
            stream=True,
        )

        for result in results:
            frame = result.orig_img
            if frame is None:
                continue

            height, width = frame.shape[:2]

            # Prétraitement optionnel du contraste (CLAHE)
            if clahe is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # --- Couche de réassociation Re-ID par apparence ---
            remapped_ids = None
            xyxy_list = None
            confidences = None
            if result.boxes is not None and result.boxes.id is not None and result.boxes.xyxy is not None:
                raw_ids = result.boxes.id.round().int().cpu().tolist()
                xyxy_list = result.boxes.xyxy.cpu().numpy()
                if result.boxes.conf is not None:
                    confidences = result.boxes.conf.cpu().numpy()

                # Remapping d'IDs si réapparition après occlusion
                remapped_ids = reid_gallery.remap_ids(raw_ids, xyxy_list, frame)

            # Mise à jour du tracker et calcul du comptage
            current_count, active_ids = tracker.update(
                result.boxes, height, width, override_track_ids=remapped_ids
            )

            # Visualisation
            frame_draw = frame.copy()
            if xyxy_list is not None:
                display_ids = remapped_ids if remapped_ids is not None else raw_ids
                Visualizer.draw_boxes(frame_draw, xyxy_list, display_ids, confidences)

            Visualizer.draw_crossing_line(frame_draw, tracker.line_p1, tracker.line_p2)
            Visualizer.draw_trails(frame_draw, tracker.track_history)
            Visualizer.draw_hud(frame_draw, current_count, tracker.entries, tracker.exits)

            if not args.no_show:
                window_name = "Real-time People Counting & Tracking"
                if not window_initialized:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    scale = min(args.display_width / width, args.display_height / height, 1.0)
                    disp_w, disp_h = int(width * scale), int(height * scale)
                    cv2.resizeWindow(window_name, disp_w, disp_h)
                    window_initialized = True

                cv2.imshow(window_name, frame_draw)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Arrêt demandé via la touche 'q'...")
                    break
            else:
                print(
                    f"[LIVE] Présents: {current_count} | Entrées: {tracker.entries} | Sorties: {tracker.exits}",
                    end="\r",
                )

    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur (Ctrl+C)...")

    finally:
        cv2.destroyAllWindows()
        print("\n--- STATISTIQUES FINALES ---")
        print(f"Total Entrées : {tracker.entries}")
        print(f"Total Sorties : {tracker.exits}")
        print("Application fermée.")


if __name__ == "__main__":
    main()
