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
from tracker import LineCrossingTracker
from visualizer import Visualizer

# Configuration des chemins
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR.parent / "models" / "yolo11n.pt"
CUSTOM_BOTSORT_CONFIG = ROOT_DIR / "configs" / "custom_botsort.yaml"


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
        default=0.25,
        help="Seuil de confiance YOLO. 0.25 laisse les détections faibles atteindre la seconde passe ByteTrack; ajuster selon les faux positifs.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.55,
        help="Seuil IoU du NMS YOLO. Une valeur modérée conserve les personnes proches sans multiplier les doublons.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Résolution d'inférence. 960 améliore les petites personnes; réduire à 640 si la latence devient prioritaire.",
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
        "--occlusion-iou",
        type=float,
        default=0.35,
        help="IoU entre deux boîtes qui déclenche l’association Re-ID de type DeepSORT.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=120,
        help="Nombre maximal de frames pendant lesquelles une piste Lost reste récupérable (environ 4 s à 30 FPS).",
    )
    parser.add_argument(
        "--occlusion-threshold",
        type=int,
        default=1,
        help="Conservé pour compatibilité; la récupération hybride utilise désormais directement la fenêtre max-age.",
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
        default=0.42,
        help="Seuil de similarité cosinus (0.42) pour favoriser la récupération après occlusion complète.",
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin de sortie pour enregistrer la vidéo annotée (recommandé dans Google Colab).",
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

    # 2. Tracker natif Ultralytics : BoT-SORT garde lui-même les pistes Lost.
    tracker = LineCrossingTracker(line_p1=line_p1, line_p2=line_p2, max_lost_frames=60)
    tracker_config = str(CUSTOM_BOTSORT_CONFIG) if CUSTOM_BOTSORT_CONFIG.exists() else "botsort.yaml"

    # Prétraitement CLAHE optionnel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.enhance_contrast else None

    print(f"Lancement du suivi natif BoT-SORT sur la source : {source}")
    print(f"Paramètres: conf={args.conf:.2f}, NMS-IoU={args.iou:.2f}, imgsz={args.imgsz}, track_buffer=60, mapping_conf>0.50, trails=off")
    print(f"Résolution d'inférence : {args.imgsz}x{args.imgsz}")
    print("Appuyez sur 'q' dans la fenêtre vidéo pour quitter, ou Ctrl+C dans le terminal.")

    window_initialized = False
    video_writer = None

    # IDs d’affichage indépendants des IDs internes ByteTrack/Re-ID.
    # Ils restent compacts et commencent toujours à 1 pour cette session.
    id_mapping = {}
    next_display_id = 1

    try:
        results = model.track(
            source=source,
            tracker=tracker_config,
            persist=True,
            classes=[0],  # 0 correspond à 'person' dans COCO
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            show=False,
            stream=True,
        )

        frame_index = 0
        for result in results:
            frame_index += 1
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

            # --- IDs fournis directement par BoT-SORT natif ---
            xyxy_list = None
            confidences = None
            display_ids = []
            boxes_for_count = result.boxes
            if result.boxes is not None and result.boxes.id is not None and result.boxes.xyxy is not None:
                raw_ids = result.boxes.id.round().int().cpu().tolist()
                xyxy_all = result.boxes.xyxy.cpu().numpy()
                confidences_all = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(raw_ids))

                # Un nouvel ID système n’est mappé que si sa confiance dépasse
                # 0.50. Un ID déjà stabilisé reste affiché même si sa confiance
                # baisse momentanément.
                keep_indices = []
                for index, (track_id, conf) in enumerate(zip(raw_ids, confidences_all)):
                    if track_id not in id_mapping and float(conf) <= 0.50:
                        continue
                    if track_id not in id_mapping:
                        id_mapping[track_id] = next_display_id
                        print(f"[DEBUG] Nouvel ID système {track_id} (Conf: {float(conf):.2f}) -> Mappé à l'utilisateur {next_display_id}")
                        next_display_id += 1
                    keep_indices.append(index)
                    display_ids.append(id_mapping[track_id])

                # Les détections non mappées ne participent ni au comptage ni
                # au dessin; cela évite de créer des IDs pour des faux positifs.
                if keep_indices:
                    boxes_for_count = result.boxes[keep_indices]
                    xyxy_list = xyxy_all[keep_indices]
                    confidences = confidences_all[keep_indices]
                else:
                    boxes_for_count = None

            # Comptage uniquement avec les IDs séquentiels stabilisés.
            current_count, active_ids = tracker.update(
                boxes_for_count, height, width,
                override_track_ids=display_ids if display_ids else None,
            )

            # Visualisation
            frame_draw = frame.copy()
            if xyxy_list is not None:
                Visualizer.draw_boxes(frame_draw, xyxy_list, display_ids, confidences)

            Visualizer.draw_crossing_line(frame_draw, tracker.line_p1, tracker.line_p2)
            Visualizer.draw_hud(frame_draw, current_count, tracker.entries, tracker.exits)

            if args.output and video_writer is None:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fps = 30.0
                if isinstance(source, (str, Path)):
                    capture = cv2.VideoCapture(str(source))
                    detected_fps = capture.get(cv2.CAP_PROP_FPS)
                    capture.release()
                    if detected_fps and detected_fps > 0:
                        fps = detected_fps
                extension = output_path.suffix.lower()
                codec = "mp4v" if extension == ".mp4" else "XVID"
                video_writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Impossible de créer la vidéo de sortie : {output_path}")
                print(f"[Output] Enregistrement de la vidéo annotée : {output_path}")

            if video_writer is not None:
                video_writer.write(frame_draw)

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
        if video_writer is not None:
            video_writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print("\n--- STATISTIQUES FINALES ---")
        print(f"Total Entrées : {tracker.entries}")
        print(f"Total Sorties : {tracker.exits}")
        print("Application fermée.")


if __name__ == "__main__":
    main()
