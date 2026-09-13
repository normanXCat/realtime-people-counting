"""Comptage de personnes YOLO11 + BoT-SORT avec gestion d'occupation robuste.

Pipeline :
1. IDManager convertit les track_id BoT-SORT en display_id séquentiels (ReID).
2. OccupancyManager applique la machine à états FSM avec :
   - Intersection vectorielle CCW pour détecter les franchissements
   - Zone morte (hystérésis) autour de la ligne virtuelle
   - Phase de warm-up pour calibrer l'effectif initial
   - Gestion des occultations (disparition ≠ sortie)
3. Le rendu visuel affiche les bounding boxes colorées par état et le HUD d'occupation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from occupancy_manager import OccupancyManager, compute_anchor


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "models" / "yolo11n.pt"
DEFAULT_TRACKER = ROOT / "configs" / "custom_botsort.yaml"


def verify_reid_weights(tracker_path: str) -> None:
    """Vérifie que le poids ReID demandé existe avant l'inférence."""
    config = yaml.safe_load(Path(tracker_path).read_text())
    if not config.get("with_reid", False):
        raise RuntimeError("[REID CHECK] with_reid n'est pas activé dans la configuration")
    model_name = str(config.get("model", "auto"))
    if model_name == "auto":
        print("[REID CHECK] ReID activée en mode auto (poids gérés par Ultralytics)")
        return
    candidates = [Path(model_name), Path(tracker_path).parent / model_name, Path.cwd() / model_name]
    weight = next((path for path in candidates if path.exists()), None)
    if weight is None:
        # Les poids YOLO de classification standards sont téléchargés par
        # YOLO(...) au premier lancement; ils ne doivent pas être bloqués par
        # ce contrôle local.
        standard_ultralytics_weights = {
            "yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt",
            "yolo11l-cls.pt", "yolo11x-cls.pt",
        }
        if model_name in standard_ultralytics_weights:
            print(f"[REID CHECK] Poids ReID Ultralytics absents localement : {model_name}")
            print(f"[REID CHECK] Téléchargement automatique attendu par Ultralytics : {model_name}")
            return
        raise FileNotFoundError(
            f"[REID CHECK] Poids ReID absents : {model_name}. "
            f"Placez ce fichier dans le dépôt ou dans le dossier courant."
        )
    print(f"[REID CHECK] Poids ReID trouvés : {weight.resolve()}")


class IDManager:
    """Convertit les IDs BoT-SORT confirmés en IDs séquentiels persistants."""

    def __init__(self, confirmation_confidence: float = 0.75) -> None:
        self.mapping_dict: dict[int, int] = {}
        self.next_id = 1
        self.confirmation_confidence = confirmation_confidence
        self.last_centers: dict[int, tuple[float, float]] = {}
        self.features: dict[int, np.ndarray] = {}
        self.last_seen: dict[int, int] = {}
        self.gallery_ttl = 300

    @staticmethod
    def appearance(frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """Signature légère de couleur/texture, calculée sur le crop personne."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

    def assign_frame(
        self,
        candidates: list[tuple[int, np.ndarray, float]],
        frame: np.ndarray,
        frame_index: int,
        frame_diagonal: float,
    ) -> list[tuple[int, int, tuple[float, float], np.ndarray, float]]:
        """Attribue les display IDs par apparence, avec mémoire des absents."""
        descriptors = [(tid, box, float(conf), self.appearance(frame, box)) for tid, box, conf in candidates]

        # Détecter un échange de deux pistes connues : si les apparences
        # correspondent nettement mieux en croisé qu'en direct, corriger le
        # mapping avant le rendu et le comptage.
        known = [
            (tid, int(self.mapping_dict[tid]), feature)
            for tid, _box, _conf, feature in descriptors
            if tid in self.mapping_dict and feature is not None and self.mapping_dict[tid] in self.features
        ]
        if len(known) >= 2:
            for i in range(len(known)):
                tid_a, did_a, feat_a = known[i]
                for j in range(i + 1, len(known)):
                    tid_b, did_b, feat_b = known[j]
                    direct = self.similarity(feat_a, self.features[did_a]) + self.similarity(feat_b, self.features[did_b])
                    crossed = self.similarity(feat_a, self.features[did_b]) + self.similarity(feat_b, self.features[did_a])
                    if crossed > direct + 0.12:
                        self.mapping_dict[tid_a], self.mapping_dict[tid_b] = did_b, did_a
                        print(f"[RE-ID-LOCK] Switch corrigé : track_id {tid_a}/{tid_b} -> IDs {did_b}/{did_a}")
        current_displays: set[int] = set()
        output = []
        for track_id, box, conf, feature in descriptors:
            display_id = self.mapping_dict.get(int(track_id))
            best_id, best_score = None, -1.0
            # Un track_id déjà connu est verrouillé sur son display_id. La
            # ReID ne doit jamais le remapper vers une autre personne active;
            # elle ne sert qu'à récupérer l'ID d'un nouveau track_id après
            # disparition de la piste précédente.
            if display_id is None and feature is not None:
                for candidate_id, candidate_feature in self.features.items():
                    if candidate_id in current_displays:
                        continue
                    # Ne jamais voler l'ID d'une personne vue récemment : la
                    # galerie sert aux pistes réellement disparues.
                    if frame_index - self.last_seen.get(candidate_id, frame_index) < 3:
                        continue
                    score = self.similarity(feature, candidate_feature)
                    if score > best_score:
                        best_id, best_score = candidate_id, score
            # Une nouvelle piste peut récupérer un ID ancien seulement avec
            # une vraie ressemblance; sinon elle reçoit un nouvel ID confirmé.
            if best_id is not None and best_score >= 0.78:
                if display_id is None or best_id != display_id:
                    display_id = best_id
                    self.mapping_dict[int(track_id)] = display_id
                    print(f"[RE-ID] track_id={track_id} récupère l'ID mémorisé {display_id} (similarité={best_score:.2f})")
            if display_id is None:
                if conf <= self.confirmation_confidence:
                    continue
                display_id = self.next_id
                self.next_id += 1
                self.mapping_dict[int(track_id)] = display_id
                print(f"[REID SUCCESS] Nouvelle entité confirmée : YOLO_ID {track_id} -> Affiche ID {display_id} (conf={conf:.2f})")
            center = compute_anchor(box)
            if feature is not None:
                old = self.features.get(display_id)
                self.features[display_id] = feature if old is None else (0.85 * old + 0.15 * feature)
                norm = np.linalg.norm(self.features[display_id])
                if norm > 1e-8:
                    self.features[display_id] /= norm
            self.last_centers[display_id] = center
            self.last_seen[display_id] = frame_index
            current_displays.add(display_id)
            output.append((int(track_id), display_id, center, box, conf))
        # Conserver la mémoire, mais purger les identités très anciennes.
        expired = [did for did, seen in self.last_seen.items() if frame_index - seen > self.gallery_ttl]
        for did in expired:
            self.features.pop(did, None)
            self.last_centers.pop(did, None)
            self.last_seen.pop(did, None)
        return output

    def stabilize_assignments(
        self,
        assignments: list[tuple[int, int, tuple[float, float]]],
        frame_diagonal: float,
    ) -> list[tuple[int, int, tuple[float, float]]]:
        """Corrige un échange évident entre deux pistes déjà connues.

        BoT-SORT peut conserver deux track_id tout en les inversant après un
        croisement. Lorsque l'affectation croisée est nettement plus proche
        des dernières positions connues, on échange uniquement les display_id
        pour cette frame et on met à jour le mapping interne.
        """
        if len(assignments) < 2:
            return assignments
        result = list(assignments)
        max_switch_cost = frame_diagonal * 0.35
        for i in range(len(result)):
            track_a, display_a, center_a = result[i]
            prev_a = self.last_centers.get(display_a)
            if prev_a is None:
                continue
            for j in range(i + 1, len(result)):
                track_b, display_b, center_b = result[j]
                prev_b = self.last_centers.get(display_b)
                if prev_b is None or display_a == display_b:
                    continue
                direct = self._distance(center_a, prev_a) + self._distance(center_b, prev_b)
                crossed = self._distance(center_a, prev_b) + self._distance(center_b, prev_a)
                if crossed + max_switch_cost < direct:
                    result[i] = (track_a, display_b, center_a)
                    result[j] = (track_b, display_a, center_b)
                    self.mapping_dict[track_a] = display_b
                    self.mapping_dict[track_b] = display_a
                    print(f"[ID-LOCK] Échange corrigé entre IDs {display_a} et {display_b}")
        for _track_id, display_id, center in result:
            self.last_centers[display_id] = center
        return result

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def get_display_id(self, track_id: int, confidence: float) -> Optional[int]:
        track_id = int(track_id)
        if track_id not in self.mapping_dict:
            if float(confidence) <= self.confirmation_confidence:
                return None
            display_id = self.next_id
            self.mapping_dict[track_id] = display_id
            self.next_id += 1
            print(
                f"[INFO] Nouvelle personne confirmée : ID {display_id} "
                f"(track_id={track_id}, conf={float(confidence):.2f})"
            )
        return self.mapping_dict[track_id]


def parse_point(value: str) -> tuple[float, float]:
    x, y = (float(v.strip()) for v in value.split(","))
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Les coordonnées de ligne doivent être normalisées entre 0 et 1")
    return x, y


def ask_counting_mode() -> bool:
    """Demande à l'utilisateur si les passages doivent être comptabilisés."""
    while True:
        answer = input(
            "Souhaitez-vous compter les personnes qui entrent et sortent ? (o/n) : "
        ).strip().lower()
        if answer in {"o", "oui", "y", "yes"}:
            return True
        if answer in {"n", "non", "no"}:
            print("Mode suivi uniquement activé : aucun comptage de passages.")
            return False
        print("Réponse invalide. Répondez par o pour oui ou n pour non.")


def calibrate(source: str, no_show: bool, p1: Optional[str], p2: Optional[str]) -> tuple[tuple[float, float], tuple[float, float]]:
    if p1 and p2:
        return parse_point(p1), parse_point(p2)
    if no_show:
        return (0.0, 0.6), (1.0, 0.6)
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Impossible de lire la première frame : {source}")
    points: list[tuple[int, int]] = []
    canvas = frame.copy()

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    window = "Calibration - cliquez 2 points, c=valider, g=reinitialiser"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        display = canvas.copy()
        for point in points:
            cv2.circle(display, point, 6, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 0), 3)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("c") and len(points) == 2:
            cv2.destroyWindow(window)
            h, w = frame.shape[:2]
            return (points[0][0] / w, points[0][1] / h), (points[1][0] / w, points[1][1] / h)
        if key == ord("g"):
            points.clear()
            print("[Calibration] Ligne réinitialisée. Cliquez à nouveau sur deux points.")
        if key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 + BoT-SORT people counting (FSM)")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--line-p1", default=None)
    parser.add_argument("--line-p2", default=None)
    parser.add_argument("--line2-p1", default=None, help="Première extrémité de la ligne extérieure du sas")
    parser.add_argument("--line2-p2", default=None, help="Deuxième extrémité de la ligne extérieure du sas")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-show", action="store_true")
    # Paramètres OccupancyManager
    parser.add_argument("--dead-zone", type=float, default=30.0,
                        help="Épaisseur de la zone morte en pixels (hystérésis)")
    parser.add_argument("--warmup-frames", type=int, default=90,
                        help="Nombre de frames de warm-up (~3s à 30fps)")
    parser.add_argument("--confirm-frames", type=int, default=15,
                        help="Frames consécutives pour confirmer une nouvelle présence")
    parser.add_argument("--grace-frames", type=int, default=300,
                        help="Frames de délai de grâce pour les pistes occultées")
    parser.add_argument("--zenithal", action="store_true",
                        help="Vue zénithale : utiliser le centre de boîte au lieu du bas")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    model_path = Path(args.model)
    verify_reid_weights(args.tracker)
    model = YOLO(str(model_path))
    counting_enabled = ask_counting_mode()

    if counting_enabled:
        line_p1, line_p2 = calibrate(args.source, args.no_show, args.line_p1, args.line_p2)
        line2_p1 = parse_point(args.line2_p1) if args.line2_p1 else None
        line2_p2 = parse_point(args.line2_p2) if args.line2_p2 else None
        if (line2_p1 is None) != (line2_p2 is None):
            raise ValueError("--line2-p1 et --line2-p2 doivent être fournis ensemble")
    else:
        line_p1, line_p2 = (0.0, 0.0), (0.0, 0.0)
        line2_p1 = line2_p2 = None

    id_manager = IDManager(confirmation_confidence=0.75)
    occupancy = OccupancyManager(
        line_p1=line_p1,
        line_p2=line_p2,
        line2_p1=line2_p1,
        line2_p2=line2_p2,
        dead_zone_margin=args.dead_zone,
        init_duration_frames=args.warmup_frames,
        confirmation_threshold=args.confirm_frames,
        grace_period_frames=args.grace_frames,
        is_zenithal=args.zenithal,
    )

    writer = None
    window_initialized = False
    frame_index = 0
    pred_events: list[dict] = []

    results = model.track(
        source=int(args.source) if args.source.isdigit() else args.source,
        tracker=args.tracker,
        persist=True,
        classes=[0],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=False,
        stream=True,
    )

    try:
        for result in results:
            t_start = time.perf_counter()
            frame_index += 1
            frame = result.orig_img
            if frame is None:
                continue
            h, w = frame.shape[:2]
            rendered = frame.copy()
            visible_count = 0

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(ids))
                raw_candidates: list[tuple[int, np.ndarray, float]] = []
                for box, track_id, conf in zip(boxes, ids, confs):
                    raw_candidates.append((track_id, box, float(conf)))

                # IDManager : attribution des display_id séquentiels
                candidates = id_manager.assign_frame(
                    raw_candidates,
                    frame,
                    frame_index,
                    float(np.hypot(w, h)),
                )
                visible_count = len(candidates)

                if counting_enabled:
                    # OccupancyManager : FSM + comptage
                    events = occupancy.process_frame(
                        candidates=candidates,
                        frame=frame,
                        frame_index=frame_index,
                        features=id_manager.features,
                    )
                    for evt in events:
                        evt["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                        pred_events.append(evt)

                    # Rendu avec bounding boxes colorées par état + HUD
                    occupancy.draw_overlay(rendered, candidates, visible_count)
                    line2_px = occupancy.line2_px(w, h)
                    if line2_px is not None:
                        cv2.line(rendered, tuple(map(int, line2_px[0])), tuple(map(int, line2_px[1])), (255, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(rendered, "L2: EXTERIEUR", tuple(map(int, line2_px[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    # Mode suivi uniquement : bounding boxes simples
                    for track_id, display_id, point, box, conf in candidates:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 80, 0), 2)
                        cv2.circle(rendered, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                        label = f"ID: {display_id} person {conf:.2f}"
                        cv2.putText(rendered, label, (x1, max(25, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2, cv2.LINE_AA)
                    # HUD minimal
                    cv2.putText(rendered, f"Present: {visible_count}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if counting_enabled:
                    # Aucune détection : gérer les occultations
                    occupancy.process_frame([], frame, frame_index)
                    occupancy.draw_overlay(rendered, [], 0)

            # -- Écriture vidéo --
            if args.output and writer is None:
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                capture = cv2.VideoCapture(args.source)
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                capture.release()
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Impossible de créer {out}")
                print(f"[OUTPUT] {out}")
            if writer is not None:
                writer.write(rendered)

            # -- Affichage --
            if not args.no_show:
                if not window_initialized:
                    cv2.namedWindow("People counting", cv2.WINDOW_NORMAL)
                    window_initialized = True
                cv2.imshow("People counting", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if counting_enabled:
                    print(
                        f"\r[LIVE] OCCUPANCY={occupancy.occupancy_count} "
                        f"IN={occupancy.total_in} OUT={occupancy.total_out} "
                        f"NEW={occupancy.total_new_presences} "
                        f"VISIBLE={visible_count} PHASE={occupancy.phase}",
                        end="",
                    )
                else:
                    print(f"\r[LIVE] PRESENT={visible_count}", end="")

    finally:
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        if counting_enabled:
            print(
                f"\n[FINAL] OCCUPANCY={occupancy.occupancy_count} "
                f"IN={occupancy.total_in} OUT={occupancy.total_out} "
                f"NEW={occupancy.total_new_presences}"
            )
        else:
            print(f"\n[FINAL] Session terminée.")

    if pred_events:
        with open("predictions_systeme.json", "w", encoding="utf-8") as f:
            json.dump(pred_events, f, indent=4)


if __name__ == "__main__":
    main()
