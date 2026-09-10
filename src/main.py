"""Comptage de personnes YOLO11 + BoT-SORT avec IDs d'affichage séquentiels.

Le tracker reste responsable des pistes; IDManager ne fait que convertir les
IDs internes confirmés en IDs d'affichage 1, 2, 3, ... . Aucun trail n'est
rendu. En mode --no-show, la ligne par défaut ou les points CLI sont utilisés.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "models" / "yolo11n.pt"
DEFAULT_TRACKER = ROOT / "configs" / "custom_botsort.yaml"


class IDManager:
    """Convertit les IDs BoT-SORT confirmés en IDs séquentiels persistants."""

    def __init__(self, confirmation_confidence: float = 0.65) -> None:
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
        current_displays: set[int] = set()
        output = []
        for track_id, box, conf, feature in descriptors:
            display_id = self.mapping_dict.get(int(track_id))
            best_id, best_score = None, -1.0
            if feature is not None:
                for candidate_id, candidate_feature in self.features.items():
                    if candidate_id in current_displays:
                        continue
                    # Ne jamais voler l’ID d’une personne vue récemment : la
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
                print(f"[INFO] Nouvelle personne confirmée : ID {display_id} (track_id={track_id}, conf={conf:.2f})")
            center = ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
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


class LineCounter:
    """Compte chaque display_id au plus une fois par direction."""

    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        self.p1 = p1
        self.p2 = p2
        self.previous: dict[int, tuple[float, float]] = {}
        self.counted_in: set[int] = set()
        self.counted_out: set[int] = set()
        self.entries = 0
        self.exits = 0

    @staticmethod
    def cross(a: tuple[float, float], b: tuple[float, float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    def segments_intersect(
        a: tuple[float, float], b: tuple[float, float],
        c: tuple[float, float], d: tuple[float, float],
    ) -> bool:
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        eps = 1e-9
        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
            return True
        return ((abs(o1) <= eps and on_segment(a, c, b)) or
                (abs(o2) <= eps and on_segment(a, d, b)) or
                (abs(o3) <= eps and on_segment(c, a, d)) or
                (abs(o4) <= eps and on_segment(c, b, d)))

    def update(self, display_id: int, point: tuple[float, float], width: int, height: int) -> None:
        previous = self.previous.get(display_id)
        self.previous[display_id] = point
        if previous is None or not self.segments_intersect(previous, point, self._line(width, height)[0], self._line(width, height)[1]):
            return
        a, b = self._line(width, height)
        line_vector = (b[0] - a[0], b[1] - a[1])
        before = self.cross(line_vector, (previous[0] - a[0], previous[1] - a[1]))
        after = self.cross(line_vector, (point[0] - a[0], point[1] - a[1]))
        if before < 0 < after and display_id not in self.counted_in:
            self.counted_in.add(display_id)
            self.entries += 1
            print(f"[COUNT] ID {display_id} -> IN")
        elif before > 0 > after and display_id not in self.counted_out:
            self.counted_out.add(display_id)
            self.exits += 1
            print(f"[COUNT] ID {display_id} -> OUT")

    def _line(self, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.p1[0] * width, self.p1[1] * height), (self.p2[0] * width, self.p2[1] * height))

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        a, b = self._line(w, h)
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 0), 3)
        cv2.putText(frame, f"IN: {self.entries}  OUT: {self.exits}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


def parse_point(value: str) -> tuple[float, float]:
    x, y = (float(v.strip()) for v in value.split(","))
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Les coordonnées de ligne doivent être normalisées entre 0 et 1")
    return x, y


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

    window = "Calibration - cliquez 2 points puis appuyez sur c"
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
        if key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 + BoT-SORT people counting")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--line-p1", default=None)
    parser.add_argument("--line-p2", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    model_path = Path(args.model)
    model = YOLO(str(model_path))
    line_p1, line_p2 = calibrate(args.source, args.no_show, args.line_p1, args.line_p2)
    manager = IDManager(confirmation_confidence=0.65)
    counter = LineCounter(line_p1, line_p2)
    writer = None
    window_initialized = False

    results = model.track(source=int(args.source) if args.source.isdigit() else args.source, tracker=args.tracker, persist=True, classes=[0], conf=args.conf, iou=args.iou, imgsz=args.imgsz, show=False, stream=True)
    try:
        for result in results:
            frame = result.orig_img
            if frame is None:
                continue
            h, w = frame.shape[:2]
            rendered = frame.copy()
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(ids))
                raw_candidates: list[tuple[int, np.ndarray, float]] = []
                for box, track_id, conf in zip(boxes, ids, confs):
                    raw_candidates.append((track_id, box, float(conf)))

                candidates = manager.assign_frame(
                    raw_candidates,
                    frame,
                    frame_index,
                    float(np.hypot(w, h)),
                )
                for track_id, display_id, point, box, _conf in candidates:
                    x1, y1, x2, y2 = map(int, box[:4])
                    counter.update(display_id, point, w, h)
                    cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 80, 0), 2)
                    cv2.circle(rendered, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                    cv2.putText(rendered, f"ID: {display_id} person", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2, cv2.LINE_AA)
            counter.draw(rendered)

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
            if not args.no_show:
                if not window_initialized:
                    cv2.namedWindow("People counting", cv2.WINDOW_NORMAL)
                    window_initialized = True
                cv2.imshow("People counting", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"\r[LIVE] IN={counter.entries} OUT={counter.exits}", end="")
    finally:
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print(f"\n[FINAL] IN={counter.entries} OUT={counter.exits}")


if __name__ == "__main__":
    main()
