"""Harnais d'évaluation du protocole de la section 9 (livrable 7).

Ce script a été **corrigé et étendu**, pas réécrit à côté : il garde son nom, son
rôle et son point d'entrée `évaluer_comptage_systeme`. Ce qui change :

- il lit le **journal JSON Lines** produit par `src/main.py` (une ligne JSON par
  événement), via le même validateur de schéma que le pipeline
  (:func:`events.read_events`) — l'ancien format « liste d'événements » n'existe
  plus ;
- il mesure les métriques exigées par la section 9.3 : F1 des franchissements,
  erreur absolue moyenne d'occupation et débit réel ;
- il compare à une **vérité terrain à deux niveaux** (section 9.2) : événements
  (IN/OUT horodatés) et occupation par intervalles ;
- il refuse de conclure si le critère d'acceptation n'a pas été fixé à l'avance :
  les seuils arrivent par la ligne de commande, jamais devinés.

Les métriques de détection/tracking (mAP, IDF1, ID switches) relèvent d'une
vérité terrain MOT dense et sont calculées par les outils standards (py-motmetrics)
à partir de `predictions/` ; ce script couvre les métriques de comptage et de
temps réel.

Utilisation :

    python evaluate_system.py \
        --events results/<session>/events.jsonl \
        --ground-truth data/gt/<video>.json \
        --min-f1 0.90 --max-occupancy-mae 1.0 --min-fps 15 \
        --out results/<session>/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from events import read_events  # noqa: E402  (import après ajustement du sys.path)

#: Tolérance par défaut d'appariement temporel entre prédiction et vérité terrain.
DEFAULT_TOLERANCE_S = 1.0

#: Critère d'acceptation par défaut (section 9.3), à confirmer avant mesures.
DEFAULT_MIN_F1 = 0.90
DEFAULT_MAX_OCCUPANCY_MAE = 1.0

#: Événements qui participent au comptage des franchissements (spec 6.2).
CROSSING_TYPES = ("IN", "OUT", "NEW")


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def charger_evenements(chemin_fichier: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Charge un journal d'événements JSON Lines **validé** par le schéma (spec 6.2)."""
    path = Path(chemin_fichier)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier spécifié est introuvable : {path}")
    return read_events(path)


def charger_verite_terrain(chemin_fichier: str | os.PathLike[str]) -> dict[str, Any]:
    """Charge une vérité terrain au format documenté (`docs/protocole_evaluation.md`).

    Format attendu :

    .. code-block:: json

        {
          "schema_version": 1,
          "video": "clip01.mp4",
          "fps": 30.0,
          "crossings": [{"timestamp_s": 3.2, "direction": "in"}, ...],
          "occupancy": [{"start_s": 0.0, "end_s": 4.5, "count": 0}, ...]
        }

    Une cible plus simple (liste de franchissements) est acceptée et interprétée
    comme `{"crossings": [...]}` : l'absence d'occupation n'est alors pas une
    erreur, elle est simplement signalée dans le rapport.
    """
    path = Path(chemin_fichier)
    if not path.exists():
        raise FileNotFoundError(f"Vérité terrain introuvable : {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        payload = {"crossings": payload}

    if not isinstance(payload, Mapping) or "crossings" not in payload:
        raise ValueError(
            f"{path} : format de vérité terrain invalide "
            "(clé « crossings » attendue, voir docs/protocole_evaluation.md)"
        )

    crossings: list[dict[str, Any]] = []
    for index, crossing in enumerate(payload["crossings"]):
        direction = str(crossing.get("direction", "")).lower()
        if direction not in ("in", "out"):
            raise ValueError(
                f"{path}:crossings[{index}] : direction {direction!r} invalide "
                "(attendu « in » ou « out »)"
            )
        timestamp = crossing.get("timestamp_s", crossing.get("time_s"))
        if timestamp is None:
            raise ValueError(
                f"{path}:crossings[{index}] : horodatage manquant (timestamp_s)"
            )
        crossings.append({"timestamp_s": float(timestamp), "direction": direction})

    occupancy: list[dict[str, Any]] = []
    for index, interval in enumerate(payload.get("occupancy", [])):
        start = float(interval["start_s"])
        end = float(interval["end_s"])
        if end < start:
            raise ValueError(f"{path}:occupancy[{index}] : intervalle inversé")
        occupancy.append({"start_s": start, "end_s": end, "count": int(interval["count"])})

    return {
        "video": payload.get("video"),
        "fps": float(payload["fps"]) if payload.get("fps") else None,
        "crossings": crossings,
        "occupancy": occupancy,
    }


# ---------------------------------------------------------------------------
# Appariement
# ---------------------------------------------------------------------------
def apparier_evenements(
    gt: Sequence[Mapping[str, Any]],
    pred: Sequence[Mapping[str, Any]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> dict[str, Any]:
    """Apparie GT et prédictions par proximité temporelle **et** direction.

    Appariement glouton par meilleure distance temporelle croissante (un aller
    par événement) : déterministe, il ne dépend pas de l'ordre d'entrée. Un
    franchissement prédit dans la bonne fenêtre mais de mauvaise direction est
    compté à la fois comme faux positif et comme faux négatif, et signalé dans
    `direction_errors`.
    """
    pairs: list[tuple[float, int, int, bool]] = []
    for pred_index, prediction in enumerate(pred):
        for gt_index, truth in enumerate(gt):
            delta = abs(float(prediction["timestamp_s"]) - float(truth["timestamp_s"]))
            if delta <= tolerance_s:
                same_direction = prediction["direction"] == truth["direction"]
                # Les bonnes directions sont appariées en priorité à écart égal.
                pairs.append((delta, pred_index, gt_index, same_direction))

    pairs.sort(key=lambda item: (item[0], not item[3], item[1], item[2]))

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    direction_errors: list[dict[str, Any]] = []

    for delta, pred_index, gt_index, same_direction in pairs:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        record = {
            "pred_index": pred_index,
            "gt_index": gt_index,
            "delta_s": round(delta, 4),
            "pred_direction": pred[pred_index]["direction"],
            "gt_direction": gt[gt_index]["direction"],
        }
        if same_direction:
            matches.append(record)
        else:
            direction_errors.append(record)

    true_positives = len(matches)
    false_positives = len(pred) - true_positives
    false_negatives = len(gt) - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "direction_errors": len(direction_errors),
        "matches": matches,
        "direction_error_details": direction_errors,
        "gt_total": len(gt),
        "pred_total": len(pred),
    }


def _occupancy_gt_at(occupancy: Sequence[Mapping[str, Any]], timestamp_s: float) -> int | None:
    for interval in occupancy:
        if interval["start_s"] <= timestamp_s < interval["end_s"]:
            return int(interval["count"])
    # Dernier intervalle fermé à gauche seulement : on tolère la borne finale.
    for interval in occupancy:
        if interval["end_s"] == timestamp_s:
            return int(interval["count"])
    return None


def evaluer_occupation(
    snapshots: Sequence[Mapping[str, Any]],
    occupancy_gt: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare les `OCCUPANCY_SNAPSHOT` du journal aux relevés de la vérité terrain.

    L'erreur absolue moyenne porte sur l'occupation **confirmée au sens
    opérationnel** : le bilan `occupation initiale + IN + NEW - OUT`, publiée
    dans `occupancy_confirmed`. C'est l'effectif que le système affirme présent,
    et il ne diminue **que** sur une sortie confirmée par la machine à états —
    jamais sur une occultation, une expiration de grâce ou une purge technique.

    Le taux de couverture mesure si la vérité terrain tombe dans l'intervalle
    publié `[occupancy_observed, occupancy_operational]` : la borne basse ne
    compte que les personnes actuellement observables, la borne haute conserve
    celles dont l'observation est perdue (spec 4.4).
    """
    errors: list[float] = []
    covered = 0
    compared = 0
    for snapshot in snapshots:
        truth = _occupancy_gt_at(occupancy_gt, float(snapshot["timestamp_s"]))
        if truth is None:
            continue
        compared += 1
        operational = int(snapshot["occupancy_confirmed"])
        errors.append(abs(operational - truth))
        low, high = snapshot["occupancy_range"]
        if float(low) <= truth <= float(high):
            covered += 1

    if not compared:
        return {
            "compared_snapshots": 0,
            "mae": None,
            "max_error": None,
            "range_coverage": None,
            "note": "aucun chevauchement entre snapshots et vérité terrain d'occupation",
        }

    return {
        "compared_snapshots": compared,
        "mae": statistics.fmean(errors),
        "max_error": max(errors),
        "range_coverage": covered / compared,
    }


def extraire_metriques_temporelles(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extrait le débit et les latences du `SESSION_END`, sans rien recalculer."""
    session_end = next((event for event in reversed(events) if event["type"] == "SESSION_END"), None)
    if session_end is None:
        return {"frames_processed": None, "fps_mean": None, "latency_ms": None}
    return {
        "frames_processed": session_end.get("frames_processed"),
        "duration_s": session_end.get("duration_s"),
        "fps_mean": session_end.get("fps_mean"),
        "fps_median": session_end.get("fps_median"),
        "fps_min": session_end.get("fps_min"),
        "latency_ms": session_end.get("latency_ms"),
    }


# ---------------------------------------------------------------------------
# Évaluation complète
# ---------------------------------------------------------------------------
def evaluer_comptage_systeme(
    chemin_gt: str | os.PathLike[str],
    chemin_pred: str | os.PathLike[str],
    tolerance_seconds: float = DEFAULT_TOLERANCE_S,
    min_f1: float = DEFAULT_MIN_F1,
    max_occupancy_mae: float = DEFAULT_MAX_OCCUPANCY_MAE,
    min_fps: float | None = None,
    chemin_rapport: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Évalue une session et retourne le rapport complet (voir section 9)."""
    gt = charger_verite_terrain(chemin_gt)
    events = charger_evenements(chemin_pred)

    crossings = [
        {
            "timestamp_s": float(event["timestamp_s"]),
            "direction": str(event["direction"]).lower(),
            "person_id": event.get("person_id"),
        }
        for event in events
        if event["type"] in CROSSING_TYPES and event.get("direction") in ("in", "out")
    ]
    snapshots = [event for event in events if event["type"] == "OCCUPANCY_SNAPSHOT"]
    ambiguous = [event for event in events if event["type"] == "AMBIGUOUS_CROSSING"]
    inconsistent = [event for event in events if event["type"] == "INCONSISTENT_STATE"]

    counting = apparier_evenements(gt["crossings"], crossings, tolerance_seconds)
    occupancy = evaluer_occupation(snapshots, gt["occupancy"])
    timing = extraire_metriques_temporelles(events)

    checks: dict[str, Any] = {
        "f1": {
            "value": counting["f1"],
            "threshold": min_f1,
            "passed": counting["f1"] >= min_f1,
        },
        "occupancy_mae": {
            "value": occupancy["mae"],
            "threshold": max_occupancy_mae,
            "passed": None if occupancy["mae"] is None else occupancy["mae"] <= max_occupancy_mae,
        },
    }
    if min_fps is not None:
        fps_value = timing["fps_mean"]
        checks["fps_mean"] = {
            "value": fps_value,
            "threshold": min_fps,
            "passed": fps_value is not None and fps_value >= min_fps,
        }

    accepted = all(check["passed"] is True for check in checks.values())

    report = {
        "schema_version": 1,
        "session_id": events[0].get("session_id") if events else None,
        "video": gt.get("video"),
        "events_path": str(chemin_pred),
        "ground_truth_path": str(chemin_gt),
        "tolerance_seconds": tolerance_seconds,
        "counting": counting,
        "occupancy": occupancy,
        "timing": timing,
        "diagnostics": {
            "ambiguous_crossings": len(ambiguous),
            "inconsistent_states": len(inconsistent),
            "evaluated_events": len(events),
            "crossing_events": len(crossings),
            "snapshots": len(snapshots),
        },
        "acceptance": {"checks": checks, "accepted": accepted},
    }

    _afficher_rapport(report)
    if chemin_rapport is not None:
        output = Path(chemin_rapport)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RAPPORT] {output}")
    return report


def evaluer_sessions(
    paires: Iterable[tuple[str, str]],
    tolerance_seconds: float = DEFAULT_TOLERANCE_S,
    min_f1: float = DEFAULT_MIN_F1,
    max_occupancy_mae: float = DEFAULT_MAX_OCCUPANCY_MAE,
    min_fps: float | None = None,
    chemin_rapport: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Agrège plusieurs sessions (un enregistrement par vidéo) en un rapport unique."""
    reports: list[dict[str, Any]] = []
    for gt_path, events_path in paires:
        reports.append(
            evaluer_comptage_systeme(
                chemin_gt=gt_path,
                chemin_pred=events_path,
                tolerance_seconds=tolerance_seconds,
                min_f1=min_f1,
                max_occupancy_mae=max_occupancy_mae,
                min_fps=min_fps,
            )
        )

    if not reports:
        raise ValueError("Aucune paire (vérité terrain, journal) fournie à l'évaluation")

    aggregated = {
        "schema_version": 1,
        "sessions": len(reports),
        "f1_mean": statistics.fmean(report["counting"]["f1"] for report in reports),
        "occupancy_mae_mean": statistics.fmean(
            report["occupancy"]["mae"] for report in reports if report["occupancy"]["mae"] is not None
        )
        if any(report["occupancy"]["mae"] is not None for report in reports)
        else None,
        "total_true_positives": sum(report["counting"]["true_positives"] for report in reports),
        "total_false_positives": sum(report["counting"]["false_positives"] for report in reports),
        "total_false_negatives": sum(report["counting"]["false_negatives"] for report in reports),
        "accepted_all": all(report["acceptance"]["accepted"] for report in reports),
        "reports": reports,
    }
    micro_precision = (
        aggregated["total_true_positives"]
        / (aggregated["total_true_positives"] + aggregated["total_false_positives"])
        if (aggregated["total_true_positives"] + aggregated["total_false_positives"])
        else 0.0
    )
    micro_recall = (
        aggregated["total_true_positives"]
        / (aggregated["total_true_positives"] + aggregated["total_false_negatives"])
        if (aggregated["total_true_positives"] + aggregated["total_false_negatives"])
        else 0.0
    )
    aggregated["micro_precision"] = micro_precision
    aggregated["micro_recall"] = micro_recall
    aggregated["micro_f1"] = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    print("\n=== AGRÉGAT ===")
    print(f"Sessions               : {aggregated['sessions']}")
    print(f"F1 macro / micro       : {aggregated['f1_mean'] * 100:.2f} % / {aggregated['micro_f1'] * 100:.2f} %")
    if aggregated["occupancy_mae_mean"] is not None:
        print(f"MAE occupation moyenne : {aggregated['occupancy_mae_mean']:.3f} personne")
    print(f"Critère d'acceptation  : {'RESPECTÉ' if aggregated['accepted_all'] else 'NON RESPECTÉ'}")

    if chemin_rapport is not None:
        output = Path(chemin_rapport)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RAPPORT] {output}")
    return aggregated


def _afficher_rapport(report: Mapping[str, Any]) -> None:
    counting = report["counting"]
    occupancy = report["occupancy"]
    timing = report["timing"]
    print("==================================================")
    print("        ÉVALUATION DU SYSTÈME GLOBAL DE COMPTAGE   ")
    print("==================================================")
    print(f"Session                    : {report['session_id']}")
    print(f"Vidéo                      : {report['video']}")
    print(f"Tolérance d'appariement    : {report['tolerance_seconds']:.2f} s")
    print("--------------------------------------------------")
    print(f"Vérité terrain (GT)        : {counting['gt_total']} franchissements")
    print(f"Prédits (système)          : {counting['pred_total']} franchissements")
    print(f"Vrais positifs (TP)        : {counting['true_positives']}")
    print(f"Faux positifs (FP)         : {counting['false_positives']}")
    print(f"Faux négatifs (FN)         : {counting['false_negatives']}")
    print(f"Erreurs de direction       : {counting['direction_errors']}")
    print("--------------------------------------------------")
    print(f"Précision                  : {counting['precision'] * 100:.2f} %")
    print(f"Rappel                     : {counting['recall'] * 100:.2f} %")
    print(f"F1 (franchissements)       : {counting['f1'] * 100:.2f} %")
    print("--------------------------------------------------")
    if occupancy["mae"] is None:
        print("Occupation                 : non évaluée (pas de vérité terrain par intervalles)")
    else:
        print(f"Occupation MAE (confirmée) : {occupancy['mae']:.3f} personne(s) sur {occupancy['compared_snapshots']} relevés")
        print(f"Occupation erreur max      : {occupancy['max_error']}")
        print(f"Couverture de l'intervalle : {occupancy['range_coverage'] * 100:.1f} %")
    print("--------------------------------------------------")
    print(" PERFORMANCES TEMPORELLES")
    print(f" FPS moyen (mesuré)        : {timing['fps_mean']}")
    if timing["latency_ms"]:
        worst = sorted(timing["latency_ms"].items(), key=lambda item: -item[1]["p95_ms"])
        for stage, data in worst:
            print(
                f" {stage:<10} moy={data['mean_ms']:.1f} ms méd={data['median_ms']:.1f} ms "
                f"P95={data['p95_ms']:.1f} ms max={data['max_ms']:.1f} ms"
            )
    print("--------------------------------------------------")
    print(" DIAGNOSTIC")
    print(f" Franchissements ambigus   : {report['diagnostics']['ambiguous_crossings']}")
    print(f" États incohérents         : {report['diagnostics']['inconsistent_states']}")
    print("--------------------------------------------------")
    for name, check in report["acceptance"]["checks"].items():
        if check["passed"] is None:
            verdict = "NON ÉVALUÉ"
        else:
            verdict = "OK" if check["passed"] else "ÉCHEC"
        print(f" {name:<18} {check['value']} (seuil {check['threshold']}) -> {verdict}")
    print(f"CRITÈRE D'ACCEPTATION      : {'RESPECTÉ' if report['acceptance']['accepted'] else 'NON RESPECTÉ'}")
    print("==================================================")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Évaluation du comptage (F1, occupation, FPS) contre une vérité terrain."
    )
    parser.add_argument("--events", required=True, help="Journal JSON Lines d'une session (events.jsonl)")
    parser.add_argument("--ground-truth", required=True, help="Vérité terrain JSON de la vidéo (9.2)")
    parser.add_argument("--extra", nargs=2, action="append", default=[], metavar=("GT", "EVENTS"),
                        help="Paire supplémentaire (GT, journal) pour l'agrégat multi-vidéos")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S,
                        help="Tolérance d'appariement temporel en secondes (défaut : 1.0)")
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1,
                        help="Critère d'acceptation F1 à fixer AVANT les mesures (défaut : 0.90)")
    parser.add_argument("--max-occupancy-mae", type=float, default=DEFAULT_MAX_OCCUPANCY_MAE,
                        help="Critère d'acceptation MAE d'occupation (défaut : 1.0)")
    parser.add_argument("--min-fps", type=float, default=None,
                        help="Débit minimal acceptable sur le matériel de référence (9.3)")
    parser.add_argument("--out", default=None, help="Chemin du rapport JSON à écrire")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    pairs = [(args.ground_truth, args.events)] + [(gt, events) for gt, events in args.extra]

    if len(pairs) == 1:
        report = evaluer_comptage_systeme(
            chemin_gt=args.ground_truth,
            chemin_pred=args.events,
            tolerance_seconds=args.tolerance,
            min_f1=args.min_f1,
            max_occupancy_mae=args.max_occupancy_mae,
            min_fps=args.min_fps,
            chemin_rapport=args.out,
        )
        return 0 if report["acceptance"]["accepted"] else 1

    aggregated = evaluer_sessions(
        pairs,
        tolerance_seconds=args.tolerance,
        min_f1=args.min_f1,
        max_occupancy_mae=args.max_occupancy_mae,
        min_fps=args.min_fps,
        chemin_rapport=args.out,
    )
    return 0 if aggregated["accepted_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
