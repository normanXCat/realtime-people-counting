"""Exécution du protocole d'évaluation (spec 9.4, livrable 7).

Ce script orchestre les baselines et les variantes d'ablation déclarées dans
`config/protocol.yaml` :

1. il vérifie que les ensembles de **calibration** et de **test** sont disjoints
   (9.1) et refuse de mélanger les deux ;
2. pour chaque variante, il construit une configuration résolue (surcharges
   revalidées par `src/config.py`, jamais un YAML écrit à la main) et la dépose
   dans le dossier de résultats — c'est ce fichier qui rend le run rejouable ;
3. il lance `src/main.py` sur chaque vidéo de l'ensemble choisi ;
4. il évalue chaque session avec `evaluate_system.py` (F1, occupation, FPS) et
   agrège un rapport par variante puis un rapport global en Markdown.

Exemple (calibration d'abord, ensemble de test seulement pour le rapport final) :

    python scripts/run_protocol.py --dataset calibration
    python scripts/run_protocol.py --dataset test --variant baseline_1 --variant variant_1

Le script n'invente aucune mesure : sans vidéos annotées déclarées, il s'arrête
en expliquant ce qui manque.

**Interaction requise.** La ligne virtuelle est obligatoirement définie à la main
au démarrage de chaque session : chaque vidéo ouvre la première image et attend
deux clics validés par « C ». Le protocole doit donc être piloté par un opérateur
(une ligne identique pour toutes les vidéos d'un même ensemble) ; la ligne
retenue est archivée par session dans `calibration.json` et `LINE_VALIDATED`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

import evaluate_system as evaluation  # noqa: E402
from config import ConfigError, load_config, override_config  # noqa: E402

DEFAULT_PROTOCOL = REPO_ROOT / "config" / "protocol.yaml"


# ---------------------------------------------------------------------------
# Chargement du protocole
# ---------------------------------------------------------------------------
def load_protocol(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Protocole introuvable : {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ConfigError(f"{path} : protocole invalide (schema_version 1 attendu)")
    for required in ("base_config", "datasets", "variants", "acceptance"):
        if required not in payload:
            raise ConfigError(f"{path} : clé obligatoire manquante {required!r}")
    return dict(payload)


def _dataset_entries(protocol: Mapping[str, Any], name: str) -> list[dict[str, str]]:
    raw = (protocol["datasets"] or {}).get(name) or []
    entries: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or "video" not in item or "ground_truth" not in item:
            raise ConfigError(
                f"datasets.{name}[{index}] doit contenir « video » et « ground_truth »"
            )
        entries.append({"video": str(item["video"]), "ground_truth": str(item["ground_truth"])})
    return entries


def check_disjoint(protocol: Mapping[str, Any]) -> None:
    """Vérifie la séparation stricte calibration / test (spec 9.1)."""
    calibration = {entry["video"] for entry in _dataset_entries(protocol, "calibration")}
    test = {entry["video"] for entry in _dataset_entries(protocol, "test")}
    overlap = calibration & test
    if overlap:
        raise ConfigError(
            "Les ensembles de calibration et de test se chevauchent "
            f"({sorted(overlap)}) : le protocole de la section 9.1 est violé."
        )


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
def build_variant_config(
    base_config: str | Path,
    variant: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Écrit la configuration résolue d'une variante (revalidée, donc rejouable)."""
    base = load_config(REPO_ROOT / base_config if not Path(base_config).is_absolute() else base_config)
    overrides = variant.get("config") or {}
    resolved = override_config(base, overrides)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Configuration résolue d'une variante du protocole (spec 9.4).\n"
        f"# variante   : {variant.get('name')}\n"
        f"# description: {variant.get('description', '')}\n"
        f"# surcharges : {json.dumps(overrides, ensure_ascii=False)}\n"
        f"# drapeaux   : {' '.join(variant.get('cli') or [])}\n"
    )
    output_path.write_text(
        header + yaml.safe_dump(resolved.to_dict(), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def run_session(
    config_path: Path,
    video: str,
    output_root: Path,
    session_id: str,
    extra_cli: Sequence[str],
    max_frames: int | None = None,
    dry_run: bool = False,
) -> tuple[int, Path]:
    """Lance une session du pipeline officiel ; retourne (code, dossier de session)."""
    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "main.py"),
        "--config", str(config_path),
        "--source", video,
        "--output-root", str(output_root),
        "--session-id", session_id,
        "--no-show",
        *extra_cli,
    ]
    if max_frames is not None:
        command += ["--max-frames", str(max_frames)]

    session_root = output_root / session_id
    if dry_run:
        print("[DRY-RUN] " + " ".join(command))
        return 0, session_root

    print(f"[RUN] {session_id}")
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return completed.returncode, session_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--dataset", default="calibration", choices=("calibration", "test"))
    parser.add_argument("--variant", action="append", default=None,
                        help="Variante à exécuter (répétable ; défaut : toutes)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Arrêt après N frames par vidéo (diagnostic seulement)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les commandes sans lancer le pipeline")
    args = parser.parse_args(argv)

    protocol = load_protocol(Path(args.protocol))
    check_disjoint(protocol)

    variants = protocol["variants"]
    selected = args.variant or list(variants)
    unknown = [name for name in selected if name not in variants]
    if unknown:
        raise ConfigError(f"Variantes inconnues : {unknown} (connues : {sorted(variants)})")

    print(
        "[PROTOCOLE] La ligne virtuelle est définie à la main : chaque session "
        "ouvre la première image (deux clics puis « C »)."
    )

    entries = _dataset_entries(protocol, args.dataset)
    if not entries:
        print(
            f"[PROTOCOLE] Aucune vidéo déclarée dans datasets.{args.dataset} "
            f"({args.protocol}).\n"
            "Déclarez les vidéos annotées (video + ground_truth) pour exécuter "
            "le protocole de la section 9 : les métriques ne peuvent pas être "
            "produites sans vérité terrain.",
            file=sys.stderr,
        )
        return 2

    acceptance = protocol["acceptance"]
    output_root = REPO_ROOT / protocol["output_root"]
    variant_reports: dict[str, Any] = {}

    for name in selected:
        variant = {**variants[name], "name": name}
        print(f"\n=== VARIANTE {name} : {variant.get('description', '')} ===")
        config_path = build_variant_config(
            protocol["base_config"], variant, output_root / name / "config.yaml"
        )
        pairs: list[tuple[str, str]] = []
        for entry in entries:
            video_stem = Path(entry["video"]).stem
            session_id = f"{name}__{video_stem}"
            code, session_root = run_session(
                config_path=config_path,
                video=entry["video"],
                output_root=output_root,
                session_id=session_id,
                extra_cli=variant.get("cli") or [],
                max_frames=args.max_frames,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue
            if code != 0:
                print(f"[AVERTISSEMENT] session {session_id} terminée avec code {code}", file=sys.stderr)
            events_path = session_root / "events.jsonl"
            if not events_path.exists():
                print(f"[AVERTISSEMENT] journal absent : {events_path}", file=sys.stderr)
                continue
            pairs.append((str(REPO_ROOT / entry["ground_truth"]), str(events_path)))

        if args.dry_run or not pairs:
            continue

        report = evaluation.evaluer_sessions(
            pairs,
            tolerance_seconds=float(acceptance["tolerance_seconds"]),
            min_f1=float(acceptance["min_f1"]),
            max_occupancy_mae=float(acceptance["max_occupancy_mae"]),
            min_fps=acceptance.get("min_fps"),
            chemin_rapport=output_root / name / "report.json",
        )
        variant_reports[name] = report

    if args.dry_run or not variant_reports:
        return 0

    _write_summary(output_root, args.dataset, acceptance, variant_reports)
    return 0 if all(report["accepted_all"] for report in variant_reports.values()) else 1


def _write_summary(
    output_root: Path,
    dataset: str,
    acceptance: Mapping[str, Any],
    variant_reports: Mapping[str, Any],
) -> None:
    lines = [
        f"# Rapport d'évaluation — ensemble « {dataset} »",
        "",
        "Critère d'acceptation (figé avant les mesures) :",
        "",
        f"- F1 des franchissements ≥ {acceptance['min_f1']}",
        f"- Erreur absolue moyenne d'occupation ≤ {acceptance['max_occupancy_mae']} personne",
        f"- Débit minimal ≥ {acceptance['min_fps']}",
        f"- Tolérance d'appariement temporel : {acceptance['tolerance_seconds']} s",
        "",
        "| Variante | Sessions | F1 macro | F1 micro | MAE occupation | FPS moyen | Accepté |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for name, report in variant_reports.items():
        fps_values = [
            session["timing"]["fps_mean"]
            for session in report["reports"]
            if session["timing"]["fps_mean"]
        ]
        fps_mean = round(sum(fps_values) / len(fps_values), 2) if fps_values else None
        mae = report["occupancy_mae_mean"]
        lines.append(
            f"| {name} | {report['sessions']} | {report['f1_mean']:.4f} | "
            f"{report['micro_f1']:.4f} | {'n/a' if mae is None else f'{mae:.3f}'} | "
            f"{'n/a' if fps_mean is None else fps_mean} | "
            f"{'oui' if report['accepted_all'] else 'non'} |"
        )
    lines += [
        "",
        "Rappels de méthode : les seuils sont choisis sur l'ensemble de calibration "
        "et jamais ajustés sur l'ensemble de test (9.1) ; les métriques de détection "
        "et de tracking (mAP, IDF1, ID switches) sont produites séparément à partir "
        "de la vérité terrain MOT dense (9.2).",
        "",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"summary_{dataset}.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[RAPPORT] {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
