#!/usr/bin/env python
"""Mesure reproductible d'un corpus vidéo, **en appelant le pipeline réel**.

Pourquoi ce script
------------------

Un harnais qui rejoue ``OccupancyManager.process_frame`` de son côté est un
second système : il diverge du premier sans prévenir. La session 0 l'a montré
de la pire manière — selon la base de temps employée, la même vidéo produisait
13 purges ou 0 (``docs/correctif_occlusion.md`` §11.7.3). Ce script **n'a donc
aucune logique de comptage** : il lance ``src/main.py`` en sous-processus, avec
la ligne virtuelle passée par ``--line``, puis relit les artefacts de session
(``summary.json``, ``events.jsonl``) que le pipeline a lui-même écrits.

Tout ce qu'il mesure est donc, par construction, ce que le système livré fait.

Le schéma de sortie est **stable** : les lots 1 à 7 comparent leurs tableaux
avant/après sur ces champs. Le modifier en cours de plan rendrait les
comparaisons entre lots incohérentes.

Usage
-----

    python scripts/measure_corpus.py --line 0.05,0.6,0.95,0.6
    python scripts/measure_corpus.py --videos test/ --line 0.05,0.6,0.95,0.6 \
        --repeat 3 --out results/baseline.jsonl
    python scripts/measure_corpus.py --videos test/fort_occ4.mp4 \
        --line 0.05,0.6,0.95,0.6 --repeat 3

Les scripts ``measure_tracker_swaps.py`` et ``diagnose_occlusion.py`` restent en
place et ne sont ni remplacés ni fusionnés : ils répondent à d'autres questions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Extensions vidéo reconnues pour le balayage d'un dossier.
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v")

#: Écart minimal, en secondes, entre deux observations d'une même identité
#: au-delà duquel l'absence est comptée comme « perte longue ». Aligné sur
#: ``timing.grace_period_seconds`` : au-delà, la FSM purge.
DEFAULT_LONG_GAP_S = 5.0


def git_sha() -> str:
    """Empreinte du dépôt au moment de la mesure, ou ``unknown``.

    Une mesure qu'on ne peut pas rattacher à un état du code n'est pas
    reproductible : l'information est donc relevée, jamais devinée.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    if out.returncode != 0 or not sha:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return sha + ("-dirty" if dirty.stdout.strip() else "")


def discover_videos(target: Path) -> list[Path]:
    """Un fichier vidéo, ou toutes les vidéos d'un dossier, triées par nom."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Chemin introuvable : {target}")
    return sorted(
        path
        for path in target.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def run_pipeline(
    video: Path,
    line: str,
    session_root: Path,
    session_id: str,
    extra_args: list[str],
) -> tuple[int, str]:
    """Lance ``src/main.py`` en sous-processus. Retourne (code, stderr).

    Le pipeline est invoqué tel quel : aucune variable d'environnement, aucun
    réglage caché. Ce qui est mesuré est ce qui est livré.
    """
    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "main.py"),
        "--mode", "comptage",
        "--source", str(video),
        "--line", line,
        "--no-show",
        "--output-root", str(session_root),
        "--session-id", session_id,
        *extra_args,
    ]
    completed = subprocess.run(
        command, cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    return completed.returncode, completed.stderr


def long_gaps(events: list[dict], threshold_s: float) -> int:
    """Nombre d'absences de plus de ``threshold_s`` **suivies d'un retour**.

    Définition explicite, car elle se distingue de deux voisines :

    - une identité perdue et **jamais** revue n'est pas comptée ici : elle
      produit un ``PURGE``, qui se lit dans le dictionnaire d'événements ;
    - une occultation courte suivie d'un retour n'est pas comptée non plus.

    L'instant de référence est celui du dernier événement mentionnant
    l'identité avant son retour, ce qui rend la mesure dépendante de la
    granularité du journal — d'où sa publication à côté des compteurs bruts,
    jamais à leur place.
    """
    last_seen: dict[int, float] = {}
    count = 0
    for event in events:
        person_id = event.get("person_id")
        if person_id is None:
            continue
        timestamp = event.get("timestamp_s")
        if not isinstance(timestamp, (int, float)):
            continue
        person_id = int(person_id)
        previous = last_seen.get(person_id)
        if (
            previous is not None
            and event["type"] in ("REID_MATCH", "STATE_TRANSITION")
            and timestamp - previous > threshold_s
        ):
            count += 1
        last_seen[person_id] = float(timestamp)
    return count


def collect(session_dir: Path, threshold_s: float) -> dict:
    """Relit les artefacts écrits par le pipeline. N'en recalcule aucun."""
    summary_path = session_dir / "summary.json"
    events_path = session_dir / "events.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    events: list[dict] = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))

    counters = summary.get("counters", {})
    continuity = summary.get("appearance_continuity", {})
    identity = summary.get("identity", {})
    fps = summary.get("fps", {})

    return {
        "event_counts": dict(sorted(Counter(e["type"] for e in events).items())),
        "frames_processed": summary.get("frames_processed"),
        "processing_fps": fps.get("mean"),
        "processing_fps_median": fps.get("median"),
        "processing_fps_min": fps.get("min"),
        "duration_s": summary.get("duration_s"),
        "technical_ids": identity.get("technical_ids"),
        "persons_created": identity.get("persons_created"),
        "technical_id_changes": continuity.get("technical_id_changes"),
        "appearance_discontinuities": continuity.get("discontinuities"),
        "descriptor_rejections": continuity.get("descriptor_rejections"),
        "long_gaps_count": long_gaps(events, threshold_s),
        "occupancy_operational": counters.get("occupancy_operational"),
        "occupancy_observed": counters.get("occupancy_observed"),
        "occupancy_uncertain": counters.get("occupancy_uncertain"),
        "occupancy_range": counters.get("occupancy_range"),
        "total_in": counters.get("total_in"),
        "total_out": counters.get("total_out"),
        "total_new": counters.get("total_new"),
    }


def measure(args: argparse.Namespace) -> list[dict]:
    videos = discover_videos(Path(args.videos))
    if not videos:
        raise FileNotFoundError(f"Aucune vidéo trouvée dans {args.videos}")

    sha = git_sha()
    extra_args: list[str] = []
    if args.max_frames is not None:
        extra_args += ["--max-frames", str(args.max_frames)]
    if args.config is not None:
        extra_args += ["--config", args.config]

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="measure_corpus_") as tmp:
        session_root = Path(args.session_root) if args.session_root else Path(tmp)
        for video in videos:
            for run_index in range(1, args.repeat + 1):
                session_id = f"{video.stem}_run{run_index}"
                print(f"[MESURE] {video.name} — rejouage {run_index}/{args.repeat}")
                code, stderr = run_pipeline(
                    video, args.line, session_root, session_id, extra_args
                )
                row = {
                    "source": str(video).replace("\\", "/"),
                    "run_index": run_index,
                    "git_sha": sha,
                    "timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    # `wall` tant que le pipeline horodate sur time.perf_counter
                    # (src/main.py). Le lot 1 traite cette base de temps ; le
                    # champ existe pour que les mesures d'avant et d'après
                    # restent distinguables sans relire le code.
                    "time_base": "wall",
                    "line": args.line,
                    "exit_code": code,
                }
                session_dir = session_root / session_id
                if code != 0 or not (session_dir / "summary.json").exists():
                    row["error"] = (stderr or "").strip()[-2000:]
                    print(f"[ÉCHEC] {video.name} run {run_index} : code {code}")
                else:
                    row.update(collect(session_dir, args.long_gap_seconds))
                rows.append(row)

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--videos",
        default="test",
        help="Dossier de vidéos ou fichier unique (défaut : test/)",
    )
    parser.add_argument(
        "--line",
        required=True,
        metavar="x1,y1,x2,y2",
        help="Ligne de convention, identique pour tout le corpus. Obligatoire : "
             "une ligne implicite rendrait les mesures incomparables.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Rejouages par vidéo, pour le contrôle de bruit (défaut : 1)",
    )
    parser.add_argument("--config", default=None, help="YAML de configuration")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limite de frames (diagnostic)"
    )
    parser.add_argument(
        "--session-root", default=None,
        help="Racine des sessions du pipeline (défaut : dossier temporaire, "
             "supprimé en fin de mesure)",
    )
    parser.add_argument(
        "--long-gap-seconds", type=float, default=DEFAULT_LONG_GAP_S,
        help=f"Seuil de « perte longue » en secondes (défaut : {DEFAULT_LONG_GAP_S})",
    )
    parser.add_argument(
        "--out", default=None,
        help="Fichier JSON Lines de sortie (défaut : results/measure_corpus.jsonl)",
    )
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat doit valoir au moins 1")

    rows = measure(args)

    out_path = Path(args.out) if args.out else REPO_ROOT / "results" / "measure_corpus.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[MESURE] {len(rows)} ligne(s) ajoutée(s) à {out_path}")
    failures = [r for r in rows if r.get("exit_code") != 0]
    if failures:
        print(f"[MESURE] {len(failures)} rejouage(s) en échec")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
