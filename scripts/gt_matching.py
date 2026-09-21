#!/usr/bin/env python
"""Rattachement `person_id` ↔ étiquettes humaines, et métriques de vérité terrain.

Ce que ce script fait — et ne fait pas
--------------------------------------

Les quatre métriques du §3 de ``CLAUDE.md`` supposent de savoir quelle piste du
pipeline correspond à quelle personne annotée (``P1`` … ``P13``). Ce lien
n'existe nulle part : l'annotation ne porte pas de coordonnées, le pipeline ne
connaît pas les étiquettes humaines.

Ce script **propose** ce lien ; il ne le décrète pas.

- ``propose`` écrit un fichier de correspondance où **chaque** rattachement
  porte sa ``methode`` et sa ``confiance``, et où ``statut`` vaut
  ``proposition``.
- ``metrics`` refuse de publier ``id_switches_reels``,
  ``fragments_par_personne`` et ``fusions_reelles`` tant que ``statut`` ne vaut
  pas ``valide`` — c'est-à-dire tant qu'un humain n'a pas relu la proposition.
  Les métriques de **comptage**, elles, ne dépendent d'aucun rattachement et
  sont publiées dans tous les cas.

La visibilité seconde par seconde est, elle aussi, une **extraction** des notes
en texte libre de la vérité terrain (``docs/correctif_occlusion.md`` §12.2.1) :
elle est déposée dans le fichier de proposition, sous ``non_visibles``, pour
être relue comme le reste. Rien n'est déduit des notes au moment du calcul.

Entrée : ``trace_per_second.jsonl`` écrit par ``src/main.py --trace-per-second``
(le pipeline réel, jamais un rejeu), et ``tests/fixtures/ground_truth_*.json``.

**Confidentialité** : le relevé complet et le fichier de correspondance portent
les boîtes englobantes des personnes filmées. Ils restent dans ``results/``,
non versionné, comme ``test/`` et ``annotation/``. Seule une trace **agrégée**
(compteurs par seconde, sans le tableau ``persons``) est versionnée.

Usage
-----

    python scripts/gt_matching.py propose \\
        --trace results/<session>/trace_per_second.jsonl \\
        --ground-truth tests/fixtures/ground_truth_fort_occ4.json \\
        --out results/gt_matching_fort_occ4.json

    python scripts/gt_matching.py metrics \\
        --trace results/<session>/trace_per_second.jsonl \\
        --ground-truth tests/fixtures/ground_truth_fort_occ4.json \\
        --matching results/gt_matching_fort_occ4.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Valeur de ``statut`` qui autorise la publication des métriques d'identité.
VALIDATED = "valide"

#: Tolérance, en secondes, entre l'apparition d'une étiquette dans l'annotation
#: et la première observation d'un ``person_id`` par le pipeline. Une personne
#: annotée « présente » à la seconde k peut n'être détectée qu'un peu après.
ENTRY_TOLERANCE_S = 2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def first_seconds_labels(truth: dict) -> dict[str, int]:
    """Première seconde où chaque étiquette est annotée présente."""
    first: dict[str, int] = {}
    for entry in truth["secondes"]:
        for label in entry["present"]:
            first.setdefault(label, int(entry["seconde"]))
    return first


def first_seconds_persons(trace: list[dict]) -> dict[int, int]:
    """Première seconde où chaque ``person_id`` est observé avec une boîte."""
    first: dict[int, int] = {}
    for row in trace:
        for person in row["persons"]:
            if person["seen_this_frame"]:
                first.setdefault(int(person["person_id"]), int(row["second"]))
    return first


def propose(trace: list[dict], truth: dict) -> dict:
    """Construit une proposition de correspondance, jamais une vérité.

    Heuristique unique et volontairement pauvre : **l'ordre d'entrée**. Une
    étiquette qui apparaît à la seconde k est rapprochée d'un ``person_id``
    apparu entre k et k + ``ENTRY_TOLERANCE_S``, à condition que l'appariement
    soit unique dans cette fenêtre. Tout le reste est laissé à renseigner.

    Le rattachement est ensuite **propagé** à toutes les secondes où le
    ``person_id`` est observé : cette propagation suppose qu'aucun swap n'a eu
    lieu, ce qui est exactement ce que la mesure doit établir. Elle est donc
    marquée ``propagation_sans_swap_suppose`` et sa confiance ne dépasse jamais
    celle du rattachement d'origine.
    """
    label_entry = first_seconds_labels(truth)
    person_entry = first_seconds_persons(trace)

    by_entry_label: dict[int, list[str]] = defaultdict(list)
    for label, second in label_entry.items():
        by_entry_label[second].append(label)
    by_entry_person: dict[int, list[int]] = defaultdict(list)
    for person_id, second in person_entry.items():
        by_entry_person[second].append(person_id)

    assignment: dict[int, dict[str, Any]] = {}
    used_labels: set[str] = set()
    for second in sorted(by_entry_label):
        labels = [lab for lab in by_entry_label[second] if lab not in used_labels]
        candidates = [
            person_id
            for offset in range(ENTRY_TOLERANCE_S + 1)
            for person_id in by_entry_person.get(second + offset, [])
            if person_id not in assignment
        ]
        if len(labels) == 1 and len(candidates) == 1:
            assignment[candidates[0]] = {
                "etiquette": labels[0],
                "methode": "heure_entree",
                "confiance": "moyenne",
                "justification": (
                    f"seule étiquette apparue à s{second} et seul person_id "
                    f"apparu entre s{second} et s{second + ENTRY_TOLERANCE_S}"
                ),
            }
            used_labels.add(labels[0])

    correspondances = []
    for row in trace:
        for person in row["persons"]:
            if not person["seen_this_frame"]:
                continue
            person_id = int(person["person_id"])
            proposal = assignment.get(person_id)
            correspondances.append(
                {
                    "second": int(row["second"]),
                    "person_id": person_id,
                    "technical_track_id": int(person["technical_track_id"]),
                    "bbox": person["bbox"],
                    "etiquettes": [proposal["etiquette"]] if proposal else [],
                    "methode": (
                        "propagation_sans_swap_suppose" if proposal else "a_renseigner"
                    ),
                    "confiance": proposal["confiance"] if proposal else "aucune",
                    "justification": proposal["justification"] if proposal else "",
                }
            )

    non_attribues = sorted(set(person_entry) - set(assignment))
    return {
        "statut": "proposition",
        "_avertissement": (
            "PROPOSITION NON VALIDÉE. Tant que « statut » ne vaut pas "
            f"« {VALIDATED} », gt_matching.py metrics refuse de publier "
            "id_switches_reels, fragments_par_personne et fusions_reelles."
        ),
        "_mode_emploi": [
            "1. Ouvrir les images results/<session>/trace_frames/s{k}.jpg : elles "
            "portent la boîte de chaque piste, étiquetée id<person_id>.",
            "2. Les comparer aux images d'annotation annotation/t{k+1}.jpg.",
            "3. Corriger « etiquettes » de chaque ligne : une étiquette, plusieurs "
            "si la boîte contient réellement plusieurs personnes (fusion), aucune "
            "si la boîte ne correspond à personne d'annoté.",
            "4. Mettre « methode » à « inspection_visuelle » et « confiance » à "
            "« haute » sur les lignes relues.",
            "5. Relire « non_visibles » (extraction des notes, §12.2.1).",
            f"6. Passer « statut » à « {VALIDATED} » quand la relecture est faite.",
        ],
        "video": truth.get("video"),
        "resolution": truth.get("resolution"),
        "etiquettes": truth.get("etiquettes"),
        "person_ids_sans_proposition": non_attribues,
        "non_visibles": {
            "_source": (
                "À EXTRAIRE des notes de la vérité terrain (champ « notes », "
                "§12.2.1 de docs/correctif_occlusion.md) : « X occulte Y » est "
                "une occultation partielle (Y reste visible), « Y n'est plus "
                "visible » une disparition totale. Un état non visible persiste "
                "jusqu'à mention explicite du contraire."
            ),
            "_format": "étiquette -> liste d'intervalles [début_s, fin_s] inclusifs",
            "intervalles": {},
        },
        "correspondances": correspondances,
    }


def visible_labels(truth: dict, matching: dict) -> dict[int, set[str]]:
    """Étiquettes **visibles** par seconde = présentes moins les non visibles."""
    intervals = matching.get("non_visibles", {}).get("intervalles", {})
    hidden: dict[int, set[str]] = defaultdict(set)
    for label, ranges in intervals.items():
        for start, end in ranges:
            for second in range(int(start), int(end) + 1):
                hidden[second].add(label)
    return {
        int(entry["seconde"]): set(entry["present"]) - hidden[int(entry["seconde"])]
        for entry in truth["secondes"]
    }


def track_fragmentation(trace: list[dict]) -> dict:
    """Fragmentation de piste — **sans aucun rattachement humain**.

    Deux grandeurs symétriques, toutes deux lisibles dans le seul relevé par
    seconde :

    - ``technical_ids_par_person_id`` : une personne logique qui traverse
        plusieurs identifiants BoT-SORT a vu sa piste mourir et renaître. Sur la
        baseline de ``fort_occ4``, ``person_id 5`` passe par 5, 7, 9, 11 puis
        12 : cinq pistes pour une personne.
    - ``person_ids_par_technical_id`` : un identifiant technique réutilisé par
        plusieurs personnes logiques. C'est le même phénomène vu de l'autre
        côté, et il ne se déduit pas du premier.

    Aucune de ces deux mesures ne dépend de la validation du rattachement aux
    étiquettes humaines : c'est ce qui les rend utilisables dès les lots 1 et 4.
    Elles ne les remplacent pas — elles ignorent les swaps entre deux personnes
    qui gardent chacune leur identifiant.
    """
    par_person: dict[int, list[int]] = defaultdict(list)
    par_technique: dict[int, set[int]] = defaultdict(set)
    for row in trace:
        for person in row["persons"]:
            if not person["seen_this_frame"]:
                continue
            technical = int(person["technical_track_id"])
            if technical < 0:
                continue
            person_id = int(person["person_id"])
            # Séquence sans répétition consécutive : on veut les changements,
            # pas une ligne par seconde.
            if not par_person[person_id] or par_person[person_id][-1] != technical:
                par_person[person_id].append(technical)
            par_technique[technical].add(person_id)

    sequences = {pid: seq for pid, seq in sorted(par_person.items())}
    distincts = {pid: len(set(seq)) for pid, seq in sequences.items()}
    partages = {
        tid: sorted(pids)
        for tid, pids in sorted(par_technique.items())
        if len(pids) > 1
    }
    return {
        "sequences_technical_ids": sequences,
        "technical_ids_distincts_par_person_id": distincts,
        "fragmentation_max": max(distincts.values(), default=None),
        "person_ids_fragmentes": sorted(pid for pid, n in distincts.items() if n > 1),
        "pistes_surnumeraires": sum(max(0, n - 1) for n in distincts.values()),
        "technical_ids_partages": partages,
    }


def counting_metrics(trace: list[dict], truth: dict, matching: dict) -> dict:
    """Métriques de comptage : aucun rattachement nécessaire."""
    present = {int(e["seconde"]): int(e["count"]) for e in truth["secondes"]}
    visible = {s: len(labels) for s, labels in visible_labels(truth, matching).items()}

    per_second = []
    for row in trace:
        second = int(row["second"])
        if second not in present:
            continue
        per_second.append(
            {
                "second": second,
                "presentes": present[second],
                "occupancy_operational": row["occupancy_operational"],
                "ecart_operational": row["occupancy_operational"] - present[second],
                "visibles_annotees": visible[second],
                "visible_count": row["visible_count"],
                "ecart_visible": row["visible_count"] - visible[second],
                "occupancy_observed": row["occupancy_observed"],
                "pistes_vues": row["pistes_vues"],
            }
        )
    return {
        "erreur_comptage_max_operational": max(
            (abs(r["ecart_operational"]) for r in per_second), default=None
        ),
        "erreur_comptage_max_visible": max(
            (abs(r["ecart_visible"]) for r in per_second), default=None
        ),
        "pistes_vues_min": min((r["pistes_vues"] for r in per_second), default=None),
        "pistes_vues_max": max((r["pistes_vues"] for r in per_second), default=None),
        "par_seconde": per_second,
    }


def identity_metrics(matching: dict, key: str = "person_id") -> dict:
    """Les trois métriques qui dépendent du rattachement validé."""
    per_label: dict[str, dict[int, int]] = defaultdict(dict)
    fusions: list[dict] = []
    for row in matching["correspondances"]:
        labels = row.get("etiquettes") or []
        for label in labels:
            per_label[label][int(row["second"])] = int(row[key])
        if len(labels) > 1:
            fusions.append(
                {"second": row["second"], key: row[key], "etiquettes": labels}
            )

    switches = {}
    fragments = {}
    for label, by_second in per_label.items():
        ordered = [by_second[s] for s in sorted(by_second)]
        switches[label] = sum(
            1 for a, b in zip(ordered, ordered[1:]) if a != b
        )
        fragments[label] = len(set(ordered))
    return {
        "cle": key,
        "id_switches_reels": sum(switches.values()),
        "id_switches_par_etiquette": switches,
        "fragments_par_personne": fragments,
        "fragments_max": max(fragments.values(), default=None),
        "fusions_reelles": len(fusions),
        "fusions_detail": fusions,
    }


def cmd_propose(args: argparse.Namespace) -> int:
    trace = load_jsonl(Path(args.trace))
    truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    proposal = propose(trace, truth)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(proposal, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    proposed = sum(1 for r in proposal["correspondances"] if r["etiquettes"])
    total = len(proposal["correspondances"])
    print(f"[PROPOSITION] {out}")
    print(f"[PROPOSITION] {proposed}/{total} lignes portent une étiquette proposée")
    print(
        "[PROPOSITION] person_id sans proposition : "
        f"{proposal['person_ids_sans_proposition']}"
    )
    print("[PROPOSITION] statut = proposition — métriques d'identité verrouillées")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    trace = load_jsonl(Path(args.trace))
    truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    matching = json.loads(Path(args.matching).read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "trace": str(args.trace).replace("\\", "/"),
        "matching": str(args.matching).replace("\\", "/"),
        "statut_matching": matching.get("statut"),
        "comptage": counting_metrics(trace, truth, matching),
        "fragmentation": track_fragmentation(trace),
    }
    if matching.get("statut") == VALIDATED:
        report["identite_person_id"] = identity_metrics(matching, "person_id")
        report["identite_technical_id"] = identity_metrics(matching, "technical_track_id")
    else:
        report["identite"] = (
            "NON PUBLIÉES : le rattachement porte le statut "
            f"« {matching.get('statut')} », pas « {VALIDATED} ». "
            "id_switches_reels, fragments_par_personne et fusions_reelles "
            "supposent un rattachement relu par un humain."
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[MÉTRIQUES] {out}")
    counting = report["comptage"]
    print(
        "[MÉTRIQUES] erreur_comptage_max_operational = "
        f"{counting['erreur_comptage_max_operational']} | "
        f"erreur_comptage_max_visible = {counting['erreur_comptage_max_visible']}"
    )
    print(
        f"[MÉTRIQUES] pistes_vues : min {counting['pistes_vues_min']}, "
        f"max {counting['pistes_vues_max']} (diagnostic, sans cible)"
    )
    frag = report["fragmentation"]
    print(
        f"[MÉTRIQUES] fragmentation de piste : max {frag['fragmentation_max']} "
        f"identifiants techniques pour un person_id, "
        f"{frag['pistes_surnumeraires']} pistes surnuméraires, "
        f"{len(frag['technical_ids_partages'])} identifiant(s) technique(s) "
        "partagé(s) par plusieurs person_id"
    )
    if "identite" in report:
        print(f"[MÉTRIQUES] {report['identite']}")
    else:
        for key in ("identite_person_id", "identite_technical_id"):
            block = report[key]
            print(
                f"[MÉTRIQUES] {block['cle']} : id_switches_reels="
                f"{block['id_switches_reels']} fusions_reelles="
                f"{block['fusions_reelles']} fragments_max={block['fragments_max']}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--trace", required=True, help="trace_per_second.jsonl")
    common.add_argument(
        "--ground-truth", required=True, help="tests/fixtures/ground_truth_*.json"
    )

    proposer = subparsers.add_parser(
        "propose", parents=[common], help="Écrit une proposition de rattachement"
    )
    proposer.add_argument("--out", required=True, help="Fichier de correspondance")
    proposer.set_defaults(func=cmd_propose)

    metrics = subparsers.add_parser(
        "metrics", parents=[common], help="Calcule les métriques de vérité terrain"
    )
    metrics.add_argument("--matching", required=True, help="Fichier de correspondance")
    metrics.add_argument("--out", default=None, help="Rapport JSON (facultatif)")
    metrics.set_defaults(func=cmd_metrics)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
