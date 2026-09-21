"""Tests de `scripts/gt_matching.py` — proposition de rattachement et métriques.

L'exigence centrale n'est pas la justesse d'une heuristique (elle est pauvre et
assumée comme telle) mais le **verrou** : aucune métrique d'identité ne doit
sortir d'un rattachement que personne n'a relu.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "gt_matching", REPO_ROOT / "scripts" / "gt_matching.py"
)
gt_matching = importlib.util.module_from_spec(_spec)
sys.modules["gt_matching"] = gt_matching
_spec.loader.exec_module(gt_matching)


def _trace_row(second, persons, **kwargs):
    row = {
        "second": second,
        "frame_index": second * 30 + 1,
        "occupancy_operational": kwargs.get("operational", len(persons)),
        "occupancy_observed": kwargs.get("observed", len(persons)),
        "occupancy_uncertain": 0,
        "visible_count": kwargs.get("visible", len(persons)),
        "occluded_count": 0,
        "pistes_vues": len(persons),
        "technical_ids_in_frame": [p["technical_track_id"] for p in persons],
        "persons": persons,
    }
    return row


def _person(person_id, technical_id=None, seen=True):
    return {
        "person_id": person_id,
        "technical_track_id": person_id if technical_id is None else technical_id,
        "state": "PRESENTE",
        "inside_occupancy": True,
        "is_visible": True,
        "seen_this_frame": seen,
        "bbox": [0.0, 0.0, 10.0, 20.0],
        "confidence": 0.9,
    }


@pytest.fixture
def truth():
    return {
        "video": "test/exemple.mp4",
        "resolution": "1280x720",
        "etiquettes": {"P1": "chemise rouge", "P2": "chemise verte"},
        "secondes": [
            {"seconde": 0, "present": ["P1"], "count": 1, "notes": ""},
            {"seconde": 1, "present": ["P1", "P2"], "count": 2, "notes": ""},
            {"seconde": 2, "present": ["P1", "P2"], "count": 2, "notes": ""},
        ],
    }


def test_proposition_est_marquee_non_validee(truth):
    trace = [
        _trace_row(0, [_person(1)]),
        _trace_row(1, [_person(1), _person(2)]),
        _trace_row(2, [_person(1), _person(2)]),
    ]
    proposal = gt_matching.propose(trace, truth)

    assert proposal["statut"] == "proposition"
    assert "NON VALIDÉE" in proposal["_avertissement"]
    # Chaque rattachement porte sa méthode et sa confiance : une correspondance
    # déduite ne doit pas être lisible comme une observation.
    for row in proposal["correspondances"]:
        assert row["methode"] in (
            "propagation_sans_swap_suppose", "a_renseigner"
        )
        assert row["confiance"] in ("moyenne", "faible", "aucune")
        if row["etiquettes"]:
            assert row["justification"]


def test_proposition_par_heure_d_entree_reste_unique(truth):
    """Deux person_id apparus dans la même fenêtre : aucune proposition."""
    trace = [
        _trace_row(0, [_person(1)]),
        _trace_row(1, [_person(1), _person(2), _person(3)]),
    ]
    proposal = gt_matching.propose(trace, truth)
    etiquetees = [r for r in proposal["correspondances"] if r["etiquettes"]]
    assert etiquetees == []
    assert set(proposal["person_ids_sans_proposition"]) == {1, 2, 3}


def test_metriques_identite_refusees_sans_validation(truth, tmp_path):
    trace = [_trace_row(0, [_person(1)])]
    matching = {
        "statut": "proposition",
        "non_visibles": {"intervalles": {}},
        "correspondances": [
            {"second": 0, "person_id": 1, "technical_track_id": 1, "etiquettes": ["P1"]}
        ],
    }
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps(trace[0]) + "\n", encoding="utf-8")
    truth_path = tmp_path / "gt.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    matching_path = tmp_path / "match.json"
    matching_path.write_text(json.dumps(matching), encoding="utf-8")
    out = tmp_path / "rapport.json"

    code = gt_matching.main([
        "metrics", "--trace", str(trace_path), "--ground-truth", str(truth_path),
        "--matching", str(matching_path), "--out", str(out),
    ])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "identite_person_id" not in report
    assert "NON PUBLIÉES" in report["identite"]
    # Les métriques de comptage, elles, ne dépendent d'aucun rattachement.
    assert report["comptage"]["erreur_comptage_max_operational"] is not None


def test_metriques_identite_publiees_apres_validation():
    matching = {
        "statut": "valide",
        "correspondances": [
            {"second": 0, "person_id": 1, "technical_track_id": 1, "etiquettes": ["P1"]},
            {"second": 1, "person_id": 2, "technical_track_id": 2, "etiquettes": ["P1"]},
            {"second": 1, "person_id": 3, "technical_track_id": 3, "etiquettes": ["P2"]},
            {"second": 2, "person_id": 4, "technical_track_id": 4,
             "etiquettes": ["P1", "P2"]},
        ],
    }
    metrics = gt_matching.identity_metrics(matching, "person_id")

    # P1 passe de l'identifiant 1 à 2 puis à 4 : deux changements, trois fragments.
    assert metrics["id_switches_par_etiquette"]["P1"] == 2
    assert metrics["fragments_par_personne"]["P1"] == 3
    # P2 : identifiant 3 puis 4, un changement.
    assert metrics["id_switches_par_etiquette"]["P2"] == 1
    assert metrics["id_switches_reels"] == 3
    # Une ligne portant deux étiquettes est une fusion réelle.
    assert metrics["fusions_reelles"] == 1
    assert metrics["fusions_detail"][0]["etiquettes"] == ["P1", "P2"]


def test_visibilite_derivee_des_intervalles_relus(truth):
    """La visibilité vient du fichier relu, jamais d'une lecture des notes."""
    matching = {"non_visibles": {"intervalles": {"P2": [[2, 2]]}}}
    visibles = gt_matching.visible_labels(truth, matching)

    assert visibles[1] == {"P1", "P2"}
    assert visibles[2] == {"P1"}


def test_comptage_compare_chaque_compteur_a_son_referentiel(truth):
    """operational ↔ présentes, visible_count ↔ visibles annotées (CLAUDE.md §6)."""
    trace = [
        _trace_row(0, [_person(1)], operational=1, visible=1),
        _trace_row(1, [_person(1), _person(2)], operational=2, visible=2),
        # Seconde 2 : P2 est annotée non visible, le pipeline n'en voit qu'une.
        _trace_row(2, [_person(1)], operational=2, visible=1),
    ]
    matching = {"non_visibles": {"intervalles": {"P2": [[2, 2]]}}}
    counting = gt_matching.counting_metrics(trace, truth, matching)

    assert counting["erreur_comptage_max_operational"] == 0
    assert counting["erreur_comptage_max_visible"] == 0
    assert counting["par_seconde"][2]["visibles_annotees"] == 1
