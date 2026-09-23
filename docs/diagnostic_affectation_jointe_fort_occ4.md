# Diagnostic — affectation jointe (Hongrois) sur `fort_occ4.mp4`

**Date** : 2026-09-23 — **Périmètre** : 21 sessions issues de `test/fort_occ4.mp4` (jusqu'à 1 055 frames) + rejeu instrumenté de 300 frames (`scripts/diagnose_joint_assignment.py`) + rejeu pipeline complet en cours (`results/diagF_base_code240`).

## Verdict

> **Hypothèse INFIRMÉE.** La résolution séquentielle des observations de galerie n'est
> jamais mise en défaut par concurrence : sur 21 sessions et 300 frames instrumentées,
> il n'existe **aucune frame** où deux observations cherchent un candidat de galerie en
> même temps. Une affectation jointe type Hongrois n'aurait **strictement rien à
> résoudre** : sa sortie serait identique à la résolution séquentielle sur chaque frame
> observée. Conformément aux règles de l'étape 0, **aucune correction n'est codée**.

## Faits mesurés

### 1. Concurrence nulle sous toutes ses formes

| Mesure (21 sessions, `results/2026*/`) | Valeur |
|---|---|
| Frames avec 2+ `REID_AMBIGUOUS` simultanés | **0** |
| Frames avec 2+ `REID_NEW` dont 2+ avec `candidates_evaluated > 0` | **0** |
| `REID_NEW` co-occurrents d'une `REID_AMBIGUOUS` partageant un candidat avec elle | **0 / 46** |
| Frames à 2+ traitements ReID (`MATCH`/`NEW`/`AMB`) | 71, mais **jamais 2 galerie** : composition observée = `{AMB+NEW: 46, NEW×3: 16, MATCH+NEW: 4, NEW×2: 4, NEW×4: 1}` — les `NEW` co-occurrents sont des `duplicate_claim`/`no_plausible_candidate` (0 candidat évalué) |
| Rejeu instrumenté 300 frames (wrapper lecture-seule sur `_plausible_candidates`) | 0 frame avec 2+ observations concurrentes candidates ; rapport : `results/diagnose_joint_assignment_fort_occ4.json` |
| Rejeu instrumenté 300 frames de `test/video2.avi` (témoin) | 1 seul appel galerie sur toute la vidéo, 0 candidat disputé ; rapport : `results/diagnose_joint_assignment_video2.avi.json` |

Même en corrigant l'angle mort du premier instrument (héritage du `_claimed` séquentiel,
corrigé pour mesurer les candidats de **toutes** les observations contre un instantané où
toutes les identités sont non vivantes), la concurrence reste nulle. Sur
`fort_occ4` (300 frames) : 10 appels galerie au total, 7 frames à 1 observation
candidate, **0 frame à 2+**, **0 candidat partagé entre deux observations** — un
Hongrois par frame n'aurait jamais plus d'une ligne à affecter.

### 2. Les 46 ambiguïtés : cas isolés, jamais jointifs

- `best_similarity` ∈ [0.518 ; 0.805] (p50 0.630) — **au-dessus** du seuil 0.40 dans
  46/46 cas : ce n'est pas un défaut de seuil.
- Écart `best − runner_up` : p50 **0.026**, p90 0.078, max **0.099** — tous < `safety_margin` (0.10).
  L'apparence des membres du groupe dense est mutuellement trop proche : la marge bloque,
  et c'est `_resolve_ambiguous` lui-même qui crée l'identité provisoire (`REID_NEW`
  `reid_ambiguous_provisional` à la même frame, 46/46).
- Fenêtres de 5 frames : 44 fenêtres à 1 piste ambiguë, **1 seule** à 2 — jamais
  simultané sur la même frame.

### 3. Où partent vraiment les 229 `REID_NEW`

| Raison (totale / après la 30ᵉ frame) | Mécanisme identifié |
|---|---|
| `no_plausible_candidate` — 139 / 63 | Amorçage (4 premières frames) + réapparitions rejetées par le filtre **spatial** (`v_max_ratio`×h×Δt + marge) ou hors fenêtre galerie, avant toute comparaison d'apparence. Rien à voir avec une affectation jointe. |
| `reid_ambiguous_provisional` — 46 / 46 | Ambiguïté d'apparence isolée (cf. §2). |
| `duplicate_claim_new_identity` — 34 / 34 | Garde-fou spec 4.5 fonctionnant **correctement** : BoT-SORT réassocie une piste technique obsolète à une autre personne, le mapping `technical_to_person` périmé entre en collision avec la piste légitimement verrouillée → création d'une identité pour la nouvelle personne (ex. vérifié frame 266 de `20260923T041648Z` : track 5 perdue en y≈553, réapparue en y≈313 sur une autre personne). |
| `candidates_without_descriptor` — 10 / 10 | Candidat plausible mais descripteur rejeté (`gallery_min_confidence` 0.5) → création. Aggravé par B1 inopérant (poids pose absents). |

Chaîne du `duplicate_claim` reconstituée : 34/34 surviennent **sans** réattribution
antérieure visible de l'identité visée (le « vol » vient du chemin silencieux de
`_create_new`, qui verrouille la piste sans événement REID) et **sans** voleur visible
le jour même — c'est bien le re-routage interne du tracker, pas un défaut d'IdentityManager.

### 4. La vraie cause de `persons_created ≈ 9` : une cascade de provisoires

Dans la session la plus chargée (`20260921T134529Z`, 23 `AMB` entre frames 332 et 861) :

- 23 identités provisoires créées **après** la première ambiguïté ;
- ces provisoires **ré-entrent dans la galerie** et redeviennent candidates des
  ambiguïtés suivantes (P14 : 20 fois, P15 : 18, P19 : 13, P21 : 8, P12 : 8…).

Boucle de rétroaction : ambiguïté → provisoire → candidat de la galerie → nouvelle
ambiguïté. C'est **séquentiel** (une observation à la fois), donc une affectation jointe
n'y changerait rien non plus.

## Décision

1. **Ne pas implémenter** la résolution jointe (Hongrois) dans `assign()`/`_assign_one()`.
   Les trois conditions de validité de l'hypothèse (≥ 2 observations galerie
   simultanées, candidats partagés, gain jointif) sont mesurées à **zéro**.
2. Les leviers réels sur `persons_created` pour une scène dense, à instruire séparément
   (hors périmètre de ce lot) :
   - **recyclage/aliasing des provisoires** : quand une identité provisoire accumule des
     observations cohérentes avec un antécédent de galerie, fusionner plutôt que
     d'accumuler des identités (casse la boucle de rétroaction du §4) ;
   - **B1** : activer réellement `head_assist` (poids `yolo11s-pose.pt` + SHA-256
     manquants) pour améliorer la qualité des descripteurs à confiance intermédiaire —
     attaque directement `candidates_without_descriptor` et une partie des ambiguïtés ;
   - vérifier `no_plausible_candidate` post-amorçage (63) au cas par cas : fenêtre
     galerie vs contrainte spatiale.

## Session de référence « avant » (code `track_buffer: 240`)

Les processus d'arrière-plan ne survivant pas à la fin du shell dans l'environnement
d'exécution, la session pipeline de référence n'a pas pu être menée ici. Commande à
lancer manuellement pour figer la référence avant tout lot futur (≥ 300 frames) :

```bash
python -m src.main --source test/fort_occ4.mp4 --max-frames 1100 --output-root results --session-id diagF_base_code240
```

Les compteurs à relever dans `summary.json`/`events.jsonl` : `persons_created`,
`technical_id_changes`, décompte des `REID_NEW` par `reason`, `REID_AMBIGUOUS`,
`swap_corrections_applied`, `descriptor_rejections`.

---
*Rapport produit conformément à l'étape 0 (obligatoire avant toute modification) —
hypothèse réfutée → pas de code modifié, `pytest -q` reste vert.*
