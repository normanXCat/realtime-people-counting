# Diagnostic — swaps d'identifiant à 3+ personnes et personne assise non re-détectée

Rapport **préalable à toute modification de code** (étape 0 du lot). Il confirme ou
infirme les deux hypothèses A et B, et documente ce qui est déjà appliqué dans la
configuration livrée.

Jeu de données analysé : les 59 sessions exploitables de `results/2026*/`
(`summary.json` + `events.jsonl`), plus les dumps de suivi
`results/video2_frames_dump.jsonl` (740 frames : `technical_track_id`, confiance,
bbox, ancre par frame). Les vidéos `test/video3.mp4` et
`test/5121204_School_Classroom_1280x720.mp4` référencées par les dumps
antérieurs **ne sont pas présentes** dans l'arbre de travail : aucun nouveau
rejeu de la scène de classe dense n'est possible ici, seul le code et les
journaux disponibles font foi.

---

## 1. Problème A — inversion d'ID à 3+ personnes en occlusion simultanée

### 1.1 Hypothèse A : `_correct_cross_swaps` ne traite que des paires

**Confirmée**, sur deux plans indépendants.

**(a) Analyse structurelle du code** (`src/identity_manager.py`, `_correct_cross_swaps`) :

- la méthode itère sur des **paires** `(obs_a, obs_b)`, calcule
  `direct = sim(a↔ref_a) + sim(b↔ref_b)` et `crossed = sim(a↔ref_b) + sim(b↔ref_a)`,
  puis échange le mapping des deux pistes dès que `crossed > direct + margin` ;
- après une correction, `corrected_tracks` exclut les deux pistes de toute
  décision ultérieure de la même frame (`break` sur la boucle interne), donc les
  pistes déjà « corrigées » ne peuvent plus participer à la résolution du reste
  du groupe ;
- il n'existe **aucune** construction de matrice de similarité N×N ni
  d'affectation globale : une permutation circulaire à 3 (A→B→C→A) n'est pas
  représentable comme une composition de transpositions décidées indépendamment.

Contre-exemple numérique (démontré par le nouveau test
`test_permutation_circulaire_a_trois_pistes_est_corrigee_en_une_fois`) :
soient trois descripteurs orthogonaux `desc_a`, `desc_b`, `desc_c`, trois pistes
techniques `7↔1`, `8↔2`, `9↔3`, puis la permutation circulaire
`7→desc_b`, `8→desc_c`, `9→desc_a`.

- paire (7, 8) : `direct = 0`, `crossed = 1` → l'algorithme échange 7 et 8 ;
- piste 8 devient `person 1` alors qu'elle porte `desc_c` (**faux**) ;
- piste 9 (`desc_a`) reste `person 3` (**faux**) ;
- `corrected_tracks = {7, 8}` bloque toute correction ultérieure de 8.

Résultat : 1 piste sur 3 correcte, et surtout **un état incohérent appliqué**
(le mapping est modifié alors que le groupe entier n'a pas été résolu).

**(b) Analyse empirique des journaux** :

| mesure | valeur |
|---|---|
| sessions exploitables analysées | 59 |
| sessions avec `occupancy_confirmed >= 3` (scènes à 3+ personnes) | 50 |
| `IDENTITY_SWAP_CORRECTED` émis, **toutes sessions confondues** | **0** |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` émis | 4 |

Les 4 discontinuités (similarités 0,385 / 0,465 vs seuil 0,50) sont toutes des
`possible_internal_swap` sur des pistes isolées (frame 267 / piste technique 5 /
`person_id` 9, et frame 792 / piste 20 / `person_id` 11) et **aucune** n'a
déclenché de correction. Dans la session `20260918T113202Z`, la fenêtre
autour de la frame 267 ne contient que **1 à 2 pistes simultanées** : ce journal
ne contient donc pas de cas à 3 pistes chevauchantes sur lequel la correction par
paires aurait pu s'exercer.

> **Limite explicite** : `events.jsonl` ne journalise pas la bbox de chaque piste
> à chaque frame ; la fenêtre « 3+ pistes simultanées et spatialement proches »
> ne peut donc pas être reconstruite exhaustivement à partir des seuls journaux.
> La confirmation de l'hypothèse A repose sur l'analyse structurelle (a), qui est
> définitive, et sur le fait mesuré que la correction par paires n'a **jamais**
> produit un seul événement sur 59 sessions, y compris 50 scènes à 3+ personnes.

### 1.2 Conséquence retenue

Corriger A nécessite une **résolution globale par groupe** (matrice N×N +
`scipy.optimize.linear_sum_assignment`), le cas à 2 pistes restant traité par le
chemin par paires déjà validé.

---

## 2. Problème B — personne assise non re-détectée

### 2.1 Hypothèse B : la détection de réapparition tombe entre `track_low_thresh`
(0,1) et `new_track_thresh` (0,7), sans `technical_track_id`

**Confirmée**, mécanisme identifié dans Ultralytics 8.4.126
(`ultralytics/trackers/track.py::on_predict_postprocess_end`) :

```python
tracks = tracker.update(det, result.orig_img, **kwargs)
if len(tracks) == 0:
    continue                      # résultat laissé tel quel : boxes SANS id
idx = tracks[:, -1].astype(int)
predictor.results[i] = result[idx]  # seules les pistes confirmées sont conservées
...
predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))
```

- tant que le tracker produit au moins une piste, seules les détections
  **associées** sont retournées : une réapparition non associée (score sous
  `new_track_thresh`) disparaît de `result.boxes` ;
- quand le tracker ne produit **aucune** piste, le résultat brut est conservé :
  `result.boxes` porte alors des détections avec `boxes.id is None`.

Mesure sur `results/video2_frames_dump.jsonl` (740 frames, BoT-SORT réel) :

| grandeur | valeur |
|---|---|
| frames contenant au moins une boîte **sans** `track_id` | 6 (frames 80, 108-112) |
| dont frames où **toutes** les boîtes sont sans `track_id` | 6 |
| frames « mixtes » (boîtes tracées **et** non tracées) | **0** |
| boîtes non tracées, confiance min / médiane / max | 0,112 / **0,174** / 0,768 |
| boîtes non tracées avec confiance dans `[0.1, 0.7[` | **6 / 7** |
| boîtes **tracées** avec confiance dans `[0.1, 0.7[` | 73 / 1068 |

Lecture : les 6 frames sans aucun `track_id` portent exactement des scores
faibles de la gamme `[track_low_thresh, new_track_thresh)`. Dans le pipeline
actuel (`src/main.py`), ces boîtes ne franchissent jamais le filtre
`if tracker_ids_present and boxes_count:` : `diagnostics` les classe en
`TRACK_ID_ABSENT` mais **aucune `Observation` n'est construite**, donc
`IdentityManager.assign` ne les voit jamais et **aucune tentative de
ré-identification n'a lieu**. La personne reste dans la borne haute de
l'occupation (`occupancy_uncertain`) sans jamais revenir dans `observed`.

Deux causes distinctes apparaissent toutefois, et le correctif B3 ne couvre que
la seconde :

1. **limite de détection** : aucune boîte YOLO dans la zone
   (`results/diagnostic_occlusion_video2_v2.json` : 2 cas sur 2 « aucun boîte
   YOLO dans la zone » ; `results/diagnostic_occlusion_video3.json` : 19 cas sur
   30). Aucun correctif `IdentityManager` ne peut récupérer une observation qui
   n'existe pas ;
2. **boîte existante mais jamais rattachée** : `video3`, 11 cas sur 30
   classés « bug de logique (rattachement IdentityManager / warm-up) » — c'est
   exactement ce que le chemin de secours « galerie » (B3) adresse pour la
   sous-cas où la détection n'atteint pas `new_track_thresh`.

### 2.2 Conséquence retenue

Un chemin de réassociation **assisté par galerie** doit être ajouté pour les
observations non trackées de la gamme `[track_low_thresh, new_track_thresh)`,
désactivé par défaut (`reid.long_term.gallery_rescue_enabled: false`) et
mesuré séparément via un événement dédié `REID_GALLERY_RESCUE`.

---

## 3. État de la configuration livrée (items A2, B1, B2)

Vérifié dans `config/pipeline.yaml` **et** `src/configs/custom_botsort.yaml`
(le test de synchronisation clé-par-clé est conservé) :

| item | valeur attendue | état livré | action |
|---|---|---|---|
| A2 `tracker.appearance_thresh` | 0,35 (durcir le ReID natif) | **0.35** dans les deux YAML | déjà appliqué — rien à dupliquer |
| B1 `presence.head_assist.enabled` | `true` | **true** | déjà appliqué |
| B2 `tracker.track_buffer` | 240 | **240** dans les deux YAML + `TrackerConfig` | déjà appliqué |

**Réserve B1 (bloquante en production)** : `models/yolo11s-pose.pt` est
**absent** du dépôt et `presence.head_assist.pose_model_expected_sha256` vaut
`null`. L'activation est donc inopérante aujourd'hui : au démarrage,
`main.py` journalise `CONFIG_WARNING` (`head_assist_model_missing`), retombe sur
`models/yolo11n.pt` et force `head_assist.enabled=false` dans la configuration
**résolue de session**. Le flag livré peut rester `true` (comportement de repli
explicite et testé), mais l'assistance tête ne sera réellement active qu'une fois
le poids téléchargé et `pose_model_expected_sha256` renseigné.

---

## 4. Plan de correction retenu (dans l'ordre demandé)

1. **A1** — résolution globale par groupe de pistes spatialement proches
   (chevauchement de bbox ou ancres à distance < seuil relatif) avec
   `scipy.optimize.linear_sum_assignment` ; cas à 2 pistes inchangé ;
   `IDENTITY_SWAP_CORRECTED` par piste corrigée avec `group_size` ;
   nouveau paramètre `reid.swap_correction.group_proximity_ratio` (aucun seuil
   en dur).
2. **A2** — déjà appliqué (0,35) : aucun changement, couvert par le test de
   synchronisation existant.
3. **B1** — déjà appliqué (`enabled: true`) ; réserve documentée ci-dessus sur
   les poids de pose.
4. **B2** — déjà appliqué (240) ; non dupliqué.
5. **B3** — chemin de secours galerie : `main.py` transmet les détections sans
   `track_id` de la gamme `[track_low_thresh, new_track_thresh)` comme
   « observations non trackées » ; `IdentityManager.try_gallery_rescue` applique
   le filtre spatio-temporel et le seuil d'apparence habituels (inchangés) et ne
   réattribue une identité que si un candidat **unique** dépasse le seuil ;
   jamais de création d'identité. Paramètres
   `reid.long_term.gallery_rescue_enabled` (défaut `false`) et
   `reid.long_term.gallery_rescue_min_confidence` (défaut = `track_low_thresh`).
   Événement dédié `REID_GALLERY_RESCUE`.

L'activation finale de `gallery_rescue_enabled` reste une décision explicite,
conditionnée au tableau avant/après (section 5 de ce dépôt, `README` / sortie de
mesure) et non un défaut silencieux.
