# Correctif occlusion, identité et assistance tête — rapport d'exécution

Ce document est le livrable attendu par `prompt/prompt_correctif_unifie.md` §8.
Il rapporte **ce qui a été mesuré**, ce qui a été appliqué, et ce qui ne l'a pas
été. Aucune conclusion n'y figure sans la mesure qui la soutient.

**Périmètre de cette session** : lots A et B uniquement. La section 6
(activation réelle du `head_assist`) n'a **pas** été exécutée — voir §6 ci-dessous.

---

## 1. État au démarrage (§0) — synchronisation

`git status -s` : arbre propre. `main` synchronisée avec `origin/main`.

**HEAD était `d4915b5 revert: annule le correctif occlusion dense et assistance
tête (lots A et B)`** : le travail des lots A et B existait déjà dans l'historique
sous `0e6e028`, et avait été **délibérément annulé**. Conséquence : **aucun**
réglage de ce document n'était appliqué au démarrage.

| Réglage | Valeur au démarrage | Cible du prompt | Statut en fin de session |
|---|---|---|---|
| `tracker.match_thresh` | 0.9 | 0.75 | **0.9 — non appliqué** (mesure défavorable, §3) |
| `tracker.proximity_thresh` | 0.5 | 0.8 | **0.5 — non appliqué** (effet mesuré nul, §3) |
| `tracker.new_track_thresh` | 0.7 | 0.45 | **0.7 — non appliqué** (bénéfice non mesuré, §3) |
| `reid.long_term.gallery_retention_seconds` | `null` | 12.0 | **12.0 — appliqué** (§4) |
| `presence.head_assist.keypoints_used` | 3 points | 5 points | **5 points — appliqué** (§4) |
| `presence.head_assist.enabled` | `false` | `true` (§6) | `false` — **hors périmètre** (§6) |
| `presence.head_assist.pose_model_expected_sha256` | `null` | SHA-256 (§6) | `null` — **hors périmètre** (§6) |

Aucun réglage n'était « déjà appliqué » : la session part de l'état annulé.

### Ligne de base des tests (§0.4)

- **585 tests réussis**, couverture **93,58 %** (seuil exigé 85 %). Aucun échec
  préexistant. `410` fonctions `test_` au total.
- Le chiffre réel est donc **410 fonctions de test** (`585` tests *collectés*,
  l'écart venant de la paramétrisation). Voir §8.

---

## 2. Pourquoi le lot A.1 n'a pas été appliqué tel quel

Le message du commit d'annulation documentait déjà une mesure défavorable aux
trois seuils BoT-SORT. Plutôt que de réappliquer le paquet, chaque seuil a été
**mesuré isolément** (`scripts/measure_tracker_swaps.py`, étendu pour exposer
`--match-thresh` et `--new-track-thresh`, qui n'étaient pas surchargeables).

Protocole : `test/5121204_School_Classroom_1280x720.mp4`, **300 frames**,
`--assumed-fps 25.0` (véritable cadence source : 743 frames à 25 ips).
Objectif : **minimiser** les identifiants techniques créés et les changements
d'identifiant.

### 2.1 Tableau avant/après (lot A.5)

| Variante | `match` | `prox` | `new_track` | ids tech. | personnes | changements d'ID | discontinuités | rejets descripteur |
|---|---|---|---|---|---|---|---|---|
| **production (référence)** | 0,9 | 0,5 | 0,7 | **11** | 7 | **4** | 0 | 36 |
| `match_thresh` 0,75 **seul** | 0,75 | 0,5 | 0,7 | 12 | 7 | 5 | 0 | 35 |
| `proximity_thresh` 0,8 **seul** | 0,9 | 0,8 | 0,7 | 11 | 7 | 4 | 0 | 36 |
| `new_track_thresh` 0,45 **seul** | 0,9 | 0,5 | 0,45 | 13 | **9** | 4 | 0 | 44 |
| **les trois ensemble** (cible du prompt) | 0,75 | 0,8 | 0,45 | **15** | 7 | **8** | 0 | 43 |
| après lots A+B (seuil A.1 inchangés) | 0,9 | 0,5 | 0,7 | **11** | 8 | **3** | 1 | 36 |

> Le clip utilisé diffère de celui du §4.2 du rapport d'origine (le fichier exact
> n'était pas identifiable dans le dépôt). Les chiffres ne sont donc **pas
> directement comparables** à ceux du rapport d'origine ; ils sont en revanche
> comparables entre eux, à vidéo, protocole et nombre de frames identiques.
> Contrôle de déterminisme : la configuration finale a été mesurée **deux fois**,
> avec des compteurs identiques.

### 2.2 Conclusion mesurée, seuil par seuil

| Seuil | Avant | Cible | Décision | Effet mesuré | Compromis |
|---|---|---|---|---|---|
| `match_thresh` | 0,9 | 0,75 | **refusé** | ids 11→12, changements d'ID 4→5 : **dégradation** | Baisser le seuil de coût accepte des appariements à IoU faible, ce qui autorise l'appariement croisé de personnes voisines. Aucun gain observé qui le compense. |
| `proximity_thresh` | 0,5 | 0,8 | **refusé** | **compteurs identiques** : aucun effet mesurable | La justification reste valable en théorie (le ReID natif n'est consulté que si l'IoU est déjà élevé, donc précisément pas quand une personne ressort d'une occlusion). Mais un effet nul ne valide rien : à conserver tel quel tant qu'aucun corpus ne discrimine. |
| `new_track_thresh` | 0,7 | 0,45 | **refusé** | ids 11→13, personnes 7→9, changements d'ID inchangés (4) | C'est le seul levier qui adresse directement le symptôme 2 : une personne partiellement masquée détectée à 0,35–0,6 ne peut jamais créer de piste à 0,7. Mais +2 identités créées sur 12 s **sans** réduction des changements d'identifiant = risque de sur-comptage (nouvelles présences) non départageable sans corpus annoté. Reste `PROVISOIRE`, non appliqué. |

**Pourquoi aucun des trois n'est retenu.** Sur le corpus réellement enregistré
(35 sessions dans `results/`), `IDENTITY_SWAP_CORRECTED` vaut **0** au total et
`TRACK_ID_APPEARANCE_DISCONTINUITY` **3** — le symptôme 1 (inversion
d'identifiants) **n'a jamais été observé**. Les deux seuils qui le ciblent
(`match_thresh`, `proximity_thresh`) ne peuvent donc pas être validés sur les
données disponibles ; le seul clip mesurable produit une dégradation nette pour
le paquet des trois, et une dégradation légère pour `match_thresh` seul. Les
appliquer reviendrait à conclure contre la seule mesure disponible.

Le garde-fou contre les faux positifs que portait `new_track_thresh: 0.7` reste
donc porté, comme déjà documenté, par le filtre géométrique
(`geometry.min_aspect_ratio_wh`, `max_aspect_ratio_wh`, `min_box_height_px`,
`min_box_width_px`) et par `timing.confirmation_seconds`. **Recommandation
explicite** : mesurer `new_track_thresh` seul sur un corpus où une réapparition
réelle après occlusion est annotée, avant tout changement.

---

## 3. Lot B — mesures avant/après (B.6)

Même protocole que §2 (même vidéo, 300 frames, 25 ips, déterminisme vérifié).

| Compteur | Avant (production) | Après (lots A+B) | Lecture |
|---|---|---|---|
| identifiants techniques créés | 11 | 11 | inchangé |
| personnes logiques créées | 7 | 8 | +1 |
| `TECHNICAL_ID_CHANGED` (changements d'identifiant) | 4 | **3** | **amélioration** |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 0 | 1 | le descripteur par bandes est plus sensible au saut d'apparence |
| `IDENTITY_SWAP_CORRECTED` | 0 | 0 | aucun swap à corriger sur ce clip |
| rejets de descripteur | 36 | 36 | inchangé |
| `POSSIBLE_MULTI_PERSON_BOX` | — | 0 sur ce clip | le signal n'est pas exercé (aucune fusion) |
| `PRESENCE_MAINTAINED_BY_HEAD` / `PRESENCE_EXTENSION_EXPIRED` | 0 | 0 | `head_assist` désactivé : chemin non exercé |

**Limite explicite.** Ce clip ne contient **aucune** fusion multi-personnes et
`head_assist` y est désactivé : les compteurs `POSSIBLE_MULTI_PERSON_BOX`,
`PRESENCE_MAINTAINED_BY_HEAD` et `PRESENCE_EXTENSION_EXPIRED` ne peuvent pas y
être mesurés. Ils sont couverts par des **tests d'intégration déterministes**
(§7), pas par cette mesure. Le gain sur les changements d'identifiant (4→3) porte
sur un seul clip de 12 s : il est un indice, **pas** une validation.

---

## 4. Réglages effectivement modifiés

| Réglage | Avant | Après | Effet mesuré | Compromis accepté |
|---|---|---|---|---|
| `reid.long_term.gallery_retention_seconds` | `null` (fenêtre totale = grâce + grâce = **10 s**) | **12.0** (fenêtre totale = 5 + 12 = **17 s**) | Fenêtre > occlusion maximale mesurée (13,5 s) : vérifié par test (`> 13.5`) et par un test de réassociation à 7 s d'occlusion, instant où l'ancienne fenêtre avait déjà libéré la mémoire. | Une identité purgée reste ré-associable plus longtemps → risque accru de réassociation erronée si une personne d'apparence proche réapparaît dans la même zone sous 17 s. Garde-fous inchangés (`similarity_threshold`, `safety_margin`) ; la purge du `person_id` reste journalisée. |
| `presence.head_assist.keypoints_used` | 3 (nez, 2 yeux) | **5** (+ 2 oreilles) | Non mesurable sur ce clip (`head_assist` désactivé). | Aucun : les oreilles sont les points qui survivent à une vue de profil. Coût de calcul nul (mêmes 17 points-clés lus). |
| `reid.long_term.feature_momentum` (nouveau) | **codé en dur `0.85`** dans `identity_manager.py` | **0.85, exposé en configuration**, borné `[0, 1[` | Identique (valeur conservée) mais désormais réglable **et** effectivement lue par le manager (test de câblage). | Aucun changement de comportement ; le seul risque est une valeur mal choisie, d'où la borne qui refuse `1.0` (le descripteur serait figé). |
| `reid.appearance_continuity.freeze_gallery_frames` (nouveau) | inexistant : `record.feature` était mélangé **à chaque frame** | **3** (borne minimale imposée ≥ 3) | Après 5 frames de swap, la référence de galerie est **inchangée** (`np.allclose`) et la correction croisée **aboutit** — test dédié. Sans le gel, la référence contenait 1 − 0,85⁵ ≈ **56 %** de l'apparence de l'autre personne au bout de 5 frames. | Le descripteur ne s'adapte plus pendant le gel : un vrai changement d'apparence (angle, lumière) sur ≤ 3 frames est ignoré par la galerie. Position, hauteur et `last_seen_s` continuent d'être mis à jour. |
| `reid.long_term.descriptor.*` (nouveau) | histogramme H×S 24×8 sur le **crop complet** | 3 bandes sur crop **érodé de 15 %**, poids `[1.0, 1.0, 0.5]`, même 24×8 par bande | Voir §5 : p95 inter passe de **0,840** (ancien) à **0,608** (nouveau). | Dimension du descripteur 192 → 576 (coût CPU négligeable : `cv2.calcHist` reste l'essentiel). Un rognage plus fort écarterait mieux le fond mais amputerait les épaules. |
| `geometry.multi_person_box.head_separation_ratio` (nouveau) | inexistant | **0.4** | Comble l'angle mort du signal par largeur : un test vérifie le déclenchement **dès la première frame**, ce qui était structurellement impossible avant. | Sans effet tant que `presence.head_assist.enabled` est faux (§6) : le seuil ne peut pas être calibré sur données réelles aujourd'hui. |
| `presence.head_assist.max_head_staleness_frames` (nouveau) | inexistant : point tête relu d'une frame arbitrairement ancienne | **1** (frame courante ou précédente) | Test paramétré : tolérance 1 → 1 maintien (puis `OCCULTEE`) ; tolérance 2 → 2 maintiens (puis `OCCULTEE`). Aucun maintien sur point périmé au-delà. | **Voir l'observation §6.3** : cette borne rend `max_presence_extension_seconds` très difficilement atteignable dans le chemin « sans détection ». |

Tous les seuils marqués `PROVISOIRE` le restent : aucun n'a été figé sur un
ensemble de calibration disjoint de l'ensemble de test.

---

## 5. B.4 — Distribution des similarités intra / inter-identités

Mesure : `scripts/measure_descriptor_similarity.py`, mêmes 300 frames, sur
**exactement les mêmes crops** pour les deux descripteurs.
`intra` = même piste technique, deux frames consécutives. `inter` = deux pistes
distinctes sur la même frame.

### 5.1 Nouveau descripteur par bandes (24 × 8 × 3 bandes)

| | n | min | p5 | p25 | médiane | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| **intra** | 762 | 0,461 | **0,911** | 0,976 | 0,987 | 0,993 | 0,997 | 0,999 |
| **inter** | 1019 | 0,007 | 0,089 | 0,251 | 0,344 | 0,420 | **0,608** | 0,887 |

Histogramme sur `[0, 1]` en 10 intervalles de 0,1 :

```
intra : [  0,  0,  0,  0,  1,  0,  1,  4, 30, 726]
inter : [ 68, 106, 208, 328, 173, 79, 41,  9,  7,   0]
```

### 5.2 Descripteur historique (crop complet, un seul histogramme)

| | n | min | p5 | p25 | médiane | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| **intra** | 762 | 0,619 | 0,952 | 0,988 | 0,993 | 0,996 | 0,998 | 0,999 |
| **inter** | 1019 | 0,026 | 0,203 | 0,382 | **0,500** | 0,591 | **0,840** | 0,956 |

Histogramme :

```
intra : [  0,  0,  0,  0,  0,  0,  2,  1,  7, 752]
inter : [ 10, 39, 91, 140, 229, 270, 119, 58, 46,  17]
```

### 5.3 Lecture et valeur proposée

Le nouveau descripteur **sépare** les deux distributions là où l'ancien les
faisait se recouvrir : p95 inter passe de **0,840 à 0,608**, et la médiane inter
de **0,500 à 0,344**. C'était la cause structurelle annoncée : l'écart
inter-personnes était du même ordre que le bruit du descripteur.

**À `similarity_threshold: 0.40` (valeur actuelle), ~25 % des paires
inter-personnes passent encore le seuil** (p75 inter = 0,420). Le seuil n'est donc
probablement plus assez discriminant avec le nouveau descripteur.

**Valeur proposée : `0,65`.** Justification : juste au-dessus du p95 inter
(0,608), donc ≥ 95 % des paires de deux personnes distinctes sont rejetées, tout
en restant très en dessous du p5 intra (0,911), donc la continuité d'une même
personne est préservée avec une large marge. Le milieu géométrique calculé par le
script (0,76) est trop haut : il couperait la queue basse de la distribution intra
(min intra = 0,461), c'est-à-dire précisément les personnes dont l'apparence
change le plus.

**Valeur NON appliquée.** `config/pipeline.yaml` conserve `0.40` et documente la
proposition. Une calibration doit être faite sur un ensemble **disjoint** de
l'ensemble de test avant de figer la valeur (le prompt §B.4 l'exige).

**Limites de cette mesure, à ne pas masquer :**
- un seul clip, une seule scène (salle de cours, caméra fixe) ;
- les paires « inter » sont des pistes techniques distinctes **supposées** être
  des personnes distinctes : un swap non détecté contaminerait l'échantillon
  (risque faible ici, 0 discontinuité sur ce clip lors de la mesure de référence) ;
- les crops proviennent des boîtes du tracker, donc la mesure ne dit rien sur le
  comportement d'un crop issu d'une fusion multi-personnes (ces crops sont de
  toute façon rejetés par la galerie).

---

## 6. Résultats de la section 6 — **NON EXÉCUTÉE**

La section 6 (activation réelle du `head_assist`) n'a **pas** été exécutée : le
périmètre de cette session s'arrête après les lots A et B, conformément à la
règle du prompt (« ne pas commencer la section 6 avant que les lots A et B soient
mesurés et acceptés »).

Par conséquent :
- `presence.head_assist.enabled` reste **`false`** ;
- `models/yolo11s-pose.pt` n'a **pas** été téléchargé et
  `pose_model_expected_sha256` reste **`null`** ;
- **aucune** mesure de `occupancy_observed` / `occupancy_operational`
  `head_assist` activé vs désactivé n'a été produite. Il n'y a donc rien à
  rapporter ici, et rien n'est déclaré acquis.

### 6.1 Prérequis de la section 6 — état

| Prérequis (§6.1) | État |
|---|---|
| B.1 fusionné et couvert par un test | **fait** (§7) |
| B.2 fusionné et couvert par un test | **fait** (§7) |
| `keypoints_used` contient les 5 points | **fait** (§4) |

### 6.2 Prérequis « modèle pose » (§6.2) — non satisfait

`models/yolo11s-pose.pt` est absent. Le dépôt ne contient que `yolo11n.pt`
(détection) et `yolo11s.pt`. Activer le flag **fait échouer `verify_weights`**
(`FileNotFoundError`, `main` retourne 3) : c'est le comportement attendu et il est
correct. Le chemin, la source officielle et la raison d'être du SHA-256 sont
documentés dans le `README`.

### 6.3 Deux écarts importants entre le prompt et le code, découverts pendant ce travail

Ces deux points conditionnent la façon dont la section 6 devra être mesurée.

**(a) `occupancy_observed` inclut les personnes `OCCULTEE` non expirées.**
Le prompt §6 pose : « une personne occultée sort de `occupancy_observed` dès le
passage en `OCCULTEE` ». Ce n'est pas ce que fait le code :
`occupancy_observed` compte les pistes avec `inside_occupancy and state is not
ABSENTE`, et `inside_occupancy` n'est mis à `false` que par `_count_out` (sortie
**confirmée**). Une personne `OCCULTEE` reste donc comptée dans
`occupancy_observed` jusqu'à l'expiration de la grâce, qui la fait passer en
`ABSENTE` et la bascule alors dans `occupancy_uncertain`. La docstring de
`occupancy_observed` dit d'ailleurs le contraire de ce que fait le code.
**Conséquence pour §6.5** : le protocole de mesure décrit par le prompt
(« la personne reste-t-elle dans `occupancy_observed` pendant tout le segment »)
ne mesurera pas ce qu'il croit mesurer. Il faut trancher d'abord lequel des deux
comportements est voulu, puis corriger soit le code, soit la docstring et le
prompt. Ce n'est **pas** corrigé ici : ce serait un changement de sémantique du
comptage, hors du périmètre « lots A et B » et non couvert par une mesure.

**(b) `max_presence_extension_seconds` devient difficilement atteignable.**
Combinaison de B.1 et B.2 :
- avec B.1, une chute de hauteur **stabilisée** redevient fiable au bout de
  `anchor.height_drop_confirm_frames` (3 par défaut) : le maintien par la tête
  sur une piste observée est donc borné à ~2 frames avant réadaptation ;
- avec B.2, un point tête plus vieux que `max_head_staleness_frames` (1) ne
  maintient plus rien dans `_handle_missing`, donc les frames consécutives sans
  détection ne peuvent plus accumuler d'extension.

La borne n'est donc atteignable que par détection **intermittente** (une frame
observée rafraîchit le point tête, la suivante manquante). C'est exactement ce que
fait le test réécrit de `test_head_assist_presence.py`, qui force le cas en
gardant la chute non confirmée (`anchor.height_drop_confirm_frames: 50`). Autrement
dit : le garde-fou est **au moins aussi strict** qu'avant, jamais un maintien
indéfini — l'exigence §6.6 est satisfaite par construction, mais l'effet utile du
réglage est plus étroit que ce que laisse croire son nom.

---

## 7. Tests ajoutés et tests corrigés

**Base : 410 fonctions `test_` / 585 tests collectés → 433 fonctions / 621 tests
collectés.** Couverture 93,33 % (seuil 85 %).

### 7.1 Tests ajoutés

| Lot | Fichier | Test |
|---|---|---|
| B.1 | `tests/test_sitting_person_and_anchor_stability.py` | `test_sitting_person_recovers_reliability_with_and_without_head_assist` (paramétré ×2 : assist activée/désactivée) |
| B.2 | `tests/test_head_assist_presence.py` | `test_head_maintenance_accepts_only_fresh_head_points` (paramétré ×2 : tolérance 1 et 2) |
| B.2 | `tests/test_head_assist_presence.py` | `test_head_maintenance_is_refused_without_any_head_point` |
| B.3 | `tests/test_identity_manager.py` | `test_descriptor_is_frozen_while_appearance_is_discontinuous` |
| B.3 | `tests/test_identity_manager.py` | `test_descriptor_resumes_blending_once_the_freeze_expires` |
| B.3 | `tests/test_swap_correction.py` | `test_la_correction_croisee_reste_possible_apres_cinq_frames_de_swap` |
| B.3 | `tests/test_config_validation.py` | `test_feature_momentum_is_configurable_and_bounded` |
| B.3 | `tests/test_config_validation.py` | `test_feature_momentum_reaches_the_identity_manager_from_the_configuration` |
| B.3 | `tests/test_config_validation.py` | `test_freeze_gallery_frames_is_configurable_with_a_minimum_of_three` |
| B.4 | `tests/test_identity_manager.py` | `test_histogram_extractor_erodes_the_crop_before_describing` |
| B.4 | `tests/test_identity_manager.py` | `test_histogram_extractor_is_banded_and_weights_the_low_band` |
| B.4 | `tests/test_identity_manager.py` | `test_histogram_extractor_refuses_a_crop_eroded_under_the_minimum` |
| B.4 | `tests/test_identity_manager.py` | `test_histogram_extractor_is_built_from_the_descriptor_config` |
| B.4 | `tests/test_config_validation.py` | `test_descriptor_parameters_are_validated` |
| B.4 | `tests/test_config_validation.py` | `test_descriptor_invalid_values_are_rejected` (paramétré ×8) |
| B.5 | `tests/test_multi_person_box.py` | `test_heads_suggest_merge_seuils` (paramétré ×5) |
| B.5 | `tests/test_multi_person_box.py` | `test_deux_tetes_separees_declenchent_le_signal_des_la_premiere_frame` |
| B.5 | `tests/test_multi_person_box.py` | `test_signal_par_tetes_inactif_quand_l_assistance_tete_est_desactivee` |
| B.5 | `tests/test_multi_person_box.py` | `test_une_seule_tete_dans_une_boite_large_ne_declenche_rien` |
| B.5 | `tests/test_multi_person_box.py` | `test_personne_absorbee_reste_en_borne_haute_jamais_purgee_comme_une_sortie` |
| A.2 | `tests/test_long_occlusion_reid.py` | `test_long_occlusion_within_the_12s_retention_is_still_reassociated` |
| A.2 | `tests/test_config_validation.py` | `test_gallery_retention_falls_back_on_the_grace_period_when_omitted` |
| A.2 | `tests/test_long_occlusion_reid.py` | `test_retention_falls_back_on_the_occupancy_grace_when_omitted` |

Aucun test n'a été supprimé ni marqué `skip`.

### 7.2 Tests corrigés — et pourquoi

| Test | Correction | Raison |
|---|---|---|
| `test_config_validation.py::test_gallery_retention_defaults_to_the_occupancy_grace_period` | renommé `test_production_gallery_retention_is_12_seconds_and_covers_long_occlusions` ; il vérifie désormais **12,0** sur la config de production ; le cas de repli est testé **séparément**, dans un manager construit avec `gallery_retention_seconds=None` | L'assertion portait sur la **valeur de production** supposée `None`. Après A.2, la production vaut 12,0 : l'assertion était fausse. Les deux couches de vérification (configuration vs intégration ReID) sont conservées dans **deux fichiers distincts** et n'ont pas été fusionnées — dette pré-existante assumée. |
| `test_long_occlusion_reid.py::test_retention_is_aligned_on_the_occupancy_grace_by_default` | renommé `test_production_retention_covers_the_measured_long_occlusions` ; **12,0** sur la production + fenêtre totale > 13,5 s ; repli testé séparément | Même raison que ci-dessus, dans la couche ReID. |
| `test_long_occlusion_reid.py::test_occlusion_beyond_retention_is_explicit_and_never_silent` | fenêtre recalibrée (`sim.steps(160)` au lieu de `70`) | La fenêtre totale du test passe de 6 s (grâce 3 + rétention 3) à **15 s** (grâce 3 + rétention 12). À 70 frames, la mémoire était encore active : l'assertion « mémoire libérée » était devenue fausse. |
| `test_occupancy_synthetic.py::test_reappearance_after_release_creates_a_new_identity` | durée d'inactivité calculée depuis la configuration (`grâce + rétention`) au lieu de `40` frames en dur | Le compte de frames était calibré sur l'ancienne fenêtre alignée (2 s). Avec 13 s, `REID_MATCH` se produisait légitimement et le test mesurait autre chose. La valeur dérivée de la configuration évite de casser à chaque évolution de seuil. |
| `test_pipeline_integration.py::test_reappearance_too_late_is_rejected` | idem (`window_frames + 10` au lieu de `50` frames) | Même cause. |
| `test_bbox_locker_integration.py::test_the_profile_is_released_once_the_person_is_gone` | idem (`window_frames + 10` au lieu de `22` frames) | La libération du profil suit la libération de l'identité dans la galerie, donc la fenêtre **totale** devenue 13 s. |
| `test_head_assist_presence.py::test_head_assist_extension_expires_after_max_seconds` | chute maintenue **non confirmée** via `anchor.height_drop_confirm_frames: 50` | Le test encodait le bug corrigé par B.1 : il attendait qu'une chute **stabilisée** laisse l'ancre non fiable indéfiniment. Après B.1, une personne assise stabilisée redevient fiable et il n'y a plus rien à maintenir. Le test mesure désormais la borne `max_presence_extension_seconds` sur le seul cas où le maintien dure : la chute non stabilisée. |
| `test_identity_manager.py::test_descriptor_is_blended_and_renormalized` | le changement d'apparence est **modéré** (`unit(1, 0.3, 0)`, similarité ≈ 0,96) au lieu de franc (`unit(0, 1, 0)`, similarité 0) | Le test vérifie le **mélange**. Or un saut franc d'une frame à l'autre est une discontinuité d'apparence, qui **gèle** désormais le descripteur (B.3) : le comportement testé était devenu, à juste titre, celui du gel. Le gel a son propre test dédié ; ce test-ci reste sur le mélange. |

---

## 8. Corrections rapport ↔ code (§7)

| Point | Constat | Correction |
|---|---|---|
| §4.1 « 352 passed » vs §4.7 « 585 tests » | Chiffres incompatibles. **410** fonctions `test_` dans `tests/`, donnant **585** tests collectés (paramétrisation incluse). Le « 585 » du §4.7 est donc le bon ordre de grandeur ; le « 352 » du §4.1 est faux. | Le `README` et ce document portent les chiffres exacts. `docs/rapport_final.md` **n'a pas été modifié** : il est en cours de refonte et la référence fiable est désormais le nombre **mesuré** (410 fonctions / 585 collectés avant cette session ; 433 / 621 après). |
| §4.7 « 5 points-clés » vs config à 3 | Écart réel : la config déclarait `[nose, left_eye, right_eye]`. | Corrigé en A.3. Les **deux** occurrences codées en dur de la valeur par défaut dans `src/config.py` (dataclass `HeadAssistConfig` **et** repli `head_assist_raw.get(...)`) ont été alignées sur les 5 points. Recherche globale faite : il n'existe **pas** de troisième occurrence dans `src/` (les seules autres mentions sont une docstring dans `geometry.py` et la liste `ALLOWED_HEAD_KEYPOINTS`, qui contenait déjà les oreilles). |
| §4.7 « fait et testé » pour l'assistance tête | L'écart est réel : les 4 tests de `test_head_assist_presence.py` injectaient `head_point` **directement**, sans jamais passer par le modèle pose. La suite était verte sans que la fonctionnalité n'ait jamais tourné en conditions réelles. | Toujours vrai : c'est précisément le trou que la **section 6.4** doit combler (test d'intégration avec un vrai tenseur `keypoints`). Non traité ici (section 6 hors périmètre). |
| §6 « une personne occultée sort de `occupancy_observed` dès le passage en `OCCULTEE` » | **Faux** vis-à-vis du code, et la docstring de `occupancy_observed` est elle-même incohérente avec son implémentation. | Documenté en §6.3(a) ci-dessus, **non corrigé** : trancher exige une décision de conception, pas un correctif de seuil. |

---

## 9. Symptômes qui subsistent après ce document

| Symptôme | État après lots A+B | Rattaché à |
|---|---|---|
| 1. Inversion d'identifiants sous occlusion dense | **Subsiste.** Non mesurable sur les données disponibles (0 `IDENTITY_SWAP_CORRECTED`, 3 discontinuités sur 35 sessions). Les seuils de swap n'ont pas été modifiés (mesure défavorable). La correction croisée est **rendue possible** après un swap long (B.3), mais aucun swap réel n'a été observé pour la valider sur données. | **Lot E** (priorité par ancienneté dans la correction de swap) et **lot D** — non exécutés ici. |
| 2. Piste perdue longtemps jamais ré-attribuée | **Partiellement adressé, non validé.** A.2 porte la fenêtre totale à 17 s (> maximum mesuré 13,5 s), mais `new_track_thresh` (qui conditionne la recréation d'une piste) n'a pas pu être validé et reste à 0,7. | **Lot E** pour la contingence ; le levier direct (`new_track_thresh`) reste `PROVISOIRE`. |
| 3. Deux personnes proches partageant une boîte et un identifiant | **Adressé en instrumentation, non mesuré sur données.** Le second signal (têtes) lève l'angle mort « dès la première frame » et la personne absorbée reste en borne haute, mais le signal par têtes est inerte tant que `head_assist` est désactivé (section 6). Aucune fusion n'apparaît sur le clip mesuré. | **Section 6** (activation du modèle pose) avant toute mesure réelle. |
| 4. Assistance tête sans effet observable | **Inchangé : `head_assist` reste désactivé.** Les bugs qui auraient dégradé le cas « personne assise » sont corrigés (B.1, B.2), mais la fonctionnalité n'a jamais tourné pour de vrai. | **Section 6** — non exécutée ici. |
| Prétraitement vidéo (CLAHE, ROI, lecture asynchrone) | Non abordé. | **Lot D** — document séparé, non exécuté. |
| Ancre géométrique tête | Non implémentée, conformément à la décision de conception de la section 2 du prompt (les pieds et la ligne validée restent l'unique ancre de comptage). | **Hors périmètre définitif.** |

---

## 10. Ce qui n'a pas pu être validé, faute de corpus

Écrit explicitement, comme le prompt l'exige :

1. **Les trois seuils BoT-SORT (`match_thresh`, `proximity_thresh`,
   `new_track_thresh`) ne sont pas validés.** Le seul corpus disponible ne
   contient aucun swap et un seul clip mesurable ; il montre une dégradation pour
   `match_thresh` seul et pour le paquet des trois. Aucun n'est appliqué.
2. **`similarity_threshold: 0.40` n'est pas recalibré.** La valeur `0,65` est
   *proposée* avec ses deux distributions, pas figée : la calibration doit se faire
   sur un ensemble disjoint de l'ensemble de test.
3. **Le gain de stabilité d'identité après lots A+B** (changements d'identifiant
   4 → 3) porte sur **un seul clip de 12 s** : indice, pas validation.
4. **Effet réel du `head_assist`, du signal multi-personnes par têtes et du
   descripteur par bandes en conditions de salle de cours** : non mesuré. Aucun
   de ces trois mécanismes n'était exercé sur le clip disponible (`head_assist`
   désactivé, aucune fusion détectée).
5. **Distribution des similarités** : un seul clip, une seule scène, sans
   vérité-terrain d'identité ; les paires « inter » sont supposées appartenir à
   des personnes distinctes.

---

# 11. Session 0 — commande de lancement, catalogue du corpus, contrôle de déterminisme

> **Mention de portée (CLAUDE.md §2).** Tout ce qui suit est mesuré sur un
> corpus de **7 vidéos, 5 726 frames, 178,6 s cumulées**, contenant au total
> **1 correction de swap signalée, 0 fusion signalée, 22 purges après grâce et
> 1 perte > 5 s suivie de redétection**. Le phénomène visé par le plan
> (inversion d'identité sous occlusion dense) est donc présent mais **rare** :
> aucun lot ne pourra être « validé » sur ce corpus au sens fort. Les seuils de
> signification à respecter sont ceux du §11.7.
>
> **Aucun fichier de code n'a été modifié dans cette session.** Les seules
> écritures dans le dépôt sont ce document, le §0 de `CLAUDE.md`, et le
> téléchargement de `models/yolo11n.pt` (non versionné, voir `.gitignore`).

## 11.1 Commande de lancement du pipeline sur un fichier vidéo

Établie en lisant `build_argument_parser()` (`src/main.py:111-150`) et vérifiée
par `python src/main.py --help`. **À inscrire une fois pour toutes dans
`CLAUDE.md` §2** :

```bash
python src/main.py --mode comptage --source test/<video>.mp4
```

`--config` vaut `config/pipeline.yaml` par défaut. `--source` est **obligatoire
pour un fichier** : `source.default` vaut `"0"` dans la configuration, c'est
à-dire l'index de la caméra (`config/pipeline.yaml:15`).

Options utiles, telles qu'elles existent réellement :

| Option | Effet réel |
|---|---|
| `--no-show` | Supprime la prévisualisation **pendant le traitement**. **N'enlève pas** la sélection manuelle de la ligne. |
| `--write-video` | Écrit `annotated.mp4` dans le dossier de session. |
| `--max-frames N` | Arrêt après N frames (diagnostic). |
| `--session-id ID` | Fixe le nom du dossier de session (défaut : horodatage UTC). |
| `--output-root DIR` | Racine des résultats (défaut `results/`). |
| `--conf`, `--iou`, `--imgsz`, `--model` | Surcharges YOLO, revalidées par `override_config`. |
| `--inside-side {positive,negative}` | Orientation **initiale proposée**, inversible en direct par la touche « I ». |

Sorties, dans `results/<session_id>/` : `events.jsonl`, `summary.json`,
`calibration.json`, `config_resolved.yaml`.

Codes de retour : `2` configuration invalide, `3` poids ou tracker introuvable,
`4` calibration annulée ou impossible, `5` source illisible.

### 11.1.1 Cette commande n'est pas automatisable — constat bloquant

**Le pipeline ne possède aucun chemin d'exécution non interactif.** Au démarrage,
`perform_line_selection` (`src/main.py:356-369` → `calibrate`,
`src/main.py:300-352`) ouvre une fenêtre OpenCV sur la première image et attend
**deux clics de souris puis la touche « C »**. La docstring de `calibrate` est
explicite : le repli historique `if no_show: return (0.0, 0.6), (1.0, 0.6)` a été
**délibérément supprimé**, et `--no-show` sans interface graphique lève
`CalibrationUnavailable` (code 4) au lieu de retomber sur une ligne par défaut.

Sur cette machine `display_available()` renvoie `True` : la commande ci-dessus
fonctionne, mais elle **bloque** sur l'attente des clics. Elle ne peut donc pas
servir à cataloguer 7 vidéos ni à rejouer une baseline deux fois.

Ce n'est pas un défaut à corriger — c'est une décision de conception assumée
(« sans ligne validée par l'opérateur, le comptage ne démarre pas »). C'est en
revanche une **contrainte de protocole** : toute mesure avant/après des lots 1 à
7 devra soit être conduite à la main par la personne responsable, soit passer
par un harnais du type décrit au §11.2. Le dépôt avait déjà tranché dans ce
sens : `scripts/measure_tracker_swaps.py` et `scripts/diagnose_occlusion.py`
documentent tous deux, en tête de fichier, qu'ils s'exécutent « sans ligne
virtuelle — la sélection manuelle empêcherait toute mesure automatisée ».

## 11.2 Harnais de catalogue employé

Les deux scripts existants ne suffisaient pas : ils n'instancient que
`IdentityManager`, donc ils ne peuvent produire ni `POSSIBLE_MULTI_PERSON_BOX`
ni `PURGE` ni les compteurs d'occupation, qui vivent dans `OccupancyManager`.

Le catalogue a donc été produit par un harnais de session (hors dépôt, dans le
répertoire temporaire de la session) qui rejoue **le pipeline réel** :
`YOLO.track` avec `src/configs/custom_botsort.yaml` → `Detection` →
`OccupancyManager.process_frame` → `ListSink`. C'est la boucle de
`src/main.py:692-843`, sans le rendu et sans la sélection de ligne.

Deux écarts assumés, à connaître avant de lire les chiffres :

1. **Ligne de catalogue conventionnelle.** `OccupancyManager` refuse d'être
   construit sans ligne (`src/occupancy_manager.py:146-155`). Le harnais lui
   fournit une ligne horizontale fixe — `(0,05 ; 0,60) → (0,95 ; 0,60)`,
   `inside_side = negative` — **identique pour les 7 vidéos**. Elle n'a aucune
   valeur opérationnelle.
   **Conséquence directe : `IN`, `OUT`, `NEW`, `INCONSISTENT_STATE` et
   `occupancy_operational` dépendent de ce tracé arbitraire et ne sont PAS
   utilisés pour classer le corpus.** Le classement ne repose que sur les
   compteurs d'identité, qui sont indépendants de la ligne.
2. **Base de temps vidéo.** Le harnais calcule `timestamp_s = frame_index / fps`
   là où `src/main.py:780` utilise `time.perf_counter() - start_s`. Ce choix
   est mesuré, pas supposé : voir §11.7, où les deux bases sont comparées.

## 11.3 Environnement et poids de modèle

| Élément | Valeur constatée |
|---|---|
| Python | 3.11.9 |
| `ultralytics` | 8.4.126 |
| `torch` | 2.13.0+cpu — **`cuda_available = False`** |
| OpenCV | 5.0.0 |
| `model.device` (config) | `cpu` |

**Poids YOLO.** `models/yolo11n.pt` était **absent** du dépôt (le dossier
`models/` n'existait pas ; `.gitignore` exclut `models/` et `*.pt`). Téléchargé
via `ultralytics.utils.downloads.attempt_download_asset("yolo11n.pt")`.

- Source exacte : `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt`
- Taille : 5 613 764 octets
- **SHA-256 : `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`**
- Ce condensat est **identique** à `model.expected_sha256`
  (`config/pipeline.yaml:26`). `verify_weights` passe donc sans modification de
  configuration.

**Poids pose.** `models/yolo11s-pose.pt` reste absent et
`pose_model_expected_sha256` reste `null` : non nécessaire en session 0
(`presence.head_assist.enabled: false`), à traiter au lot 6.

## 11.4 Catalogue du corpus (CLAUDE.md §2)

Dossier des vidéos de test : **`test/`** (exclu du versionnement par
`.gitignore`). Sept fichiers, tous en `.mp4`.

### 11.4.1 Caractéristiques des fichiers

| Vidéo | Frames | FPS source | Durée | Résolution | Poids |
|---|---|---|---|---|---|
| `fort_occ.mp4` | 1 057 | 50,00 | 21,1 s | 1920×1080 | 13,9 Mo |
| `fort_occ2.mp4` | 328 | 30,00 | 10,9 s | 1920×1080 | 12,3 Mo |
| `fort_occ3.mp4` | 330 | 29,97 | 11,0 s | 3840×2160 | 29,5 Mo |
| **`fort_occ4.mp4`** | **1 055** | **30,00** | **35,2 s** | **1280×720** | 32,1 Mo |
| `fort_occ5.mp4` | 1 996 | 30,00 | 66,5 s | 1280×720 | 60,5 Mo |
| `rare_occ.mp4` | 216 | 24,00 | 9,0 s | 3840×2160 | 26,3 Mo |
| **`rare_occ2.mp4`** | **744** | **29,97** | **24,8 s** | **1280×720** | 15,6 Mo |

### 11.4.2 Mesure du pipeline en l'état, pleine longueur

Chaque vidéo a été traitée **intégralement**, une seule fois, configuration de
production inchangée.

| Vidéo | Détections | Identités créées | ids techniques | `TECHNICAL_ID_CHANGED` | `TRACK_ID_APPEARANCE_DISCONTINUITY` | `IDENTITY_SWAP_CORRECTED` | `POSSIBLE_MULTI_PERSON_BOX` | `PURGE` | `OCCLUDED` | `REID_AMBIGUOUS` | `REID_MATCH` | perte > 5 s redétectée | ips traitées |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fort_occ` | 15 289 | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 4,31 |
| `fort_occ2` | 5 250 | 21 | 21 | 0 | 1 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 3,93 |
| `fort_occ3` | 2 551 | 17 | 19 | 2 | 0 | **1** | 0 | 3 | 38 | 0 | 2 | 0 | 3,83 |
| **`fort_occ4`** | 5 680 | 18 | 15 | **5** | **1** | 0 | 0 | **13** | **49** | **5** | 5 | 0 | 4,47 |
| `fort_occ5` | 14 902 | 11 | 13 | 4 | 1 | 0 | 0 | 4 | 23 | 0 | 4 | **1** | 4,28 |
| `rare_occ` | 1 296 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3,11 |
| **`rare_occ2`** | 2 553 | 7 | 5 | 0 | 0 | 0 | 0 | 2 | 6 | 0 | 0 | 0 | 4,06 |

### 11.4.3 Décompte des phénomènes et classement

Le §2.3 impose de classer sur « swaps + fusions + pertes > 5 s », **réellement
comptés**, et non sur l'étiquette du nom de fichier.

Traduction en compteurs, justifiée :

- **swaps** → `IDENTITY_SWAP_CORRECTED` (swap réellement détecté et corrigé) ;
- **fusions** → `POSSIBLE_MULTI_PERSON_BOX` ;
- **pertes > 5 s** → `PURGE` **+** les écarts > 5 s entre deux observations d'un
  même `person_id`. Une `PURGE` n'est émise qu'après expiration de la fenêtre de
  grâce, et `timing.grace_period_seconds` vaut **5,0 s**
  (`config/pipeline.yaml:292`) : toute purge est donc, par construction, une
  perte d'au moins 5 s. `mark_purged` (`src/identity_manager.py:1273-1311`) la
  journalise en `purge_kind="technical"`, `is_exit=False`.

| Vidéo | Étiquette | **Score strict §2.3** | Proxys d'identité (`chgID` + discontinuités + ambigus) | Verdict |
|---|---|---|---|---|
| `fort_occ` | fort | **0** | 0 | étiquette **non confirmée** |
| `fort_occ2` | fort | **0** | 1 | étiquette **non confirmée** |
| `fort_occ3` | fort | 4 | 2 | confirmée |
| **`fort_occ4`** | fort | **13** | **11** | confirmée — **maximum du corpus** |
| `fort_occ5` | fort | 5 | 5 | confirmée |
| `rare_occ` | rare | **0** | 0 | confirmée |
| `rare_occ2` | rare | 2 | 0 | **écart signalé** (2 purges) |

**Écarts étiquette / décompte, signalés comme l'exige le §2.3 :**

1. **`fort_occ.mp4` et `fort_occ2.mp4` ne présentent aucun phénomène compté**
   malgré leur préfixe `fort_occ` — alors que ce sont les deux vidéos les plus
   peuplées du corpus (15 289 et 5 250 détections). Forte densité de personnes
   n'y signifie pas instabilité d'identité. Ce sont exactement les clips sur
   lesquels les deux sessions précédentes auraient de nouveau conclu à tort.
2. **`rare_occ2.mp4` contient 2 purges** (donc 2 pertes ≥ 5 s) et 6
   occultations, alors que son préfixe annonce une occlusion faible. L'écart
   est faible mais réel.
3. **Aucun fichier n'a été renommé**, conformément à la consigne.

### 11.4.4 Vidéos retenues — choix figé pour tous les lots

| Rôle | Vidéo | Justification |
|---|---|---|
| **Mesure principale (occlusion dense)** | **`test/fort_occ4.mp4`** | Maximum du corpus sur **les deux** critères : score strict §2.3 = 13 (13 purges) et proxys d'identité = 11 (5 changements d'identifiant technique, 1 discontinuité d'apparence, 5 ré-identifications ambiguës). C'est aussi la seule vidéo où `REID_AMBIGUOUS` est non nul, et celle où l'écart identités créées (18) / identifiants techniques (15) est le plus marqué. Ni la plus longue (`fort_occ5`, 66,5 s) ni la première par ordre alphabétique. |
| **Contrôle de non-régression (occlusion faible)** | **`test/rare_occ2.mp4`** | Retenue plutôt que `rare_occ.mp4` : 744 frames contre 216, résolution 1280×720 identique à la vidéo principale (donc charge de calcul comparable), et 24,8 s contre 9,0 s. Ses 2 purges sont signalées au §11.4.3 et constituent la ligne de base à ne pas dégrader. `rare_occ.mp4` (0 phénomène) reste disponible comme second contrôle. |

**Ce choix est figé.** Aucun lot ne doit en changer, sous peine de rendre les
tableaux avant/après incomparables entre lots.

### 11.4.5 Trois observations à verser aux lots concernés

Elles ne sont pas des conclusions de lot : ce sont des faits mesurés en session 0
qui orientent la lecture des lots à venir.

**(a) La galerie d'apparence ne se remplit presque jamais — lot 3.**
`REID_DESCRIPTOR_REJECTED` est massif sur tout le corpus : 248, 291, 242, 260 et
**375** écritures refusées pour respectivement 16, 21, 17, 18 et **11** identités.
En regard, `REID_MATCH` vaut 0 sur quatre vidéos sur sept. Le ReID long terme est
donc, en pratique, **structurellement inopérant sur ce corpus** : il ne peut pas
ré-associer une personne dont il n'a jamais pu mémoriser l'apparence. Les
raisons exactes des rejets sont mesurées au §11.5 (vidéo principale) et au
§11.6 (vidéo de contrôle).

**(b) Le filtre géométrique s'exécute avant la détection de fusion — lot 5.**

> **CORRECTION APPORTÉE AU LOT 0 (§13.5).** La conclusion de ce paragraphe —
> « aucune fusion n'est signalée sur le corpus » — est **fausse pour le système
> livré**. Elle vient d'un harnais de session 0 auquel les stabilisateurs
> n'étaient pas branchés. Mesuré par le pipeline réel, `fort_occ4` produit
> **52 `POSSIBLE_MULTI_PERSON_BOX`**, pas 0. Reste exact ci-dessous : l'ordre
> des opérations, et les 235 rejets de boîtes larges sur `fort_occ.mp4`.

`check_detection_geometry` (`src/geometry.py:413-438`) est appliqué en tête de
`process_frame` (`src/occupancy_manager.py:365-399`) et rejette toute boîte de
ratio W/H > `max_aspect_ratio_wh` = 2,5, **avant** que le chemin de détection de
fusion (`check_box_width_growth`, qui émet `POSSIBLE_MULTI_PERSON_BOX`) ne soit
atteint. Or une boîte contenant deux personnes assises côte à côte est
précisément une boîte anormalement **large**.

Mesure sur `fort_occ.mp4`, qui concentre les rejets du corpus : **les 235 rejets
sont tous `aspect_ratio_out_of_range`, et tous portent sur des boîtes trop
LARGES** (ratio W/H supérieur à 2,5, jusqu'à **3,85**) — aucun rejet pour boîte
trop petite ou trop verticale. Ces 235 détections sont donc supprimées avant
d'avoir pu être examinées comme fusions possibles.

**Ce que cette mesure ne dit pas** : qu'il s'agisse effectivement de fusions.
Un ratio de 3,85 peut aussi être un artefact du détecteur. Départager exige
l'annotation du §3. Ce qui est établi, c'est l'**ordre des opérations** : sur
cette vidéo, le signal de fusion ne pouvait structurellement pas se déclencher.

À nuancer pour la vidéo principale : **`fort_occ4` ne produit aucun rejet
géométrique**. Sur elle, le filtre n'explique donc pas le `POSSIBLE_MULTI_PERSON_BOX`
à 0 — l'angle mort d'historique décrit au diagnostic §4.1.7 reste la piste.

**(c) La marge sur le plancher de FPS est très faible — §4 et lot 6.**
Voir §11.8.


## 11.5 Ligne de base détaillée de la vidéo principale (`fort_occ4`)

Mesure instrumentée, base de temps vidéo, pleine longueur.

| Grandeur | Valeur |
|---|---|
| Frames traitées | 1 055 (35,2 s) |
| Détections suivies | 5 680 |
| Identités logiques créées | 18 |
| Identifiants techniques distincts | 15 |
| `TECHNICAL_ID_CHANGED` | 5 |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 1 |
| `IDENTITY_SWAP_CORRECTED` | 0 |
| `POSSIBLE_MULTI_PERSON_BOX` | 0 |
| `OCCLUDED` | 49 |
| `PURGE` | 13 |
| `REID_MATCH` / `REID_NEW` / `REID_AMBIGUOUS` | 5 / 18 / 5 |
| `REID_DESCRIPTOR_REJECTED` | 260 |
| `ANCHOR_UNRELIABLE` | 50 |
| `occupancy_operational` en fin de séquence | 13 |
| `occupancy_observed` en fin de séquence | 3 |

**Les 13 purges, dans le détail** : toutes ont pour raison `grace_expired` et
pour dernier état `OCCULTEE`. Impact sur l'occupation : **10 `uncertain`**,
3 `none`. Durées de vie des identités purgées, en secondes : 0,33 / 1,07 / 1,57 /
2,10 / 4,17 / 7,37 / 8,80 / 9,43 / 9,50 / 11,93 / 13,90 / 20,27 / 23,77.

Cela confirme par la mesure ce que le code annonçait : une purge n'arrive
**qu'après** expiration de la grâce de 5 s, donc chaque purge est bien une perte
d'au moins 5 s. `purge_kind` vaut toujours `technical` et `is_exit` toujours
`false` — aucune purge n'a produit de sortie fantôme.

**Écarts entre observations d'une même identité** : 37 au total, dont 4 au-delà
de 1 s et **aucun au-delà de 3 s**. Autrement dit, sur cette vidéo, une personne
retrouvée l'est vite ; une personne perdue longtemps n'est jamais retrouvée —
elle est purgée. Les deux mécanismes ne se recouvrent pas.

**Répartition de la présence** (frames observées par identité logique, sur
1 055) : 980, 794, 681, 462, 442, 404, 343, 270, 267, 263, 222, 201, 126, 69,
64, 48, 33, 11. Une moitié des identités vit moins de 300 frames : c'est le
symptôme de fragmentation, quantifié.

**Rejets d'écriture en galerie** : 260, dont **238 `low_confidence`** et
22 `edge_truncated`. Avec `gallery_min_confidence: 0.70`
(`config/pipeline.yaml:135`) face à `model.confidence: 0.10`
(`config/pipeline.yaml:33`), **91 % des rejets viennent du seul seuil de
confiance**. La galerie d'apparence est donc quasiment vide sur une scène de
salle de classe, où les personnes assises et partiellement masquées sortent
précisément à faible confiance. Le ReID long terme ne peut pas ré-associer ce
qu'il n'a jamais pu mémoriser : c'est la donnée d'entrée du **lot 3**.

## 11.6 Ligne de base de la vidéo de contrôle (`rare_occ2`)

Mesure instrumentée, base de temps vidéo, pleine longueur. **C'est la ligne à ne
pas dégrader** lors des lots suivants (§7 : un `new_track_thresh` abaissé ne doit
pas faire exploser les créations de piste sur scène calme).

| Grandeur | Valeur |
|---|---|
| Frames traitées | 744 (24,8 s) |
| Détections suivies | 2 553 |
| Identités logiques créées | **7** |
| Identifiants techniques distincts | 5 |
| `TECHNICAL_ID_CHANGED` | 0 |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 0 |
| `IDENTITY_SWAP_CORRECTED` | 0 |
| `POSSIBLE_MULTI_PERSON_BOX` | 0 |
| `OCCLUDED` | 6 |
| `PURGE` | 2 (`grace_expired`, `grace_expired_in_progress` — toutes deux `uncertain`) |
| `AMBIGUOUS_CROSSING` | 1 |
| `REID_DESCRIPTOR_REJECTED` | 76 (63 `low_confidence`, 13 `edge_truncated`) |
| Écarts d'observation | 4, **aucun au-delà de 1 s** |

## 11.7 Contrôle de déterminisme (CLAUDE.md §2.1)

Baseline rejouée sur `fort_occ4`, sans rien changer. **Quatre rejouages** : deux
en base de temps vidéo, deux en base de temps murale — celle qu'utilise
réellement `src/main.py:780` (`time.perf_counter() - start_s`).

Aucune graine aléatoire n'est fixée dans `src/` ni dans `scripts/`
(`grep -rn "seed\|torch.manual\|np.random.seed"` : aucun résultat). Un éventuel
déterminisme est donc empirique, non garanti par construction. `torch` tourne
sur 2 threads, en CPU.

### 11.7.1 Base de temps vidéo — écart nul

| Compteur | run 1 | run 2 | run du catalogue | écart |
|---|---|---|---|---|
| Tous les compteurs d'événements | identiques | identiques | identiques | **0** |
| `persons_created` | 18 | 18 | 18 | 0 |
| `technical_id_changes` | 5 | 5 | 5 | 0 |
| `PURGE` | 13 | 13 | 13 | 0 |
| `INCONSISTENT_STATE` | 10 | 10 | 10 | 0 |
| ips de traitement | 4,247 | 4,121 | 4,474 | **≠** |

**Trois exécutions, aucun écart sur aucun compteur** — y compris entre des runs
dont la vitesse de traitement différait de 8 %. Le déterminisme est robuste à la
charge machine dès lors que la base de temps est celle de la vidéo.

### 11.7.2 Base de temps murale — écart non nul

| Compteur | wall run 1 | wall run 2 | écart |
|---|---|---|---|
| ips de traitement | 4,176 | 3,042 | −27 % |
| durée murale | 252,7 s | 346,9 s | +94 s |
| `TECHNICAL_ID_CHANGED` | 7 | 8 | **+1** |
| `IN` | 4 | 5 | **+1** |
| `NEW` | 9 | 8 | **−1** |
| `REID_MATCH` | 7 | 8 | **+1** |
| `REID_AMBIGUOUS` | 2 | 1 | **−1** |
| identifiants techniques | 5 | 6 | **+1** |
| `REID_DESCRIPTOR_REJECTED` | 260 | 261 | **+1** |
| `ANCHOR_UNRELIABLE` | 50 | 49 | **−1** |
| `INCONSISTENT_STATE` | 152 | 189 | **+37** |
| `OCCUPANCY_SNAPSHOT` | 223 | 260 | **+37** |

**La source du bruit est identifiée** : le FPS de traitement, qui dépend de la
charge de la machine, entre directement dans toutes les durées exprimées en
secondes. Entre les deux runs il a varié de 4,176 à 3,042 ips sans qu'aucun
paramètre ne change.

### 11.7.3 Effet de la base de temps elle-même — et non plus seulement du bruit

Comparaison de la **même vidéo**, **même configuration**, seule la base de temps
change :

| Compteur | base vidéo | base murale | commentaire |
|---|---|---|---|
| `PURGE` | **13** | **0** | disparition totale de la traçabilité des pertes |
| `INCONSISTENT_STATE` | 10 | 152–189 | ×15 à ×19 |
| `OCCUPANCY_SNAPSHOT` | 36 | 223–260 | `snapshot_interval_seconds: 1.0` appliqué à 252 s murales au lieu de 35 s vidéo |
| `TECHNICAL_ID_CHANGED` | 5 | 7–8 | |
| identifiants techniques | 15 | 5–6 | |
| `IN` / `NEW` | 6 / 7 | 4–5 / 8–9 | **le comptage lui-même change** |

**Explication mécanique du `PURGE` à 0**, vérifiée dans le code :

- la grâce FSM est convertie en **frames** par `TimingConfig.frame_budget`
  (`src/config.py:440-457`) : `grace_period_frames = round(5,0 × fps)` = **150
  frames** ;
- la fenêtre de galerie, elle, reste en **secondes** dans
  `within_gallery_window` (`src/identity_manager.py:1175-1179`) :
  `grace_period_seconds + purge_retention_seconds` = 5 + 12 = **17 s** ;
- en base murale, 17 s de temps mural ≈ 2,4 s de temps vidéo ≈ **73 frames**, à
  4,2 ips traitées. La galerie libère donc l'identité **avant** que la FSM ait
  atteint ses 150 frames de grâce ;
- `release_expired` retire alors la piste de `self.tracks`
  (`src/occupancy_manager.py:631-633`), si bien que `_handle_missing` ne la voit
  plus jamais et ne peut plus la purger. `mark_purged` n'est jamais appelée,
  l'événement `PURGE` n'est jamais émis.

C'est exactement le scénario que la docstring de `within_gallery_window` dit
vouloir empêcher : « La galerie n'est jamais plus stricte que la machine à
états : sinon elle libérerait une identité avant que le FSM ait pu la purger, ce
qui supprimerait l'événement `PURGE` et le passage en occupation incertaine. »
Le garde-fou est écrit, mais il suppose que les deux mécanismes partagent la
même base de temps — ce que l'implémentation actuelle ne garantit pas.

Le commentaire de `src/occupancy_manager.py:632` (« La purge a déjà été
journalisée lors du passage à ABSENTE : la libération de mémoire n'est donc
jamais silencieuse ») est donc **faux dans cette configuration** : la libération
y est parfaitement silencieuse.

### 11.7.4 Seuils de signification à appliquer dans tous les lots suivants

Conclusion opérationnelle du §2.1, à reprendre tel quel dans chaque livrable :

| Protocole de mesure | Bruit run-à-run mesuré | Écart minimal significatif |
|---|---|---|
| **Base de temps vidéo** (recommandé) | **0** sur tous les compteurs | **≥ 1** |
| Base de temps murale (`src/main.py` en l'état) | ±1 sur les compteurs d'identité et de comptage ; ±37 sur `INCONSISTENT_STATE` et `OCCUPANCY_SNAPSHOT` | ≥ 2 pour l'identité ; ≥ 38 pour les deux autres |

**Recommandation de protocole** : conduire toutes les mesures avant/après des
lots 1 à 7 en base de temps vidéo. C'est le seul régime où le bruit est nul,
donc le seul où un écart de 1 sur `IDENTITY_SWAP_CORRECTED` ou
`POSSIBLE_MULTI_PERSON_BOX` — compteurs dont les valeurs de base sont 0 et 1 —
puisse signifier quelque chose. En base murale, ces compteurs sont noyés dans le
bruit et **aucun lot ne serait démontrable**.

Ce point n'est pas une anticipation du lot 1 : aucune ligne de code n'a été
modifiée. C'est la mesure de bruit que le §2.1 exige avant toute mesure
avant/après, et elle conclut sur le protocole, pas sur un correctif.

## 11.8 Plancher de FPS — recalcul demandé au §4

Le §4 demande de recalculer le plancher de 3,0 ips avec la profondeur réelle de
la zone et de le confronter à `timing.confirmation_seconds`.

**Confrontation à `confirmation_seconds`** : elle vaut **0,5 s**
(`config/pipeline.yaml:291`), soit moins que les ≈ 1,15 s de séjour supposées.
Ce n'est donc **pas** elle qui domine : la contrainte dominante reste bien le
nombre d'observations dans la zone, comme le §4 le supposait. Le plancher de
3,0 ips n'a pas à être relevé de ce fait.

**Profondeur réelle de la zone** : inconnue. Elle n'est renseignée nulle part
dans le dépôt et dépend de l'installation caméra. **Le recalcul complet reste
donc à faire par la personne responsable** ; le plancher de 3,0 ips est conservé
comme valeur de travail, non validée.

**FPS mesurés en l'état, modèle pose *inactif*, CPU, `imgsz: 960` :**

| Vidéo | ips traitées | Marge sur 3,0 ips |
|---|---|---|
| `fort_occ4` (principale) | 4,47 / 4,25 / 4,12 / 4,00 | +33 % à +49 % |
| `fort_occ4`, base murale, run lent | **3,04** | **+1 %** |
| `rare_occ` (3840×2160) | **3,11** | **+4 %** |
| `fort_occ3` (3840×2160) | 3,83 | +28 % |
| `fort_occ5` | 4,28 | +43 % |
| `rare_occ2` (contrôle) | 4,06 / 4,14 | +35 % |

**Alerte à verser au lot 6.** La marge est déjà mince : deux mesures sont à
3,0–3,1 ips, c'est-à-dire **au plancher**. Or le critère du §4 exige ≥ 3,0 ips
**avec le modèle pose actif**, et `yolo11s-pose.pt` est sensiblement plus lourd
que `yolo11n.pt` utilisé ici. Sur cette machine (CPU seul, `cuda_available =
False`), **le critère de FPS du lot 6 a de fortes chances d'être violé**. Il
faudra soit une accélération matérielle, soit une baisse de `image_size`, soit
une révision explicite du critère — décision à prendre par la personne
responsable, pas en cours de lot.

**Contrainte relative du §4** (≥ 80 % de la baseline) : la baseline de
`fort_occ4` étant de **4,47 ips**, le plancher relatif est de **3,58 ips**. Il
est plus contraignant que le plancher absolu de 3,0 ips : c'est lui qui devra
être opposé aux lots.

## 11.9 Écarts relevés en chemin (CLAUDE.md §9)

| Point | Constat | Conséquence |
|---|---|---|
| **Corpus des sessions précédentes introuvable** | Le §2.1 et le §5.1 de ce document mesurent sur `test/5121204_School_Classroom_1280x720.mp4`. Ce fichier **n'existe plus** dans `test/`, qui contient exclusivement les 7 fichiers `fort_occ*` / `rare_occ*` catalogués ci-dessus. | **Aucun chiffre des §2 à §5 de ce document n'est reproductible ni comparable au présent catalogue.** Les tableaux avant/après des lots 1 à 7 repartent de la ligne de base du §11.5, pas de ceux-là. Le commentaire de `config/pipeline.yaml:56-58`, qui cite ce fichier à l'appui du refus des trois seuils BoT-SORT, repose donc sur une mesure non rejouable. |
| **Nombre de tests** | `grep -rc "^def test_" tests/` → **433** fonctions de test réparties sur **30** fichiers. | Concorde avec le chiffre « 433 après » annoncé au §8. Le §8 reste exact ; rien à corriger. La suite n'a **pas** été exécutée dans cette session : aucun code n'ayant été modifié, il n'y avait rien à revalider. |
| **Poids du modèle absents du dépôt** | `models/` n'existait pas. Sans `models/yolo11n.pt`, `verify_weights` échoue (`FileNotFoundError`, code 3) et **le pipeline ne démarre pas du tout**. | Prérequis d'installation non documenté dans `CLAUDE.md` §2. Le SHA-256 obtenu correspond à celui attendu : voir §11.3, à reporter dans `docs/rapport_final.md` §4 à la clôture. |
| **`POSSIBLE_MULTI_PERSON_BOX` = 0 sur tout le corpus** | Aucune fusion signalée sur 5 726 frames, alors que le symptôme 1 du §1 la décrit comme un problème persistant. | Soit les fusions n'existent pas dans ce corpus, soit le détecteur ne les voit pas. Le diagnostic §4.1.7 documente déjà un angle mort : `check_box_width_growth` a besoin d'un historique de largeur **avant** la fusion, donc deux personnes assises côte à côte dès la première frame ne déclenchent jamais le signal. **Départager les deux demande une annotation humaine** (§3) — c'est précisément ce que la vérité terrain doit trancher. |

## 11.10 Ce qui reste à faire avant le lot 1

Les trois lignes du §0 de `CLAUDE.md` ne sont pas toutes levées par cette session.

| Étape §0 | État à l'issue de cette session |
|---|---|
| **Catalogue du corpus (§2)** | **Fait.** §11.4, vidéos figées au §11.4.4. |
| **Contrôle de déterminisme (§2.1)** | **Fait.** §11.7, seuils de signification au §11.7.4. |
| **Vérité terrain (§3)** | **NON FAITE — bloquante.** Voir ci-dessous. |
| **Critères d'acceptation validés (§4)** | **NON VALIDÉS.** Voir ci-dessous. |

### 11.10.1 Vérité terrain — demande explicite

Le §3 est formel : l'annotation ne doit **pas** être remplacée par une
estimation automatique produite par le pipeline testé. Rien dans cette session ne
peut s'y substituer — tous les chiffres ci-dessus sont des compteurs émis par le
pipeline lui-même, donc des proxys.

Ce qui est demandé, et que je ne peux pas produire seul :

- un fichier `tests/fixtures/ground_truth_fort_occ4.json`, versionné ;
- portant sur un segment de 30 à 60 s de **`test/fort_occ4.mp4`** — la vidéo
  fait 35,2 s, **le segment peut donc être la vidéo entière** ;
- pour chaque seconde : le nombre de personnes réellement présentes dans la zone
  comptée, et l'identité stable de chacune (`P1`, `P2`, …) ;
- les segments d'occlusion ≥ 3 s marqués (utiles au lot 6).

**Aide au choix du segment, à partir de la mesure** : si un sous-segment doit
être préféré à la vidéo entière, les 13 purges du §11.5 en donnent les points
d'entrée — ce sont les instants où le pipeline a perdu une identité pendant plus
de 5 s. Les identités les plus fragmentées (11, 33, 48, 64 et 69 frames
observées sur 1 055) sont les meilleures candidates à l'inspection visuelle.

Tant que ce fichier n'existe pas, `id_switches_reels`, `fragments_par_personne`,
`erreur_comptage_max` et `fusions_reelles` **ne sont pas calculables**, et
aucun lot ne peut être déclaré accepté au sens du §4.

### 11.10.2 Critères d'acceptation — à confirmer

Le tableau du §4 reste **`NON VALIDÉ`** : ses valeurs sont proposées, pas
confirmées. Deux d'entre elles appellent une décision à la lumière de cette
session :

1. **`id_switches_reels` ≤ 1** — la baseline demandée est maintenant partiellement
   mesurable : 5 `TECHNICAL_ID_CHANGED` et 1 discontinuité d'apparence sur
   `fort_occ4`. Mais ce sont des proxys : la valeur réelle exige le §11.10.1.
2. **FPS ≥ 3,0 ips avec modèle pose actif** — voir l'alerte du §11.8 : ce
   critère est probablement inatteignable sur la machine actuelle. Mieux vaut le
   trancher maintenant qu'au lot 6.

### 11.10.3 Question de protocole à trancher

Le §11.1.1 établit que le pipeline officiel ne peut pas tourner sans opérateur.
Il faut choisir, **avant le lot 1**, comment les mesures avant/après seront
conduites :

- **(a)** mesures à la main par la personne responsable, avec la même ligne
  cliquée à chaque fois — au risque d'une variabilité de tracé qui s'ajoutera au
  bruit du §11.7 ;
- **(b)** mesures par un harnais versionné dans `scripts/`, sur le modèle du
  §11.2, avec une ligne de calibration lue depuis un fichier figé — ce qui
  supprime la variabilité de tracé, mais suppose d'ajouter un script au dépôt
  (donc une modification, hors périmètre de cette session).

Je recommande **(b)**, pour la raison mesurée au §11.7 : le bruit n'est nul
qu'en conditions strictement reproductibles, et une ligne re-cliquée à chaque
mesure ne l'est pas. Mais c'est une décision de la personne responsable, pas la
mienne.

---

# 12. Vérité terrain de `fort_occ4` — reçue

Fichier versionné : **`tests/fixtures/ground_truth_fort_occ4.json`**.

Images d'appui : `annotation/t01.jpg` … `t35.jpg`, une par seconde — **non
versionnées**. Elles montrent les mêmes personnes que les vidéos du corpus et
relèvent donc du même régime que `test/` (`.gitignore`, voir
`docs/politique_confidentialite.md`). Le fichier de vérité terrain y renvoie
par son champ `image`, mais il se lit sans elles : `present`, `count` et
`notes` sont autonomes. Les images sont conservées hors dépôt par la personne
responsable ; il faut les lui demander pour revérifier une annotation seconde
par seconde.

Il lève l'étape bloquante du §3, demandée au §11.10.1.

## 12.1 Contenu et contrôle de cohérence

| Élément | Valeur |
|---|---|
| Vidéo annotée | `test/fort_occ4.mp4`, 30,0 ips, 35,17 s, 1280×720 |
| Couverture | **35 secondes, de 0 à 34** — la vidéo entière, pas un extrait |
| Étiquettes humaines | **13** (`P1` … `P13`), chacune décrite par un repère visuel (couleur de vêtement, rang, côté) |
| Segments d'occlusion documentés | 4 |
| `count` ≠ `len(present)` | **aucune incohérence** |
| Étiquettes déclarées mais jamais utilisées | aucune |
| Étiquettes utilisées mais non déclarées | aucune |

**Effectif réel, seconde par seconde** : 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 9,
9, 10, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13,
13, 13. La salle se remplit de 4 à 13 personnes, sans jamais se vider : aucune
sortie réelle sur la séquence.

**Segments d'occlusion annotés :**

| Étiquette | Début | Fin | Durée | Cause notée |
|---|---|---|---|---|
| `P7` | 19 s | 25 s | **6 s** | s'assoit pendant que les autres cherchent leur place |
| `P9` | 20 s | 24 s | **4 s** | `P10` devant `P9` |
| `P12` | 30 s | 34 s | **4 s** | `P4` devant `P12` |
| `P8` | 14 s | 16 s | 2 s | plusieurs personnes devant (`P6`, `P2`, `P9`) |

Trois segments sur quatre dépassent 3 s et relèvent donc directement du lot 6
(§7.5). Le quatrième (`P8`, 2 s) est en dessous du seuil demandé au §3 — il est
conservé tel quel, il ne gêne rien.

**Deux champs restent à compléter** : `annotateur` et `date` valent
`A RENSEIGNER`. Sans conséquence sur les mesures, mais à renseigner pour la
traçabilité.

**Deux grandeurs distinctes, toutes deux présentes dans le fichier** :
`count` et `present` comptent les personnes **présentes** ; la **visibilité**,
elle, est consignée seconde par seconde dans le champ `notes`, en clair
(« n'est plus visible », « en dehors de la vue », « pas encore visible »). Les
deux diffèrent sensiblement — 13 présentes contre 11 visibles à la dernière
seconde — et ce sont deux compteurs distincts du pipeline qui s'y comparent.
Voir §12.2.1.

## 12.2 Première confrontation aux mesures de session 0

Confrontation partielle, sur les seules grandeurs déjà mesurées au §11.5.
**Ce n'est pas encore le tableau complet des métriques du §3** — voir §12.3.

### 12.2.1 « Présente » et « visible » ne sont pas la même grandeur

Distinction qui conditionne toute la lecture de ce paragraphe :

- à la seconde 34, **13 personnes sont présentes** dans la salle — c'est ce que
  le champ `count` enregistre ;
- mais **11 seulement sont visibles** à l'image ; parmi elles, plusieurs sont
  **très floues et de petite taille** (rangs du fond, personnes assises
  partiellement masquées).

**La visibilité est annotée** : elle figure dans le champ `notes`, seconde par
seconde, en langage clair. Les deux personnes non visibles à la fin de la
séquence y sont nommées et datées :

| Étiquette | Dernière mention de visibilité | Notes d'appui | Cause annotée |
|---|---|---|---|
| **P9** | non visible à partir de **s25** | « P9 en dehors de la vue » (s25), « n'est plus visible » (s26), « pas encore visible » (s27, s29, s31–s34) | « sorti de la zone de caméra **mais reste dans la salle** » (s33) |
| **P13** | non visible à partir de **s32** | « P13 n'est plus visible » (s30), réapparaît (s31), « P13 et P9 ne sont pas visibles » (s32) | non précisée |

Règle de lecture appliquée : un état « non visible » **persiste** jusqu'à
mention explicite du contraire. C'est ce qui donne 13 − 2 = **11 visibles** à
s34, P13 n'étant jamais annoncé comme redevenu visible après s32.

Les notes distinguent par ailleurs « X occulte Y » (occultation **partielle**,
Y reste visible) de « Y n'est plus visible » (disparition **totale**). Sans
cette distinction, s34 — « P4 occulte P12, P12 occulte P3, P2 occulte P10 » —
compterait 4 personnes non visibles au lieu de 2.

Or les deux compteurs du pipeline ne se comparent pas au même référentiel
(`CLAUDE.md` §6) :

| Compteur du pipeline | Doit être comparé à | Valeur de référence à la seconde 34 |
|---|---|---|
| `occupancy_operational` | personnes **présentes** | **13** |
| `occupancy_observed` | personnes **visibles** | **11** |
| `occupancy_uncertain` | présentes non visibles | **2** |

**Ce qui manque n'est pas l'annotation mais sa forme** : la visibilité est
consignée en texte libre, pas dans un champ structuré. Elle se lit sans
ambiguïté, mais un calcul automatique de `erreur_comptage_max` sur les 35
secondes suppose de la dériver des `notes`. C'est une extraction, **pas une
reprise de l'annotation** : l'information est déjà là.

### 12.2.2 Comparaison, avec le bon référentiel pour chaque compteur

> **RÉSERVE APPORTÉE AU LOT 0 (§13.5).** Les valeurs de pipeline de ce tableau
> viennent du harnais de session 0, dont le lot 0 a montré qu'il divergeait du
> système livré. `occupancy_operational` (13) et `occupancy_observed` (3) sont
> **confirmés identiques** sur le pipeline réel ; les conclusions ci-dessous
> tiennent donc. Mais le relevé seconde par seconde reste à refaire avec
> `scripts/measure_corpus.py` avant d'opposer `erreur_comptage_max` aux
> critères du §4.


| Grandeur | Vérité terrain | Pipeline (§11.5) | Écart |
|---|---|---|---|
| Personnes **présentes** en fin de séquence | **13** | `occupancy_operational` = **13** | **0 — exact** |
| Personnes **visibles** en fin de séquence | **11** | `occupancy_observed` = **3** | **−8** |
| Présentes non visibles | **2** | `occupancy_uncertain` = **10** | **+8** |
| Identités logiques créées sur la séquence | 13 personnes | **18** | **+5 identifiants surnuméraires** |

Quatre enseignements, tous soutenus par la mesure :

1. **`occupancy_operational` est exact** : 13 contre 13. Le bilan officiel ne se
   trompe pas — cohérent avec le fait que la vérité terrain ne comporte aucune
   sortie réelle et que le pipeline n'a produit aucun `OUT` fantôme (les 13
   purges sont toutes `is_exit=false`, §11.5).
2. **`occupancy_observed` s'effondre** : 3 personnes vues alors que **11** sont
   visibles. `erreur_comptage_max` vaut donc **au moins 8** sur la dernière
   seconde, contre une cible de ≤ 1 au §4. C'est précisément le compteur que
   l'assistance tête doit protéger (`CLAUDE.md` §6).
3. **L'incertitude est surévaluée d'autant** : le pipeline classe 10 personnes
   en `occupancy_uncertain` alors que **2** seulement le sont réellement. Il
   déclare « perdues » 8 personnes qui sont à l'image. L'invariant
   `observed + uncertain == operational` est bien respecté (3 + 10 = 13) : ce
   n'est pas un bug d'invariant, c'est une erreur de classement.
4. **18 identités pour 13 personnes** : au moins 5 identifiants surnuméraires,
   cohérent avec les 5 `TECHNICAL_ID_CHANGED` du §11.5. La fragmentation est
   confirmée par une source indépendante du pipeline.

### 12.2.3 Pourquoi ces 8 personnes sont perdues — hypothèse à tester

Le fait que les personnes manquantes soient décrites comme **« très floues et
petites »** recoupe deux mesures indépendantes de la session 0 :

- **238 rejets `low_confidence`** en écriture de galerie sur cette vidéo
  (§11.5), contre `gallery_min_confidence: 0.70` — une personne floue produit
  une détection à faible confiance ;
- **`gallery_min_crop_height_px: 32`** (`config/pipeline.yaml:136`) — une
  personne petite produit un crop sous le seuil.

Les deux verrous frappent donc exactement la population décrite. **Ce n'est
pas encore une conclusion** : la mesure dit que les rejets existent et que les
personnes manquantes sont floues et petites, elle n'établit pas encore que les
premiers causent les secondes. Le vérifier demande de relever, pour chaque
personne visible non comptée, la confiance et la taille de sa boîte — ce qui
relève de l'outillage évoqué au §12.3 et au §11.10.3.

## 12.3 Ce qui reste à calculer

Les quatre métriques du §3 ne sont **pas** toutes calculables en l'état — mais
la donnée manquante vient du **pipeline**, pas de l'annotation :

| Métrique | Calculable ? | Ce qu'il manque |
|---|---|---|
| `erreur_comptage_max` | **presque** — borne inférieure de 8 établie ci-dessus | côté vérité terrain, **rien** : la visibilité est annotée seconde par seconde (§12.2.1). Il manque `occupancy_observed` **relevé seconde par seconde**, que le run du §11.5 n'a pas conservé (état final seulement) : un rejouage instrumenté suffit. |
| `fragments_par_personne` | non | l'association entre `person_id` du pipeline et étiquette humaine |
| `id_switches_reels` | non | idem |
| `fusions_reelles` | non | idem |

Autrement dit : l'annotation fournit à la fois **qui est présent** et **qui est
visible**, seconde par seconde. Ce qui manque est le **rattachement** entre les
`person_id` du pipeline et les étiquettes `P1` … `P13`. Trois des quatre
métriques en dépendent ; la quatrième n'attend qu'un relevé par seconde.

Ce rattachement est le travail à faire avant le lot 1 — il est mécanisable
(rejouer la séquence en journalisant la boîte de chaque `person_id` par seconde,
puis apparier avec les étiquettes sur les images `annotation/t*.jpg`), mais il
suppose d'ajouter un script au dépôt, ce qui rejoint la décision de protocole
laissée ouverte au §11.10.3.

Le §0 de `CLAUDE.md` est mis à jour en conséquence : la vérité terrain est
**fournie**, les métriques dérivées restent **à outiller**.

---

# 13. Lot 0 — outillage de mesure

> **Mention de portée.** Ce lot ne corrige aucun symptôme : il rend les mesures
> des lots 1 à 7 possibles et représentatives du système livré. Sa seule
> production chiffrée est la baseline FPS du §4 et le bruit run-à-run du
> pipeline réel.
>
> **Il corrige aussi plusieurs chiffres du §11**, obtenus en session 0 par un
> harnais qui divergeait du pipeline : voir §13.5. C'est précisément ce que le
> critère d'acceptation du lot devait détecter.

## 13.1 Vidéos utilisées

| Rôle | Vidéo | Frames | Durée | Occlusion |
|---|---|---|---|---|
| Mesure principale | `test/fort_occ4.mp4` | 1 055 | 35,2 s | dense (13 purges, 5 chgID en session 0) |
| Contrôle de non-régression | `test/rare_occ2.mp4` | 744 | 24,8 s | faible (2 purges) |

Le lot 0 ne modifiant aucun seuil métier, la mesure de non-régression sur
`rare_occ2` n'a pas été rejouée : il n'y a pas d'avant/après à comparer. Elle
reprend au lot 1, avec le harnais versionné.

## 13.2 Ce qui a été fait

### 13.2.1 Drapeau `--line` (§0.1)

`src/main.py` accepte désormais `--line x1,y1,x2,y2`, en coordonnées
normalisées. Comportement, conforme au §0.1 :

| Situation | Comportement | Code |
|---|---|---|
| `--line` absent, mode interactif | **inchangé** : première image, deux clics, « C » | — |
| `--line` fourni | ligne explicite, validée, tracée ; la fenêtre n'est pas ouverte | 0 |
| `--no-show` **sans** `--line` | refus immédiat, message explicite | **6** |
| `--line` malformé ou hors bornes | refus avant ouverture de session de comptage | 2 |
| source illisible malgré un `--line` valide | erreur de source, inchangée | 5 |

**Aucun repli automatique n'a été réintroduit.** L'ancienne ligne implicite à
60 % reste supprimée : sans `--line`, rien n'est deviné, le pipeline refuse.

**Validation unique.** `line_from_argument` appelle `geometry.validate_line`,
c'est-à-dire exactement la fonction qu'utilise `LineSelection.confirm` pour la
ligne cliquée. Il n'existe pas de second chemin de contrôle, et un test vérifie
que les deux provenances produisent la même `ValidatedLine`.

**Code de sortie 6.** `NonInteractiveLineRequired` hérite de
`CalibrationUnavailable` — tout code qui rattrapait l'exception historique
fonctionne encore — mais porte un code distinct : « il manque `--line` » et
« aucun écran disponible » ne se corrigent pas de la même façon, et les
confondre ferait chercher au mauvais endroit.

**Traçabilité.** `LINE_VALIDATED` et `calibration.json` portent désormais
`line_origin` (`argument` ou `operator_clicks`). Sans ce champ, deux sessions
aux compteurs identiques seraient indiscernables quant à l'origine de leur
ligne.

### 13.2.2 Harnais versionné `scripts/measure_corpus.py` (§0.2)

Enveloppe mince, **sans aucune logique de comptage** : elle lance `src/main.py`
en sous-processus, puis relit `summary.json` et `events.jsonl` que le pipeline a
lui-même écrits. Ce qu'elle mesure est donc, par construction, ce que le système
livré fait.

```bash
python scripts/measure_corpus.py --videos test/ --line 0.05,0.6,0.95,0.6 \
    --repeat 3 --out results/baseline.jsonl
```

Schéma de sortie, **stable** pour les lots 1 à 7 : `source`, `run_index`,
`git_sha`, `timestamp`, `time_base`, `line`, `exit_code`, `frames_processed`,
`processing_fps` (+ médiane et minimum), `duration_s`, `event_counts` (le
dictionnaire complet), `technical_ids`, `persons_created`,
`technical_id_changes`, `appearance_discontinuities`, `descriptor_rejections`,
`long_gaps_count`, `occupancy_operational`, `occupancy_observed`,
`occupancy_uncertain`, `occupancy_range`, `total_in`, `total_out`, `total_new`.

`measure_tracker_swaps.py` et `diagnose_occlusion.py` sont **inchangés**, ni
fusionnés ni réécrits.

### 13.2.3 Deux compteurs ajoutés à `summary.json`

Le schéma du §0.2 exige `technical_ids` et `persons_created`, que le résumé de
session ne publiait pas. Ils sont **accumulés au fil des frames** par la boucle
d'exécution, et non relus dans l'état final de `IdentityManager` : celui-ci
supprime les identités expirées de `records` comme de `technical_to_person`
(`release_expired`, `src/identity_manager.py:1314-1327`), si bien qu'un décompte
pris à la fin ignorerait précisément les identités perdues — celles qui nous
intéressent. Aucun attribut interne n'est lu, aucun calcul métier n'est déplacé.

## 13.3 Fichiers modifiés — dont un hors périmètre

| Fichier | Nature | Dans le périmètre du §0.4 ? |
|---|---|---|
| `src/main.py` | `--line`, analyse, refus non interactif, traçabilité, totaux d'identité | oui (point d'entrée, arguments) |
| `scripts/measure_corpus.py` | nouveau harnais | oui (`scripts/`) |
| `tests/test_line_argument.py` | nouveau | oui |
| `tests/test_main_units.py`, `tests/test_main_e2e.py` | attentes ajustées (§13.6.2) | oui |
| **`src/events.py`** | **une ligne : `line_origin` ajouté aux champs optionnels de `LINE_VALIDATED`** | **NON — écart signalé** |

**Sur `src/events.py`.** Le §0.4 limite le lot au point d'entrée, aux arguments
et à `scripts/` ; `events.py` n'y figure pas. La modification est d'une ligne,
purement déclarative : elle ajoute `line_origin` à la liste des champs
**optionnels** de `LINE_VALIDATED`. Aucune logique n'est touchée, aucun champ
existant n'est modifié, et `validate_event` n'aurait de toute façon pas rejeté
un champ non déclaré — le journal aurait simplement porté un champ hors schéma.
J'ai préféré le déclarer plutôt que de laisser un champ non spécifié dans un
journal qui sert de référence à toutes les mesures. **C'est un écart au
périmètre, réversible en une ligne** : il revient à la personne responsable de
le confirmer ou d'en demander le retrait.

`OccupancyManager`, `IdentityManager` et la configuration du tracker n'ont
**pas** été touchés, conformément au §0.4.

## 13.4 Baseline FPS du §4 — figée

Trois rejouages complets de `fort_occ4` par le pipeline réel, rendu compris.

| Rejouage | Frames | FPS moyen | FPS médian | FPS minimum | Durée |
|---|---|---|---|---|---|
| 1 | 1 055 | 3,701 | 3,911 | 2,426 | 298,5 s |
| 2 | 1 055 | 3,826 | 3,829 | 3,481 | 293,4 s |
| 3 | 1 055 | 3,825 | 3,921 | 2,793 | 282,1 s |

**Médiane retenue : `3,825 ips`**, inscrite dans `CLAUDE.md` §4.

Conséquences :

- le plancher **relatif** du §4 (≥ 80 %) vaut **3,06 ips**, donc il est **plus
  contraignant** que le plancher absolu de 3,0 ips : c'est lui qu'il faut
  opposer aux lots ;
- la marge réelle avant le plancher absolu est de **0,8 ips**, modèle pose
  **inactif**, sur CPU seul. Le lot 6 activera `yolo11s-pose.pt`, sensiblement
  plus lourd : **le critère FPS a de fortes chances d'être violé**, et c'est une
  décision à prendre avant ce lot, pas pendant ;
- le `fps_min` descend à **2,426 ips** sur un des rejouages : le plancher absolu
  est déjà franchi ponctuellement, sans modèle pose.

**Réserve sur « en base de temps vidéo ».** La mesure était demandée en base de
temps vidéo ; elle n'a pas pu l'être. `src/main.py:913` horodate sur
`time.perf_counter()` et le pipeline n'offre aucune base de temps vidéo : la lui
donner **est le lot 1**, pas le lot 0. Le FPS de traitement, lui, ne dépend pas
de la base de temps interne — c'est un rapport frames / durée réelle — donc la
baseline ci-dessus est valide telle quelle. Seuls les **compteurs** du §13.5
sont affectés par cette réserve.

## 13.5 Vérification du lot (§0.3) — le critère n'est pas satisfait, et c'est le résultat

Le §0.3 demande de rejouer `fort_occ4` avec le harnais versionné et de
**retrouver exactement** les compteurs de la session 0 (§11.5).

| Compteur | Session 0 (harnais ad hoc) | Lot 0 (pipeline réel) | Écart |
|---|---|---|---|
| `POSSIBLE_MULTI_PERSON_BOX` | **0** | **52** | **+52** |
| `PURGE` | 13 | 17 – 19 | +4 à +6 |
| `TECHNICAL_ID_CHANGED` | 5 | 8 | +3 |
| `technical_ids` | 15 | 21 | +6 |
| `persons_created` | 18 | 17 | −1 |
| `INCONSISTENT_STATE` | 10 | 5 | −5 |
| `IN` / `NEW` | 6 / 7 | 2 / 7 | −4 / 0 |
| `occupancy_operational` | 13 | 13 | **0** |
| `occupancy_observed` | 3 | 3 | **0** |
| `OCCLUDED` | 49 | 49 | **0** |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 1 | 1 | **0** |

**Deux causes identifiées, toutes deux dans le harnais de session 0 :**

1. **Les stabilisateurs n'étaient pas branchés.** `config/pipeline.yaml` active
   `stabilization.use_bbox_locker: true`, et `src/main.py:729-734` passe le
   `BBoxHeightLocker` à `OccupancyManager`. Le harnais de session 0 construisait
   `OccupancyManager(config, event_sink, line)` **sans** locker ni
   stabilisateur. Or le verrouillage de hauteur modifie les dimensions de boîte
   que voit `check_box_width_growth` — d'où **52 fusions signalées au lieu de 0**.
2. **Le budget de frames était calculé sur le mauvais FPS.** `src/main.py:935`
   recalcule `frame_budget(fps_estimator.require_fps())` avec le FPS **mesuré**
   (≈ 3,8 ips), soit une grâce de `round(5,0 × 3,8)` ≈ **19 frames**. Le harnais
   de session 0 figeait `frame_budget(30)` — le FPS **source** — soit
   **150 frames**. Les purges ne pouvaient pas tomber aux mêmes instants.

**Conséquence directe : l'observation (b) du §11.4.5 est invalidée.** Elle
concluait, à partir des 0 `POSSIBLE_MULTI_PERSON_BOX` du corpus, que le signal
de fusion ne se déclenchait jamais. C'était un artefact du harnais : sur le
pipeline réel, `fort_occ4` produit **52 fusions signalées**, le signal
fonctionne. Ce qui reste vrai du §11.4.5(b) est le constat d'ordre des
opérations — le filtre géométrique précède la détection de fusion — et les
235 rejets de boîtes larges mesurés sur `fort_occ.mp4`. Ce qui est faux est la
conclusion qu'aucune fusion n'est détectée.

**Le critère §0.3 n'est donc pas « retrouvé exactement », et c'est exactement ce
que le lot devait révéler.** La fiche demandait d'expliquer tout écart avant
d'aller plus loin : c'est fait, les deux causes sont nommées et chiffrées. La
leçon est celle qu'énonçait le §0.2 : un harnais qui rejoue `process_frame` de
son côté est un second système, et il diverge sans prévenir.

**Les chiffres du §11 restent valides comme comparaison relative entre vidéos du
corpus** — toutes mesurées avec le même harnais — **mais ils ne décrivent pas le
système livré**. La référence pour les lots 1 à 7 est désormais le tableau
ci-dessus et le §13.4, pas le §11.5.

### 13.5.1 Bruit run-à-run du pipeline réel

Trois rejouages identiques, mêmes arguments, même ligne :

| Compteur | run 1 | run 2 | run 3 | Bruit |
|---|---|---|---|---|
| `technical_ids`, `persons_created`, `technical_id_changes` | 21, 17, 8 | 21, 17, 8 | 21, 17, 8 | **0** |
| `appearance_discontinuities`, `descriptor_rejections` | 1, 268 | 1, 268 | 1, 268 | **0** |
| `POSSIBLE_MULTI_PERSON_BOX`, `OCCLUDED`, `REID_MATCH` | 52, 49, 8 | 52, 49, 8 | 52, 49, 8 | **0** |
| `occupancy_operational` / `observed` / `uncertain` | 13 / 3 / 10 | 13 / 3 / 10 | 13 / 3 / 10 | **0** |
| `IN` / `OUT` / `NEW` | 2 / 0 / 7 | 2 / 0 / 7 | 2 / 0 / 7 | **0** |
| `PURGE` | 19 | 18 | 17 | **±2** |
| `STATE_TRANSITION` | 132 | 131 | 130 | **±2** |
| `OCCUPANCY_SNAPSHOT` | 264 | 261 | 253 | **±11** |
| `long_gaps_count` | 1 | 3 | 0 | **±3** |

**C'est nettement meilleur que ce que la session 0 laissait craindre.** Le §11.7
mesurait, en base murale, un bruit de ±1 sur les compteurs d'identité ; ici
**ils sont strictement stables sur trois rejouages**. Seuls quatre compteurs
bougent, tous sensibles à la durée murale de la session.

**Seuils de signification pour les lots 1 à 7**, à opposer à tout avant/après :

| Compteur | Écart minimal significatif |
|---|---|
| Identité, occupation, fusions, IN/OUT/NEW | **≥ 1** |
| `PURGE`, `STATE_TRANSITION` | **≥ 3** |
| `long_gaps_count` | **≥ 4** — et jamais seul, voir ci-dessous |
| `OCCUPANCY_SNAPSHOT` | **≥ 12** (compteur de cadence, sans portée métier) |

**`long_gaps_count` est le compteur le moins fiable du schéma** : 1, puis 3,
puis 0 sur trois rejouages identiques. Il est dérivé des événements du journal,
dont la granularité dépend de la durée murale. Il est publié pour la complétude
du schéma, mais **aucune conclusion de lot ne doit reposer sur lui seul**. Le
lot 1, qui traite la base de temps, devrait le stabiliser : à revérifier alors.

## 13.6 Tests

**Suite complète : `639 passed, 1 skipped`, couverture `93,33 %`** (seuil exigé
85 %). Le test ignoré (`test_calibration_window.py:227`) l'était déjà avant ce
lot : il exige un serveur X pour vérifier un comportement Qt.

Décompte réel (`grep -rc "^def test_" tests/`) : **445** fonctions de test sur
**31** fichiers, contre 433 sur 30 avant ce lot.

### 13.6.1 Tests ajoutés — `tests/test_line_argument.py`

Trois groupes, calqués sur les exigences du §0.1 :

- **analyse syntaxique** : forme valide, espaces tolérés, cinq formes malformées
  (trois valeurs, cinq valeurs, chaîne vide, non numérique, mauvais séparateur),
  quatre valeurs refusées (hors bornes haut et bas, ligne dégénérée, ligne plus
  courte que `min_length_ratio`), et source illisible signalée dès la
  construction de la ligne ;
- **mode non interactif** : refus sans `--line` ; l'exception reste un
  `CalibrationUnavailable` ; avec `--line`, la fenêtre n'est pas sollicitée ;
  sans `--no-show` ni `--line`, le mode interactif reçoit exactement les mêmes
  paramètres qu'auparavant ;
- **équivalence** : une ligne cliquée et la même ligne passée en argument
  produisent des `ValidatedLine` identiques (points, orientation, résolution,
  longueur). C'est ce test qui rend le harnais légitime : mesurer avec `--line`
  revient à mesurer ce qu'un opérateur aurait obtenu en cliquant.

### 13.6.2 Tests corrigés — cinq, aucun supprimé ni ignoré

| Test | Pourquoi son attente était devenue fausse | Ajustement |
|---|---|---|
| `test_main_units.py::test_no_show_avec_affichage_conserve_la_selection_manuelle`, renommé `test_no_show_sans_ligne_refuse_meme_avec_un_affichage` | Il vérifiait que `--no-show` ouvre **tout de même** la fenêtre de sélection. Le §0.1 fait de `--no-show` un mode non interactif réel, où `--line` est obligatoire : la fenêtre ne doit plus s'ouvrir. | Le test vérifie désormais le refus **et** que `select_line` n'est jamais appelée. L'exigence de fond — aucune ligne n'est jamais devinée — est inchangée ; seule la manière de refuser change. |
| `test_main_units.py::test_no_show_sans_interface_graphique_annonce_l_erreur` | Il exigeait un message mentionnant « interface graphique » et « deux clics ». Le refus intervient maintenant plus tôt et pour une autre cause : il manque `--line`. L'absence d'écran n'est plus l'obstacle, puisque `--line` permet de s'en passer. | Assertions portées sur le nouveau message (`--line`, « non interactif », « devin »). L'exception attendue reste `CalibrationUnavailable`, ce qui vérifie au passage que la hiérarchie d'exceptions est préservée. Le cas « pas d'écran **sans** `--no-show` » reste couvert, inchangé, par `test_main_e2e.py::test_sans_affichage_refuse_de_demarrer`. |
| `test_main_e2e.py::test_no_show_sans_affichage_refuse_de_demarrer`, renommé `test_no_show_sans_ligne_refuse_de_demarrer` | Il attendait le code 4 (« calibration indisponible »). Le refus « il manque `--line` » porte un code dédié, 6. | Code attendu porté à 6, assertions de message mises à jour. Tout le reste est conservé : aucun comptage, pas de `calibration.json`, et la même séquence d'événements `SESSION_START` / `LINE_CALIBRATION_UNAVAILABLE` / `SESSION_END`. |
| `test_main_units.py::test_source_inouvrable_produit_une_erreur_explicite` | Il utilisait `--no-show` seulement pour éviter l'interactivité. Le refus « il manque `--line` » tombe désormais **avant** l'ouverture de la source : le chemin visé — source illisible, code 5 — n'était plus atteint. | Ajout de `--line 0.05,0.6,0.95,0.6`. L'intention est préservée et le test devient plus net : il éprouve la source, non l'absence de ligne. |
| `test_main_e2e.py::test_source_inouvrable_pendant_la_selection` | Même cause. | Même ajustement. |

Aucun test n'a été supprimé, aucun n'a été passé en `skip`.

## 13.7 Ce qui reste non résolu

1. **Le critère d'acceptation §0.3 n'est pas satisfait** (§13.5). Les deux
   causes sont identifiées et chiffrées ; la fiche prévoit elle-même de refaire
   cette vérification après le lot 1, qui change la base de temps. C'est alors
   que la baseline définitive sera établie.
2. **La ligne n'est pas publiée dans `config_resolved.yaml`**, contrairement au
   §0.1. Elle l'est dans `calibration.json` et dans `LINE_VALIDATED`, avec sa
   provenance. La publier dans la configuration résolue exigerait d'ajouter les
   deux points à `LineConfig` (`src/config.py`), dont la docstring précise
   qu'ils n'y figurent **volontairement** pas — et `config.py` est hors du
   périmètre du §0.4. Point laissé ouvert, à trancher.
3. **`long_gaps_count` n'est pas fiable** (§13.5.1) : ±3 sur trois rejouages
   identiques. À réévaluer après le lot 1.
4. **La modification de `src/events.py` sort du périmètre du lot** (§13.3) et
   attend confirmation ou retrait.
5. **Les métriques de vérité terrain du §3 ne sont pas rejouées ici.** Le lot 0
   ne change aucun seuil métier : il n'y a pas d'avant/après à produire. Mais
   les chiffres d'`occupancy_observed` du §12.2 proviennent du harnais de
   session 0, dont le §13.5 montre qu'il divergeait du pipeline. **Ils devront
   être recalculés avec le pipeline réel avant d'être opposés aux critères du
   §4.** Le relevé par seconde reste à refaire dans ces conditions.

---

# 14. Lot 1 — base de temps, `track_buffer` en secondes, seuil NMS

> **Accepté par la personne responsable le 2026-09-21**, avec trois réserves
> inscrites et non bloquantes : `track_buffer_seconds: 15.0` reste `PROVISOIRE`
> et son effet propre n'est pas isolé (§14.8) ; le chemin caméra de la
> conversion n'est couvert qu'unitairement (§14.10.5) ; `bootstrap_frames`
> reste exprimé en frames (§14.10.8, dette à traiter au lot 2).

> **Mention de portée.** Mesuré sur `test/fort_occ4.mp4` (mesure principale,
> 1 055 frames, 35,2 s) et `test/rare_occ2.mp4` (non-régression), ligne figée
> `--line 0.05,0.6,0.95,0.6` (`CLAUDE.md` §2). Le corpus contient **13
> personnes annotées, 4 segments d'occlusion documentés dont 3 de plus de 3 s**
> (§12). Les métriques d'**identité** (`id_switches_reels`,
> `fragments_par_personne`, `fusions_reelles`) ne sont **pas** publiées ici : le
> rattachement `person_id` ↔ étiquettes est une proposition non validée, et sa
> validation est reportée au lot 2 (décision de la personne responsable,
> 2026-09-21). Ce lot se mesure donc sur les compteurs internes et sur les
> métriques de **comptage** de la vérité terrain.

## 14.1 Outillage ajouté avant le lot (commits `2f15649`, `5aeb689`)

| Élément | Rôle |
|---|---|
| `src/second_trace.py` + `src/main.py --trace-per-second` | Relevé de l'état du pipeline à la première frame de chaque seconde **vidéo**, plus une image par seconde. Lecture seule : aucune logique de comptage. |
| `scripts/gt_matching.py` | Propose le rattachement `person_id` ↔ `P1…P13`, chaque ligne portant sa méthode et sa confiance ; **refuse** de publier les métriques d'identité tant que le fichier ne porte pas `statut: valide`. |
| `tests/fixtures/trace_baseline_fort_occ4.jsonl` | Relevé de référence **avant** le lot 1. |
| `tests/fixtures/gt_matching_fort_occ4.json` | Proposition de rattachement, **non validée** : 8 `person_id` sur 17 rattachés, 5 étiquettes orphelines, 2 rattachements automatiques contredits par l'inspection des découpes. |

La seconde vidéo est dérivée de l'**index de frame** (`frame k × fps_source`),
jamais de l'horloge du pipeline : c'est ce qui permet de comparer le relevé aux
images `annotation/t{k+1}.jpg` quelle que soit la base de temps interne.

## 14.2 Ce qui a été changé dans le code

### 14.2.1 §1.0 — une seule horloge, choisie par la source

`src/main.py` horodatait **toutes** ses observations sur `time.perf_counter()`,
y compris sur fichier vidéo (`src/main.py:913` avant ce lot). Désormais :

| Source | Base retenue (`timing.time_base: auto`) | Horodatage |
|---|---|---|
| Fichier vidéo | `video` | `(frame_index − 1) / fps_source` |
| Caméra | `wall` | `time.perf_counter() − start` |

- `video` et `wall` restent forçables, pour comparer les deux régimes.
- La base retenue est **publiée** dans `summary.json` (`time_base`) et dans
  `config_resolved.yaml` ; elle est affichée au démarrage (`[TEMPS] base=…`).
- **Le budget de frames suit la même base** : en base vidéo il est calculé une
  fois sur la cadence source (5 s de grâce = 150 frames à 30 ips), au lieu
  d'être recalculé à chaque frame sur le FPS de traitement (≈ 19 frames à
  3,8 ips). C'est le mécanisme précis que le §13.5 avait identifié comme cause
  de la divergence du harnais de session 0.
- Si le FPS de la source est illisible, le pipeline retombe sur la base murale
  **en le journalisant** (`CONFIG_WARNING`, code `time_base_video_unavailable`) :
  aucune cadence n'est devinée.

Le FPS de **traitement** continue d'être mesuré sur l'horloge murale et publié
à part : c'est une propriété de la machine, pas de la scène, et le critère FPS
du §4 s'y oppose inchangé.

### 14.2.2 §1.1 — `track_buffer` exprimé en secondes

`tracker.track_buffer_seconds: 15.0` (`PROVISOIRE`) devient la valeur de
référence ; `track_buffer` en frames en est **déduit** au démarrage, une seule
fois, par `track_buffer_frames_from_seconds()` — une seule implémentation pour
les deux sources, car deux arrondis divergents donneraient deux horizons
différents pour la même configuration.

| Source | FPS de conversion | Moment | Mécanisme |
|---|---|---|---|
| Fichier | cadence de la source (connue d'avance) | avant le premier appel au tracker | copie `tracker_resolved.yaml` dans la session, lue par Ultralytics |
| Caméra | FPS mesuré sur `fps_warm_up_frames` (60) | une fois, après le warm-up | ajustement de `max_time_lost` sur le tracker vivant |

- **Aucune réévaluation continue** : sur caméra le FPS dérive, et réévaluer
  changerait l'horizon du tracker sans prévenir — le défaut même qu'on corrige.
- Valeur convertie journalisée en INFO, et en `CONFIG_WARNING` si elle sort de
  `[15, 1200]` frames (avertir, pas refuser).
- Si `track_buffer_seconds` vaut `null`, repli sur la valeur en frames avec un
  `CONFIG_WARNING` explicite (compatibilité ascendante).
- Si Ultralytics ne se laisse pas ajuster (structure interne absente), la
  valeur du YAML reste en vigueur et l'échec est journalisé : la mesure est
  dégradée, le comptage ne l'est pas.

**Écart assumé avec la fiche** : elle prévoyait de mesurer le FPS sur
`fps_warm_up_frames` dans tous les cas. Sur fichier, la cadence est connue
d'avance et le tracker est construit **avant** la première frame : attendre
60 frames pour convertir laisserait le tracker tourner 60 frames avec le mauvais
horizon. La conversion y est donc faite avant le premier appel.

### 14.2.3 §1.2 — seuil NMS : valeur réellement en vigueur

La fiche demandait de **relever** la valeur appliquée avant de la changer, 0,7
n'étant que le défaut Ultralytics. Relevé :

| Source | Clé | Valeur |
|---|---|---|
| `config/pipeline.yaml:33` | `model.nms_iou` | **0,55** — marquée `PROVISOIRE — à calibrer` |
| `src/main.py` (appel `model.track`) | `iou=config.model.nms_iou` | la même, aucune valeur codée en dur |

**Le point de départ n'est donc pas 0,7 mais 0,55**, c'est-à-dire un seuil
*plus* agressif que le défaut : deux boîtes qui se recouvrent à plus de 55 %
sont réduites à une seule avant que le tracker les voie. Sur des personnes
assises côte à côte, c'est précisément la configuration qui détruit une
détection valide. L'écart à mesurer (0,55 → 0,85) est donc plus grand que ce que
la fiche supposait.

## 14.3 Mesure — compteurs internes, avant / après §1.0 + §1.1

Protocole : `scripts/measure_corpus.py`, donc `src/main.py` en sous-processus,
ligne `0.05,0.6,0.95,0.6`, un rejouage par ligne. « Avant » = `HEAD` au commit
`2f15649` (outillage seul, aucun seuil métier modifié), mesuré dans un worktree
séparé pour que l'écriture du code du lot ne perturbe pas la mesure.

### 14.3.1 Mesure principale — `fort_occ4`

| Compteur | Avant (base murale) | Après (base vidéo) | Écart | Significatif ? (§13.5.1) |
|---|---|---|---|---|
| `occupancy_operational` | 14 | **12** | −2 | oui (seuil ≥ 1) |
| `occupancy_observed` | 4 | 2 | −2 | oui |
| `OUT` | 1 | **0** | −1 | oui |
| `NEW` | 9 | 7 | −2 | oui |
| `TECHNICAL_ID_CHANGED` | 9 | **6** | −3 | oui |
| `PURGE` | 20 | 14 | −6 | oui (seuil ≥ 3) |
| `POSSIBLE_MULTI_PERSON_BOX` | 58 | 55 | −3 | oui |
| `OCCLUDED` | 49 | 52 | +3 | oui |
| `INCONSISTENT_STATE` | 4 | 9 | +5 | oui |
| `REID_MATCH` | 9 | 6 | −3 | oui |
| `REID_NEW` | 16 | 18 | +2 | oui |
| `long_gaps_count` | 1 | 0 | −1 | non (bruit ±3) |
| `OCCUPANCY_SNAPSHOT` | 273 | 36 | −237 | mécanique, voir ci-dessous |
| FPS moyen | 3,559 | **3,816** | +0,257 | ≥ 3,06 : critère tenu |

**`OCCUPANCY_SNAPSHOT` n'est pas une dégradation** : `snapshot_interval_seconds`
vaut 1,0 s. En base murale, 35 s de vidéo traitées en ≈ 290 s murales
produisaient ≈ 273 instantanés ; en base vidéo, elles en produisent 36, soit un
par seconde **de scène**. C'est le comportement attendu, et c'est la
démonstration la plus directe que l'horloge a changé de référentiel.

**Le `OUT` fantôme disparaît.** La vérité terrain ne contient **aucune sortie**
sur les 35 secondes (§12.1) ; la baseline en produisait une, le pipeline corrigé
n'en produit aucune. C'est le seul compteur de ce tableau qui se compare
directement à une observation humaine, et il va dans le bon sens.

### 14.3.2 Non-régression — `rare_occ2`

| Compteur | Avant | Après | Écart |
|---|---|---|---|
| `IN` / `OUT` / `NEW` | 4 / 1 / 2 | **4 / 1 / 2** | 0 |
| `occupancy_operational` | 5 | 5 | 0 |
| `occupancy_observed` | 3 | 3 | 0 |
| `technical_ids` / `persons_created` | 7 / 7 | 7 / 7 | 0 |
| `TECHNICAL_ID_CHANGED` | 1 | **0** | −1 |
| `INCONSISTENT_STATE` | 1 | 0 | −1 |
| `PURGE` | 2 | 2 | 0 |
| FPS moyen | 3,836 | 3,712 | −0,124 (≥ 3,06) |

**Aucun compteur dégradé au-delà du bruit** : le critère de non-régression du
§4 est tenu sur cette vidéo. Le seul écart net est favorable (un changement
d'identifiant technique en moins).

## 14.4 §1.3 — investigation du filtre géométrique : **aucun correctif appliqué**

La fiche prescrivait de trancher entre trois hypothèses avant de toucher à quoi
que ce soit. Elles le sont.

### 14.4.1 Hypothèse 1 (artefact du harnais) — **écartée pour les rejets**

| Mesure sur `fort_occ.mp4` | Session 0 (harnais ad hoc) | Lot 1 (pipeline réel) |
|---|---|---|
| `DETECTION_REJECTED_GEOMETRY` | 235 | **235** |
| dont `aspect_ratio_out_of_range` | 235 | **235** |
| `POSSIBLE_MULTI_PERSON_BOX` | 0 | **6** |

Les 235 rejets sont **réels** : ils se retrouvent au rejet près sur le pipeline
réel. En revanche le « 0 fusion » de la session 0 était bien un artefact : le
pipeline en signale 6. La nuance compte — le §13.5 avait raison sur les fusions,
il ne dit rien des rejets, et ceux-ci ne s'expliquent pas par le harnais.

Sur `fort_occ2.mp4` : 26 rejets, tous `aspect_ratio_out_of_range`, 1 fusion
signalée, sur 328 frames.

### 14.4.2 Hypothèses 2 et 3 — tranchées en regardant les boîtes

Dix boîtes rejetées ont été extraites et inspectées
(`results/geom_crops/rejets_fort_occ.jpg`, non versionné : personnes
identifiables) : les cinq plus larges (ratio 3,85 à 3,45) et cinq médianes
(ratio 2,66).

**Les dix montrent la même chose : le haut du crâne d'une personne placée au
ras de l'objectif**, en bas de l'image, jamais deux personnes côte à côte.

Le décompte confirme ce que l'œil voit :

| Fait mesuré | Valeur |
|---|---|
| Pistes techniques concernées par les 235 rejets | **une seule**, la n° 15 |
| Étendue | frames 137 à 988 (sur 1 057) |
| Boîte médiane | **396 × 147 px** dans une image 1920 × 1080 |
| Ratio médian W/H | 2,66 (seuil `max_aspect_ratio_wh` = 2,5) |
| Confiance médiane | 0,67 — ce ne sont pas des détections douteuses |

- **Hypothèse 3 (fusions réellement perdues) : écartée.** Une boîte unique,
  attachée à une piste unique pendant 851 frames, n'est pas deux personnes
  côte à côte. Router ces boîtes vers le chemin fusion ne produirait rien.
- **Hypothèse 2 (rejets légitimes) : vraie sur la forme, fausse sur le fond.**
  La boîte est bien aberrante — plus large que haute — mais l'objet derrière
  est une **vraie personne**, pas un reflet ni un fragment. Le filtre écarte
  une personne réelle, systématiquement, pendant 80 % de la vidéo.

### 14.4.3 Décision : ne rien corriger dans ce lot

**Le correctif conditionnel prévu par la fiche n'est pas appliqué**, parce que
la condition qui l'autorisait (hypothèse 3 confirmée) est fausse. C'est le
résultat de l'investigation, pas un renoncement.

Ce que la mesure a révélé est un **autre** problème, non prévu par la fiche :
une personne trop proche de l'objectif présente une silhouette plus large que
haute et se fait écarter avant tout suivi. Il est consigné ici et **pas
corrigé** :

- il ne concerne pas la vidéo de mesure principale (`fort_occ4` : **0** rejet
  géométrique) ni la vidéo de contrôle (`rare_occ2` : **0**) ;
- relâcher `max_aspect_ratio_wh` pour l'absorber changerait le filtre pour tout
  le corpus, sans mesure à l'appui sur la vidéo de référence ;
- la bonne réponse est probablement une règle de **bord d'image** (une boîte
  tronquée par le bord n'a pas de ratio interprétable), ce qui relève du lot 2,
  qui traite déjà le bord du cadre.

**À verser au lot 2** : une boîte coupée par le bord bas de l'image ne devrait
pas être jugée sur son ratio W/H.

## 14.5 Déterminisme — ce que le §1.0 change vraiment

Le §2.1 de `CLAUDE.md` exige de connaître le bruit avant de conclure quoi que
ce soit. Deux exécutions de `fort_occ4`, **mêmes arguments, même ligne**, mais
sous des charges machine différentes : l'une par `scripts/measure_corpus.py`,
l'autre par `src/main.py --trace-per-second`, qui écrit en plus 36 images.

| Base de temps | Compteurs différant entre les deux exécutions |
|---|---|
| **Murale (avant)** | **13** : `IN` (2 vs 1), `OUT` (1 vs 0), `NEW` (9 vs 8), `TECHNICAL_ID_CHANGED` (9 vs 7), `POSSIBLE_MULTI_PERSON_BOX` (58 vs 52), `PURGE` (20 vs 18), `STATE_TRANSITION` (137 vs 126), `REID_MATCH` (9 vs 7), `REID_NEW` (16 vs 17), `REID_AMBIGUOUS`, `INCONSISTENT_STATE`, `STABILIZATION`, `OCCUPANCY_SNAPSHOT` |
| **Vidéo (après)** | **aucun** — les 21 types d'événements sont identiques au comptage près |

**C'est le résultat principal du §1.0.** Le §13.5.1 avait conclu ces compteurs
« strictement stables » à partir de trois rejouages lancés à la suite, dans les
mêmes conditions de charge : cette stabilité était une propriété de la
**mesure**, pas du système. Dès que la charge change — et écrire 36 images
suffit — la base murale déplace `IN`, `OUT` et `NEW`, c'est-à-dire le bilan
officiel lui-même.

**Conséquence pour les lots suivants** : les seuils de signification du §13.5.1
(`PURGE` ±2, `STATE_TRANSITION` ±2, `OCCUPANCY_SNAPSHOT` ±11, `long_gaps_count`
±3) ont été établis en base murale. En base vidéo, le bruit mesuré ici est
**nul sur tous les compteurs**. Tout écart avant/après devient donc un signal,
à condition d'être mesuré dans cette base. Cette table remplace celle du
§13.5.1 pour les lots 4, 2, 3, 5, 6 et 7.

## 14.6 §1.2 — seuil NMS : les valeurs proposées sont **mesurées et rejetées**

Chaque valeur isolément, sur `fort_occ4`, en base vidéo (donc sans bruit de
mesure, §14.5), toutes choses égales par ailleurs.

| Compteur | **0,55** (en vigueur) | 0,85 (proposé) | 0,90 (proposé) |
|---|---|---|---|
| `technical_ids` | **21** | 35 | 65 |
| `persons_created` | **18** | 22 | 45 |
| `TECHNICAL_ID_CHANGED` | **6** | 14 | 21 |
| `PURGE` | **14** | 20 | 42 |
| `OCCLUDED` | **52** | 101 | 135 |
| `POSSIBLE_MULTI_PERSON_BOX` | **55** | 79 | 88 |
| `REID_NEW` | **18** | 22 | 45 |
| `occupancy_operational` (réel : 13) | 12 | 13 | **16** |
| `visible_count` max (visibles réels : jusqu'à 12) | 9 | 9 | 11 |
| `erreur_comptage_max_visible` | **9** | **9** | **9** |
| Pistes surnuméraires (§14.7) | **6** | 13 | 7 |
| FPS moyen | 4,189 | 4,157 | 4,279 |

**Décision : `nms_iou` reste à 0,55. Les deux valeurs de la fiche sont
écartées par la mesure**, pas par principe.

L'hypothèse du §1.2 était que la NMS détruisait des détections de personnes
assises côte à côte, et qu'en relâchant le seuil on les récupérerait. La mesure
dit le contraire de ce qui était attendu :

- le nombre d'identifiants techniques **triple** à 0,90 (21 → 65) et les
  identités créées passent de 18 à 45, pour 13 personnes réelles ;
- `occupancy_operational` passe de 12 à **16**, soit trois personnes comptées
  en trop là où la baseline en manquait une ;
- `POSSIBLE_MULTI_PERSON_BOX` **augmente** (55 → 88) au lieu de diminuer : les
  boîtes supplémentaires ne séparent pas les personnes, elles se superposent ;
- et surtout **`erreur_comptage_max_visible` ne bouge pas** : 9 dans les trois
  configurations. Les personnes manquantes ne manquaient pas par suppression
  NMS.

Le risque que la fiche demandait de surveiller — « des doublons survivent la
NMS et sont promus en identités distinctes → sur-comptage » — est donc
**réalisé, et mesuré**. Il l'est déjà à `new_track_thresh: 0.7` ; l'abaisser à
0,45 au lot 4 l'aggraverait.

**Aucune modification de `config/pipeline.yaml` pour ce point** : la valeur en
vigueur est conservée, et son commentaire `PROVISOIRE — à calibrer` est
complété par le résultat ci-dessus, pour qu'une session ultérieure ne refasse
pas cette mesure.

## 14.7 Métriques de vérité terrain — et ce que ce lot **n'a pas** déplacé

Relevé par seconde du pipeline réel contre `tests/fixtures/ground_truth_fort_occ4.json`.
Les métriques d'**identité** ne figurent pas ici : le rattachement
`person_id` ↔ étiquettes est une proposition non validée, et sa validation est
reportée au lot 2 (décision du 2026-09-21). Ce qui suit ne dépend d'aucun
rattachement.

| Métrique (§3, §4) | Avant | Après §1.0 + §1.1 | Cible §4 | Verdict |
|---|---|---|---|---|
| `occupancy_operational` à la dernière seconde (réel : 13) | 13 *(ou 14 selon le run)* | **12** | exact | **non tenu**, écart 1 |
| `erreur_comptage_max_operational` | 4 | 4 | voir ci-dessous | inchangé |
| `erreur_comptage_max_visible` | **9** | **9** | ≤ 1 | **non tenu**, inchangé |
| `pistes_vues` minimum | 2 | 2 | diagnostic | inchangé |
| Fragmentation max (identifiants techniques par `person_id`) | 5 | 5 | — | inchangé |
| Pistes surnuméraires | 7 | **6** | — | −1 |
| Identifiants techniques partagés par plusieurs `person_id` | 2 | 2 | — | inchangé |
| FPS moyen | 3,559 | **3,816** | ≥ 3,06 | tenu |

### 14.7.1 `erreur_comptage_max_operational` = 4 : ce que ce chiffre est

Il est **entièrement porté par la seconde 0**, où l'annotation compte 4
personnes et le pipeline 0 : le warm-up n'est pas terminé. Hors seconde 0,
l'écart vaut −1 pendant chaque entrée et −2 aux secondes 15 à 18, où deux
personnes entrent coup sur coup. C'est le retard d'entrée documenté au §4 de
`CLAUDE.md`, et c'est le comportement correct : `occupancy_operational`
n'incrémente qu'au franchissement confirmé, l'annotation compte dès
l'apparition dans le champ.

### 14.7.2 Le point qui n'est pas tenu : 12 au lieu de 13 en fin de séquence

`WARMUP_END` explique l'écart, et il vient du §1.0 lui-même :

| | Avant (base murale) | Après (base vidéo) |
|---|---|---|
| Fin du warm-up | frame 17, t = 33,2 s **murales** | frame 16, t = 0,5 s **vidéo** |
| `initial_occupancy` | **4** | **3** |
| Bilan final | 4 + 1 IN + 8 NEW − 0 OUT = 13 | 3 + 2 IN + 7 NEW − 0 OUT = **12** |

La note d'annotation de la seconde 0 dit : « *Une partie de P1 et P4 sont
derrières une porte* ». Quatre personnes sont présentes, mais deux ne sont que
partiellement visibles ; à la frame 16 le pipeline n'en place que trois. La
quatrième, déjà à l'intérieur, ne franchira jamais la ligne, donc rien ne la
rattrape ensuite.

**Ce n'est pas une régression du comptage, c'est un changement d'instant de
mesure** : le warm-up se termine désormais à 0,5 s de **scène** au lieu de
33 s d'horloge murale, c'est-à-dire beaucoup plus tôt dans l'action. Le
compteur est plus exact en base murale par accident — la lenteur du traitement
laissait le temps à la quatrième personne d'être vue.

### 14.7.2.1 Ventilation des 12 — d'où vient chaque personne comptée

`occupancy_operational = initial + IN + NEW − OUT`. Après le lot 1, sur
`fort_occ4` :

| Origine | Mesuré | Attendu d'après la vérité terrain | Écart |
|---|---|---|---|
| `initial` (warm-up) | **3** | **0** | +3 |
| `IN` (franchissement confirmé) | **2** | **13** | −11 |
| `NEW` (apparition intérieure) | **7** | **0** | +7 |
| `OUT` | 0 | 0 | 0 |
| **Total** | **12** | **13** | −1 |

**L'écart de 1 sur le total masque une erreur de structure bien plus grande.**
La vérité terrain est sans ambiguïté : à la seconde 0, **aucune personne n'est
dans la salle**. La note de `t01.jpg` dit « *Une partie de P1 et P4 sont
derrières une porte* » — les quatre sont en train d'entrer, pas installées.
`P5` « entre dans la salle » à la seconde 3, `P1` à la seconde 5, `P7` à la
seconde 8, et ainsi de suite jusqu'à `P13` à la seconde 29 : la salle se
remplit de 0 à 13 par la porte, sans jamais se vider (§12.1).

Le comptage attendu est donc **0 initial et 13 franchissements**. Le pipeline
en produit 3, 2 et 7 : il place trois personnes au warm-up alors qu'elles sont
encore dans l'embrasure, et il rattrape ensuite la majorité des autres par
`NEW`, c'est-à-dire comme des apparitions intérieures — des personnes
considérées comme déjà là, jamais comme entrées.

Que le total tombe à 12 relève de la compensation : trois personnes comptées
trop tôt, onze entrées manquées, sept apparitions intérieures. **Un total
presque juste n'est pas un comptage juste**, et c'est précisément pourquoi le
§4 exige `occupancy_operational` exact *et* aucun IN/OUT fantôme.

**Aucun correctif dans ce lot** : le §8 de `CLAUDE.md` interdit de toucher à la
logique IN / OUT / NEW, et la question relève du périmètre du lot 2 (warm-up,
rétention, bord du cadre). Ce qui est établi ici est le **chiffre**, pas la
solution.

**À verser au lot 2** : le warm-up ne devrait pas figer l'effectif initial sur
une seule frame quand des personnes sont partiellement masquées à l'entrée, et
la question de fond est de savoir pourquoi onze entrées réelles sont vues comme
des apparitions intérieures.

### 14.7.3 Ce que le lot 1 n'a pas déplacé, et c'est l'essentiel

`visible_count` est **identique seconde par seconde** avant et après, sauf à la
seconde 0 (0 → 3). L'effondrement décrit au §12.2.2 est intact : 5 personnes
vues à la seconde 34 pour 11 visibles annotées, `erreur_comptage_max_visible`
toujours à 9.

**Ce lot n'est donc pas un succès au regard du §4** : il ne rapproche d'aucun
critère visible. Il rend les mesures suivantes interprétables et déterministes
(§14.5), ce qui était sa raison d'être, mais il ne corrige pas le symptôme.
Le décrochage reste au niveau des pistes — `pistes_vues` descend à 2 pour 10
personnes visibles à la seconde 28, inchangé — ce qui confirme l'analyse ayant
conduit à avancer le lot 4 avant le lot 2.

## 14.8 Seuils modifiés

| Paramètre | Avant | Après | Effet mesuré | Compromis accepté | Statut |
|---|---|---|---|---|---|
| `timing.time_base` | n'existait pas (temps mural systématique) | `auto` | bruit run-à-run : 13 compteurs divergents → **0** (§14.5) ; `OUT` fantôme supprimé ; `TECHNICAL_ID_CHANGED` 9 → 6 ; `PURGE` 20 → 14 | sur fichier, les durées ne reflètent plus le temps de calcul ; le warm-up se termine plus tôt dans l'action, d'où 12 au lieu de 13 (§14.7.2) | **figé** — ce n'est pas un réglage mais une correction de référentiel |
| `tracker.track_buffer_seconds` | n'existait pas | `15.0` | non isolable de `time_base` dans cette mesure (voir ci-dessous) | horizon plus long = plus de mémoire, risque accru de réassociation erronée | `PROVISOIRE` |
| `tracker.fps_warm_up_frames` | n'existait pas | `60` | aucun sur fichier (conversion faite avant le premier appel au tracker) | — | `PROVISOIRE` |
| `tracker.track_buffer` | `150` (frames, écrit à la main) | **déduit** : 450 frames sur fichier à 30 ips | horizon réel porté de 5 s à 15 s de scène | — | déduit, plus réglé à la main |
| `model.nms_iou` | `0.55` | **`0.55`, inchangé** | 0,85 et 0,90 mesurés et écartés (§14.6) | — | `PROVISOIRE`, commentaire complété |

**Limite de méthode, à dire clairement** : `time_base` et
`track_buffer_seconds` ont été mesurés **ensemble**, pas isolément. Le §7 de
`CLAUDE.md` demande l'inverse. La raison est que les deux sont indissociables
ici : convertir `track_buffer` en secondes suppose de savoir quelle horloge
fait foi, et mesurer la base de temps sans corriger `track_buffer` laisserait
l'horizon du tracker à 150 frames, soit 5 s de scène au lieu de 15. L'effet
propre de `track_buffer_seconds` n'est donc **pas** établi par ce lot, et le
tableau du §14.3.1 ne doit pas lui être attribué.

## 14.9 Tests

Suite complète : **664 tests collectés, 663 passés, 1 ignoré**, couverture
**93,39 %** (seuil 85 %). Le test ignoré (`test_calibration_window.py:227`)
l'était déjà : il exige un serveur X. Décompte réel
(`grep -rc "^def test_" tests/`) : **469** fonctions sur **34** fichiers,
contre 445 sur 31 au lot 0.

### 14.9.1 Tests ajoutés

| Fichier | Ce qu'il protège |
|---|---|
| `tests/test_time_base_lot1.py` (10 tests) | `auto` suit la source ; `video`/`wall` restent forçables ; une base inconnue est refusée par la configuration ; le budget de frames suit la cadence de la base (150 frames de grâce à 30 ips contre 19 à 3,8 ips) ; conversion unique secondes → frames ; la copie `tracker_resolved.yaml` ne modifie pas le YAML versionné ; ajustement du tracker vivant, et échec signalé sans interrompre le comptage ; repli si `track_buffer_seconds` vaut `null` ; valeurs publiées dans la configuration résolue |
| `tests/test_second_trace.py` (5 tests) | la seconde vient de l'index de frame, jamais d'une horloge ; refus sur caméra ; une ligne et une image par seconde ; **le relevé ne modifie ni l'état du pipeline ni la frame** |
| `tests/test_gt_matching.py` (9 tests) | la proposition est marquée non validée et chaque ligne porte méthode et confiance ; les métriques d'identité sont **refusées** sans validation humaine ; les métriques de comptage le sont toujours ; chaque compteur est comparé à son référentiel ; fragmentation de piste et identifiants techniques partagés |

### 14.9.2 Tests corrigés — aucun

Aucun test existant n'a eu besoin d'être modifié. Cinq ont **échoué en cours de
route**, et c'est le correctif qui était fautif, pas eux : les avertissements de
base de temps étaient émis **avant** `SESSION_START` et polluaient le journal
d'une source illisible. Les événements ont été retirés du journal (le repli
reste sur stderr, dans les logs, et la base retenue est publiée dans
`summary.json`), après quoi les cinq tests sont repassés sans être touchés.

## 14.10 Ce qui reste non résolu

1. **Le symptôme visé n'a pas bougé** : `erreur_comptage_max_visible` vaut 9
   avant comme après, pour une cible de ≤ 1. Le lot est un préalable de
   mesure, pas un correctif (§14.7.3).
2. **`occupancy_operational` finit à 12 pour 13 personnes réelles**, à cause de
   l'instant de fin du warm-up (§14.7.2). À traiter au lot 2.
3. **L'effet propre de `track_buffer_seconds` n'est pas isolé** (§14.8), et la
   valeur 15,0 s reste `PROVISOIRE` : la plus longue occlusion du corpus reste
   à mesurer en base vidéo.
4. **Une personne au ras de l'objectif est systématiquement écartée** par le
   filtre de ratio sur `fort_occ` (§14.4.3). Non corrigé, versé au lot 2.
5. **La conversion de `track_buffer` sur caméra n'a pas été mesurée en vrai** :
   le corpus ne contient que des fichiers. Le chemin est testé unitairement,
   pas en conditions réelles.
6. **Les seuils de signification du §13.5.1 sont périmés** pour les mesures
   faites en base vidéo : le bruit y est nul (§14.5). Les lots suivants doivent
   se mesurer dans cette base, sans quoi les tableaux ne sont pas comparables.
7. **Reporté du lot 0** : la ligne n'est toujours pas publiée dans
   `config_resolved.yaml` (§13.7.2). Inchangé, toujours ouvert.
8. **`bootstrap_frames` est encore compté en frames** (`occupancy_manager.py:321`,
   publié dans `summary.json`) : même défaut d'unité que `track_buffer` avant ce
   lot. Il compte les frames traitées avant qu'un budget existe ; sa durée réelle
   dépend donc de la source, ce que le §1.0 vient précisément de corriger
   ailleurs. **Dette inscrite, à traiter au lot 2** : l'exprimer en secondes de
   scène.
9. **Le comptage est structurellement faux sur `fort_occ4`** alors que son total
   est presque juste : 3 initial + 2 IN + 7 NEW au lieu de 0 + 13 + 0
   (§14.7.2.1). Aucun correctif ici — la logique IN/OUT/NEW est interdite de
   modification (§8 de `CLAUDE.md`) et la question relève du lot 2.

---

# 15. Lot 4 — seuils BoT-SORT

> **Accepté par la personne responsable le 2026-09-21**, avec la réserve de
> non-régression de `rare_occ2` inscrite au §15.13.3.
>
> **Statut : accepté.** Un seul seuil est modifié :
> `tracker.new_track_thresh` 0,7 → **0,60** (`PROVISOIRE`), décision de la
> personne responsable du 2026-09-21 après mesure de 0,60, 0,55 et 0,45
> (**§15.13**, qui remplace les §15.7 à §15.9 pour la valeur appliquée).
> `match_thresh` est mesuré et **écarté** ; `proximity_thresh` est mesuré et
> **non appliqué**, à reprendre après le lot 3. Le filtre d'inclusion est
> **écarté** (§15.12).
>
> **Réserve inscrite** : la non-régression de `rare_occ2` n'est pas tenue
> (identités 7 → 10, doublons 1 → 7) ; elle est **acceptée en connaissance
> de cause**, car la mesure qui compte s'améliore : 5 personnes réelles vues
> sur 5 au lieu de 4 (§15.13.3).
>
> *Les §15.1 à §15.11 décrivent la première passe, qui avait appliqué 0,45.*

> **Mention de portée.** Mesuré sur `test/fort_occ4.mp4` (mesure principale,
> 1 055 frames, 35,2 s, 13 personnes annotées, 4 segments d'occlusion dont 3 de
> plus de 3 s) et `test/rare_occ2.mp4` (non-régression, 744 frames, 24,8 s,
> **aucune vérité terrain**). Base de temps vidéo : bruit run-à-run **nul** sur
> tous les compteurs (§14.5), vérifié à nouveau ici (§15.2). Ligne de
> convention figée `--line 0.05,0.6,0.95,0.6`.
>
> **Ce qui ne s'interprète pas ici.** `IN`, `OUT`, `NEW` et
> `occupancy_operational` dépendent de la ligne de convention. Sur `fort_occ4`,
> cette ligne passe au-dessus du trajet des pieds des personnes qui entrent :
> relevé du 2026-09-21 dans `results/line_check/`, non versionné, et une
> ligne candidate au seuil de la porte, `0.17,0.86,0.47,0.86`, qui n'a pas été
> adoptée. Ces quatre compteurs sont **rapportés sans interprétation** (§15.4).
> Les indicateurs du lot sont `visible_count`, `pistes_vues` et la
> fragmentation de piste.
>
> Les métriques d'**identité** (`id_switches_reels`, `fusions_reelles`,
> `fragments_par_personne`) restent **non publiées** : le rattachement est
> toujours au statut `proposition` (validation reportée au lot 2).

## 15.1 Protocole

| Variante | `match_thresh` | `proximity_thresh` | `new_track_thresh` |
|---|---|---|---|
| `base` (production avant lot) | 0,9 | 0,5 | 0,7 |
| `m075` | **0,75** | 0,5 | 0,7 |
| `p080` | 0,9 | **0,8** | 0,7 |
| `n045` (**retenue**) | 0,9 | 0,5 | **0,45** |
| `n045p080` | 0,9 | **0,8** | **0,45** |
| `combo` (cible de la fiche) | **0,75** | **0,8** | **0,45** |

Toutes les autres valeurs sont celles de `HEAD` (`4347386`) : `nms_iou` 0,55,
`track_buffer_seconds` 15,0 (450 frames), `appearance_thresh` 0,25. Chaque
variante est un couple de fichiers de configuration hors dépôt
(`results/lot4/pipeline_<v>.yaml` → `results/lot4/botsort_<v>.yaml`), passé par
`--config` : les fichiers versionnés n'ont pas été touchés pendant la mesure.

Trois passes :

1. **Compteurs internes et FPS** : `scripts/measure_corpus.py`, une variante et
   une vidéo à la fois, **seul sur la machine**.
2. **Vérité terrain** : `src/main.py --trace-per-second` sur `fort_occ4`, puis
   `scripts/gt_matching.py metrics`. Les relevés ont tourné en parallèle : leur
   FPS n'est pas utilisé.
3. **Complément** : la variante `n045p080`, deux rejouages pour estimer le bruit
   du FPS, et les relevés par seconde de `rare_occ2`.

La fiche demandait d'étendre `scripts/measure_tracker_swaps.py` pour isoler
`match_thresh` si besoin. **Ce n'était pas nécessaire** : `--config` suffit à
isoler chaque seuil sur le pipeline réel.

## 15.2 Déterminisme — revérifié

| Contrôle | Résultat |
|---|---|
| `base` du lot 4 contre l'état après le lot 1 (§14.3.1) | identiques : 21 `technical_ids`, 18 `persons_created`, 6 `TECHNICAL_ID_CHANGED`, 14 `PURGE`, 55 `POSSIBLE_MULTI_PERSON_BOX` |
| Passe 1 (seule) contre passe 2 (3 exécutions parallèles, écriture d'images), pour les 5 variantes | compteurs identiques sur les 5 |
| Deux rejouages de `base` et de `p080` | compteurs identiques |
| **FPS** entre deux rejouages identiques | `base` 4,305 / 4,195 ; `p080` **3,566 / 4,087** |
| Configuration **versionnée** après modification (sans `--config`) contre la variante `n045` | compteurs identiques sur les deux vidéos (`fort_occ4` : 46 / 38 / 12 / 28 `PURGE` ; `rare_occ2` : 12 / 12 / 0 / 3 `PURGE`) ; FPS 3,864 et 3,717 |

**Le FPS est la seule grandeur bruitée**, et son bruit atteint **0,5 ips** entre
deux exécutions aux compteurs identiques. La baisse apparente de `p080` à la
première exécution (4,31 → 3,57) était du bruit : le rejouage donne 4,09. Une
différence de FPS inférieure à 0,5 ips entre deux variantes n'est donc pas un
effet, et le tableau du §15.4 ne s'interprète qu'au regard du plancher.

## 15.3 Métriques de vérité terrain — `fort_occ4`

| Métrique | `base` | `m075` | `p080` | **`n045`** | `n045p080` | `combo` | Cible §4 |
|---|---|---|---|---|---|---|---|
| `erreur_comptage_max_visible` | 9 | 9 | 9 | **7** | 6 | 6 | ≤ 1 |
| Somme des \|écarts visibles\| sur 35 s | 143 | 159 | 139 | **77** | 75 | 78 | — |
| Secondes où \|écart visible\| ≤ 1 (sur 35) | 12 | 10 | 11 | **20** | 20 | 20 | — |
| Secondes en surcompte visible (écart > 0) | 1 | 0 | 1 | **7** | 6 | 6 | — |
| Surcompte visible maximal | +1 | 0 | +1 | **+2** | +2 | +2 | — |
| `pistes_vues` min / max | 2 / 9 | 2 / 8 | 2 / 9 | **3 / 13** | 3 / 13 | 3 / 13 | diagnostic |
| Fragmentation max (id. techniques par `person_id`) | 5 | 6 | **3** | 4 | 4 | 5 | — |
| Pistes surnuméraires | 6 | 13 | **4** | 11 | 13 | 18 | — |
| `person_id` fragmentés | 3 | 7 | 3 | 9 | 9 | 10 | — |
| Identifiants techniques partagés par plusieurs `person_id` | 2 | 1 | 2 | 2 | 3 | 1 | — |
| `erreur_comptage_max_operational` | 4 | 4 | 4 | 5 | 5 | 7 | *non interprété (§15.4)* |

### 15.3.1 `visible_count` seconde par seconde (extrait)

| Seconde | Visibles annotées | `base` | `p080` | **`n045`** | `n045p080` | `combo` |
|---|---|---|---|---|---|---|
| s16 | 11 | 7 | 7 | **10** | 10 | 10 |
| s20 | 12 | 5 | 5 | **7** | 7 | 7 |
| s24 | 12 | 3 | 4 | **5** | 6 | 7 |
| s28 | 10 | 2 | 3 | **6** | 6 | 5 |
| s29 | 11 | 5 | 6 | **11** | 11 | 10 |
| s32 | 11 | 4 | 5 | **12** | 12 | 12 |
| s34 | 11 | 5 | 6 | **10** | 10 | 10 |

**L'effondrement de fin de séquence décrit aux §12.2.2 et §14.7.3 est levé par
`new_track_thresh` seul** : de s29 à s34, 10 à 12 pistes visibles pour 10 à 11
personnes visibles, contre 4 à 5 en production. L'écart maximal (7) se situe
désormais entre s20 et s24, où `visible_count` reste à 5–7 pour 12 personnes.

### 15.3.2 Le gain n'est pas fait de doublons

Un surcompte visible de +1 ou +2 apparaît avec `n045` (7 secondes sur 35). Pour
vérifier que le gain de fin de séquence n'est pas simplement un empilement de
boîtes sur les mêmes personnes, j'ai calculé un indicateur objectif : une boîte
visible **incluse à ≥ 80 %** dans une autre boîte visible de la même seconde
(script ad hoc `results/lot4/inclusion.py`, non versionné). Retirer toutes ces
boîtes est conservateur : une personne assise derrière une autre peut
légitimement y être incluse.

| | `base` | `m075` | `p080` | `n045` | `n045p080` | `combo` |
|---|---|---|---|---|---|---|
| Boîtes incluses, total sur 36 s | 2 | 1 | 5 | 23 | 25 | 22 |
| `erreur_comptage_max_visible` après retrait | 9 | 9 | 9 | **7** | 6 | 6 |
| `visible_count` après retrait, s29 à s34 | 4–5 | 3–4 | 5–6 | **8–10** | 7–10 | 8–10 |

Même en retirant toutes les boîtes incluses, `n045` voit 8 à 10 personnes à la
fin, contre 4 à 5. **Le gain est réel.** Il est aussi **en partie gonflé** par
des doublons : l'inspection de `s32` (`results/lot4/trace/n045/trace_frames/`)
montre des boîtes emboîtées sur les personnes assises du groupe de gauche, là
où la production n'avait aucune piste.

## 15.4 Compteurs internes

### 15.4.1 Mesure principale — `fort_occ4`

| Compteur | `base` | `m075` | `p080` | **`n045`** | `n045p080` | `combo` |
|---|---|---|---|---|---|---|
| `technical_ids` | 21 | 30 | 19 | **46** | 43 | 73 |
| Identités créées (`persons_created` = `REID_NEW`) | 18 | 18 | 17 | **38** | 34 | 52 |
| `TECHNICAL_ID_CHANGED` (= `REID_MATCH`) | 6 | 14 | 5 | **12** | 14 | 23 |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 1 | 2 | 1 | **5** | 4 | 6 |
| `IDENTITY_SWAP_CORRECTED` | 0 | 0 | 0 | **0** | 0 | 0 |
| `REID_AMBIGUOUS` | 4 | 7 | 4 | **22** | 16 | 36 |
| `PURGE` | 14 | 15 | 13 | **28** | 22 | 40 |
| `OCCLUDED` | 52 | 53 | 56 | **84** | 84 | 93 |
| `POSSIBLE_MULTI_PERSON_BOX` | 55 | 42 | 50 | **52** | 52 | 56 |
| `INCONSISTENT_STATE` | 9 | 7 | 11 | **20** | 18 | 22 |
| `DETECTION_REJECTED_GEOMETRY` | 0 | 0 | 0 | **0** | 0 | 0 |
| `long_gaps_count` | 0 | 0 | 0 | **1** | 0 | 0 |
| FPS moyen (1re exécution) | 4,305 | 4,041 | 3,566 | **4,176** | 4,029 | 3,566 |
| *`[BILAN]` — non interprété :* | | | | | | |
| *`IN` / `OUT` / `NEW`* | *2 / 0 / 7* | *1 / 0 / 6* | *1 / 0 / 5* | ***8 / 0 / 7*** | *6 / 0 / 10* | *8 / 0 / 9* |
| *`occupancy_operational`* | *12* | *10* | *9* | ***18*** | *19* | *20* |
| *`occupancy_observed`* | *2* | *2* | *1* | ***6*** | *9* | *6* |
| *`occupancy_uncertain` / `occupancy_range`* | *10 / [2, 12]* | *8 / [2, 10]* | *8 / [1, 9]* | ***12 / [6, 18]*** | *10 / [9, 19]* | *14 / [6, 20]* |

### 15.4.2 Non-régression — `rare_occ2`

| Compteur | `base` | `m075` | `p080` | **`n045`** | `n045p080` | `combo` |
|---|---|---|---|---|---|---|
| `technical_ids` | 7 | 9 | 7 | **12** | 12 | 18 |
| Identités créées | 7 | 8 | 7 | **12** | 12 | 16 |
| `TECHNICAL_ID_CHANGED` | 0 | 2 | 0 | **0** | 0 | 2 |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 0 | 0 | 0 | **0** | 0 | 0 |
| `REID_AMBIGUOUS` | 0 | 1 | 0 | **6** | 6 | 10 |
| `PURGE` | 2 | 4 | 2 | **3** | 3 | 6 |
| `OCCLUDED` | 6 | 8 | 6 | **20** | 20 | 24 |
| `INCONSISTENT_STATE` | 0 | 1 | 0 | **1** | 1 | 1 |
| Fragmentation max / pistes surnuméraires (relevé par seconde) | 1 / 0 | — | 1 / 0 | **1 / 0** | 1 / 0 | 2 / 1 |
| Pistes visibles max (relevé par seconde) | 5 | — | 5 | **7** | 7 | 7 |
| Boîtes incluses (§15.3.2), total | 15 | — | 15 | **26** | 26 | 27 |
| FPS moyen | 4,253 | 3,956 | 3,215 | **4,124** | 3,384 | 3,697 |
| *`IN` / `OUT` / `NEW` — non interprété* | *4 / 1 / 2* | *3 / 0 / 2* | *4 / 1 / 2* | ***5 / 2 / 2*** | *5 / 2 / 2* | *3 / 1 / 2* |
| *`occupancy_operational` / `observed` — non interprété* | *5 / 3* | *5 / 3* | *5 / 3* | ***5 / 2*** | *5 / 2* | *4 / 2* |

## 15.5 `rare_occ2` : ce que sont les 5 identités supplémentaires

La vidéo n'a pas de vérité terrain : les compteurs seuls ne disent pas si les
identités en plus sont de fausses créations ou des personnes que la production
ratait. Deux constats répondent en partie.

1. **Ce ne sont pas des fragments** : fragmentation 1 et 0 piste surnuméraire
   avec `n045`, comme en production. Les pistes en plus sont **simultanées**
   (pistes visibles : 4 → 6–7 entre s11 et s24), pas successives.
2. **Inspection de `s20`** (`results/lot4/trace/{base,n045}_rare_occ2/trace_frames/s20.jpg`) :
   5 personnes sont réellement visibles (l'enseignant de dos et quatre
   enfants).

   | | Production | `n045` |
   |---|---|---|
   | Pistes visibles | 4 | 7 |
   | Personnes réelles suivies | 4 : la fille assise à droite n'a **aucune** piste | **5** : elle est récupérée |
   | Doublons | 0 | **2** : une boîte « haut du corps » emboîtée dans la boîte entière, pour deux des garçons |
   | Écart à la réalité | −1 | **+2** |

**Sur cette seconde, `n045` récupère une vraie personne et crée deux
doublons.** L'écart absolu passe de 1 à 2. L'indicateur d'inclusion le confirme
sur toute la vidéo : 15 → 26 boîtes incluses, et jusqu'à 3 à la fois.

**Verdict de non-régression (§4) : non tenu.** En base vidéo le bruit est nul,
donc tout compteur dégradé l'est au-delà du bruit : identités 7 → 12,
`OCCLUDED` 6 → 20, `REID_AMBIGUOUS` 0 → 6. Et le surcompte visible par
doublons est un vrai défaut, pas seulement un compteur interne.

**Pourquoi la NMS ne les supprime pas** : une boîte « haut du corps » incluse
dans la boîte entière a un IoU égal au rapport des aires, soit environ 0,4 à
0,5 pour une moitié de corps, sous le seuil `nms_iou` 0,55. Avant ce lot,
ces détections restaient orphelines parce que leur confiance était sous 0,7 ;
à 0,45, elles créent une piste. Ce mécanisme est déduit de la géométrie
observée, **pas mesuré** sur les confiances.

## 15.6 Lecture seuil par seuil

| Seuil | Effet sur les indicateurs du lot | Décision |
|---|---|---|
| **`match_thresh` 0,9 → 0,75** | `visible_count` : aucun gain (erreur 9, somme 143 → 159, **pire**). Fragmentation : pistes surnuméraires 6 → 13, `person_id` fragmentés 3 → 7, `TECHNICAL_ID_CHANGED` 6 → 14. `rare_occ2` : 2 changements d'identifiant technique apparus | **Écarté.** Un seuil de coût plus strict rompt des associations légitimes de personnes qui bougent peu, et les pistes rompues se recréent. C'est l'inverse de l'hypothèse du §4.1.1 de `docs/diagnostic.md`. |
| **`proximity_thresh` 0,5 → 0,8** | Seul : `visible_count` quasi inchangé (erreur 9, somme 143 → 139). Fragmentation **meilleure** : surnuméraires 6 → 4, fragmentation max 5 → 3. Avec `new_track` 0,45 : erreur 7 → 6, somme 77 → 75, mais surnuméraires 11 → 13 et identifiants partagés 2 → 3. `rare_occ2` : aucun effet | **Non appliqué, à reprendre après le lot 3.** C'est le cas que la fiche demandait de repérer : il agit sur la continuité mais pas sur ce qui est vu, et son effet s'inverse quand on le combine. Avec un descripteur peu discriminant (`model: auto`), consulter l'apparence plus souvent ne peut pas trancher proprement. |
| **`new_track_thresh` 0,7 → 0,45** | Seul levier de `visible_count` : erreur 9 → 7, somme 143 → 77, secondes à ±1 : 12 → 20, `pistes_vues` min 2 → 3. Coût : identités 18 → 38, surnuméraires 6 → 11, surcompte visible jusqu'à +2 ; `rare_occ2` dégradé (§15.5) | **Appliqué, `PROVISOIRE`, soumis à acceptation.** |
| **Combinaison des trois** (cible de la fiche) | Erreur visible 6 (vs 7 pour `n045`), somme 78 (vs 77). Coût maximal : 52 identités, 18 surnuméraires, 73 identifiants techniques ; `rare_occ2` : 16 identités | **Écartée.** Pas meilleure que `n045` sur `visible_count`, nettement pire sur la fragmentation. |

**Interaction avec la NMS (fiche §A.5).** La fiche demandait de mesurer
`new_track_thresh` 0,45 combiné au « seuil NMS relevé du lot 1.2 ». Ce seuil
n'a pas été relevé : le lot 1 a mesuré et écarté 0,85 et 0,90 (§14.6), et
`nms_iou` vaut 0,55 dans toutes les mesures de ce lot. La combinaison mesurée
est donc celle qui est en vigueur. Le risque que la fiche redoutait (des
doublons promus en identités) **se réalise quand même, sans relèvement de la
NMS** : ce sont les boîtes emboîtées du §15.5.

**Report du garde-fou (fiche §A.1).** Le report sur le filtre géométrique et
sur `timing.confirmation_seconds` est **documenté** dans le YAML, comme
demandé, et **mesuré inefficace** contre les doublons observés :
`DETECTION_REJECTED_GEOMETRY` = 0 sur les deux vidéos, quelle que soit la
variante. Une boîte « haut du corps » a un ratio plausible et une présence
continue, donc ni le ratio ni la durée de confirmation ne l'arrêtent. Aucune
de ces valeurs n'a été modifiée : les calibrer contre ce défaut serait un
correctif nouveau, hors de la fiche.

## 15.7 Position par rapport aux critères du §4

| Critère | Avant (`base`) | Après (`n045`) | Verdict |
|---|---|---|---|
| `erreur_comptage_max_visible` ≤ 1 | 9 | **7** | **non tenu**, progrès de 2 |
| `erreur_comptage_max_operational` | 4 | 5 | **non interprété** : dépend de la ligne de convention |
| `id_switches_reels` ≤ 1 | non publié | non publié | rattachement non validé |
| `fusions_reelles` non signalées = 0 | non publié | non publié | rattachement non validé |
| Non-régression `rare_occ2` | — | identités 7 → 12, doublons visibles | **non tenu** |
| FPS ≥ 3,06 ips (plancher relatif) | 4,305 | 4,176 | tenu (écart sous le bruit de 0,5 ips) |
| FPS ≥ 3,0 ips (plancher absolu) | — | — | tenu : minimum de toutes les exécutions du lot = 3,215 (`p080`, `rare_occ2`) |

**C'est le premier lot qui fait progresser le critère visible**, alors que
les lots 0 et 1 ne l'avaient pas déplacé. Mais il le fait en créant des pistes
en double, sur les deux vidéos. Ce lot **n'est donc pas un succès au sens du
§4** : un critère progresse sans être atteint, et un autre, la non-régression,
est perdu.

## 15.8 Seuils modifiés

| Paramètre | Avant | Après | Effet mesuré | Compromis accepté | Statut |
|---|---|---|---|---|---|
| `tracker.new_track_thresh` (`config/pipeline.yaml` **et** `src/configs/custom_botsort.yaml`) | 0.7 | **0.45** | §15.3 : erreur visible 9 → 7, effondrement de fin levé | identités ×2, doublons emboîtés, non-régression `rare_occ2` perdue | **`PROVISOIRE`**, soumis à acceptation |
| `tracker.match_thresh` | 0.9 | 0.9 | 0,75 mesuré et écarté | — | inchangé |
| `tracker.proximity_thresh` | 0.5 | 0.5 | 0,8 mesuré, effet mitigé | — | inchangé, **à reprendre après le lot 3** |
| `geometry.*_aspect_ratio_wh`, `min_box_*_px`, `timing.confirmation_seconds` | — | inchangés | aucun rejet sur les deux vidéos | report du garde-fou **documenté**, pas calibré | commentaires ajoutés |
| `presence.head_assist.keypoints_used` (fiche §A.3) | 5 points | 5 points | — | — | **déjà fait** : `config/pipeline.yaml:402` et les deux replis de `src/config.py` (l. 563 et l. 1137) portent déjà les 5 points. La seule occurrence restante de `("nose", "left_eye", "right_eye")` dans `src/` est un exemple de docstring (`src/geometry.py:598`), sans effet, laissé en place. |

Les valeurs par défaut des dataclasses de `src/config.py` (`new_track_thresh:
float = 0.7`, repli à 0,7 si la clé manque dans le YAML) **n'ont pas été
modifiées** : le YAML versionné porte la clé, et changer ces replis reviendrait
à toucher du code que la fiche ne demande pas de modifier. Point signalé : un
YAML sans la clé retomberait sur 0,7.

Le commentaire historique de `config/pipeline.yaml`, qui justifiait le refus
des trois seuils par une mesure sur
`test/5121204_School_Classroom_1280x720.mp4` (vidéo disparue, §11.9), est
**conservé** et annoncé comme « mesure antérieure, NON REJOUABLE ». Il est
précédé du résultat du lot 4.

## 15.9 Décision attendue de la personne responsable

Trois options, toutes réversibles (une ligne dans deux fichiers) :

1. **Accepter `new_track_thresh` 0,45 en l'état**, avec la non-régression de
   `rare_occ2` perdue. Les doublons emboîtés deviennent un défaut connu, à
   traiter au lot 5 (retouches locales), par exemple par une suppression des
   boîtes incluses dans une boîte de la même classe. Ce correctif n'est **pas**
   fait ici.
2. **Refuser, et revenir à 0,7.** Le lot 4 est alors conclu sans aucun seuil
   modifié, et `visible_count` reste à 9 pour le lot 2.
3. **Accepter à titre de mesure seulement** : garder 0,45 pour que les lots 2 et
   3 se mesurent sur des pistes qui survivent, puis décider à la clôture en
   fonction du lot 5.

Je n'ai pas tranché à votre place. La valeur est appliquée et commitée parce
que la fiche prescrit de modifier le YAML et que le §5 autorise une valeur
`PROVISOIRE` réversible pour débloquer une mesure. Revenir à 0,7 annule
entièrement le lot.

## 15.10 Tests

Suite complète : **664 tests collectés, 663 passés, 1 ignoré**, couverture
**93,39 %** (seuil 85 %). Le test ignoré (`test_calibration_window.py:227`,
serveur X) l'était déjà. Décompte réel (`grep -rc "^def test_" tests/`) :
**469** fonctions sur **34** fichiers, inchangé depuis le lot 1.

### 15.10.1 Test corrigé — un

| Test | Attente précédente | Pourquoi elle a changé | Correction |
|---|---|---|---|
| `test_config_validation.py::test_yolo_threshold_is_low_enough_for_the_tracker_low_stage` | `new_track_thresh == 0.7` | La valeur de production change par décision de ce lot. Le test la figeait comme constante, alors que l'exigence qu'il porte est la cohérence entre les seuils. | Attente portée à 0,45, et ajout de l'invariant `new_track_thresh >= track_high_thresh`. Les autres assertions (seuil YOLO ≤ plancher bas, aucun avertissement de cohérence) sont inchangées. |

Les tests qui portent l'exigence de fond passent **sans modification** :
`test_track_loss_recovery.py::test_new_track_threshold_stays_strict_despite_the_low_yolo_gate`
(0,45 > 0,4 > 0,10), et le test de synchronisation des deux YAML
(`test_config_validation.py`, l. 136-143).

### 15.10.2 Tests ajoutés — aucun

Le lot ne modifie que des valeurs de configuration. L'indicateur d'inclusion
du §15.3.2 est un script d'analyse ad hoc, non versionné : il n'entre dans
aucune décision automatique.

## 15.11 Ce qui reste non résolu

1. **Doublons emboîtés** (§15.5) : c'est le coût de `new_track_thresh` 0,45 et
   la cause de la perte de non-régression. Ni la NMS à 0,55, ni le filtre
   géométrique, ni `confirmation_seconds` ne les arrêtent. Non corrigé, à
   rattacher au lot 5 si l'option 1 ou 3 du §15.9 est retenue.
2. **`erreur_comptage_max_visible` vaut encore 7** : l'écart résiduel est entre
   s20 et s24 (5–7 visibles pour 12 personnes).
3. **`proximity_thresh`** : à remesurer après le lot 3. Son effet sur la
   fragmentation est réel, seul (6 → 4 pistes surnuméraires), mais s'inverse
   quand on le combine.
4. **Identités créées ×2 sur `fort_occ4`** (18 → 38 pour 13 personnes) : la
   galerie de niveau 2 ne rattache pas les nouvelles pistes aux identités
   existantes (`REID_AMBIGUOUS` 4 → 22). C'est le domaine du lot 3.
5. **Métriques d'identité toujours non publiées** : le rattachement doit être
   validé au lot 2 ; sans lui, `id_switches_reels` et `fusions_reelles` ne
   peuvent pas être opposés au §4.
6. **Bruit de FPS de 0,5 ips** entre rejouages identiques (§15.2) : les
   comparaisons de FPS entre lots demandent désormais plusieurs rejouages, ce
   qui compte pour le lot 6 où la marge est faible.
7. **Ligne de convention** : `IN`, `OUT`, `NEW` et `occupancy_operational` ne
   s'interprètent pas tant qu'elle reste au-dessus du trajet des pieds
   (`results/line_check/`). Changer de ligne relève d'une décision humaine
   (§2 de `CLAUDE.md` : ligne figée pour les lots 1 à 7).
8. **Reportés et inchangés** : la ligne n'est pas publiée dans
   `config_resolved.yaml` (lot 0) ; `bootstrap_frames` est encore en frames
   (lot 1, dette du lot 2) ; `track_buffer_seconds` 15,0 est toujours
   `PROVISOIRE`.

## 15.12 Filtre « boîte incluse dans une boîte plus confiante » — **écarté**

> **Décision de la personne responsable, 2026-09-21 : écarté, chiffres à
> l'appui.** Simulation en lecture seule, **aucun correctif** : aucune ligne de
> code ni de configuration n'a été modifiée pour cette section.

### 15.12.1 Ce qui a été simulé

La règle simulée : supprimer toute boîte vue incluse à **plus de 80 %** dans une
autre boîte vue de la même frame et **plus confiante**. Elle vise les doublons
du §15.5, c'est-à-dire une boîte partielle emboîtée dans la boîte entière de la
même personne.

- **Données** : les relevés par seconde avec `new_track_thresh` 0,45 (36 frames
  de `fort_occ4`, 25 de `rare_occ2`), sur les boîtes que le pipeline associe aux
  pistes. C'est une **approximation** d'un filtre réel, qui agirait sur chaque
  frame et en amont du tracker ; retirer une détection change aussi le suivi
  des frames suivantes, ce que la simulation ne reproduit pas.
- **Classement doublon / personne réelle** : à l'œil, sur des découpes de la
  vidéo brute où la boîte incluse et la boîte hôte sont tracées
  (`results/lot4/filtre_*_p*.jpg`, non versionnés : personnes identifiables).
  Deux cas sur 46 sont incertains et signalés.
- Scripts ad hoc, non versionnés : `results/lot4/filter_candidates.py`,
  `filter_missed.py`, `filter_effect.py`, `label_pairs.py`.

### 15.12.2 Résultat à 0,45

| | `fort_occ4` | `rare_occ2` |
|---|---|---|
| Boîtes que le filtre supprimerait | 22 | 24 |
| … dont **doublons** (suppression correcte) | **15** (1 incertain) | **9** |
| … dont **personnes réelles** (suppression à tort) | **7** (1 incertain) | **15** |
| Doublons laissés en place (la boîte partielle est la plus confiante) | 1 sur 16 | 2 sur 11 |
| Somme des écarts visibles à la vérité terrain | 77 → **89** (pire) | — |
| Secondes où \|écart visible\| ≤ 1 | 20 → **15** | — |
| Pistes visibles au maximum | — | 7 → 5 |

**Qui sont les personnes supprimées à tort :**

- **`rare_occ2`** : pour 13 des 15, la boîte hôte est celle de l'**enseignant,
  filmé de dos au premier plan**. Sa boîte couvre presque la moitié de l'image,
  donc tout enfant placé derrière lui y est inclus. Les deux autres sont une
  enfant derrière un camarade, à deux secondes différentes.
- **`fort_occ4`** : des personnes debout dans l'embrasure ou assises au fond,
  masquées par quelqu'un qui passe devant elles.

**Même à 0,7, le filtre ferait des dégâts** : sur `fort_occ4`, il supprimerait 2
personnes réelles et aucun doublon ; sur `rare_occ2`, 14 personnes réelles
pour 1 doublon.

**Conclusion.** Le critère « incluse + moins confiante » ne distingue pas un
doublon d'une personne masquée derrière une autre. Le masquage est justement
la situation que le projet doit traiter (§1 de `CLAUDE.md`). Sur la scène
calme, le filtre ferait plus de mal que de bien. **Écarté.**

### 15.12.3 Origine des doublons de `fort_occ4` : le détecteur, pas une piste Kalman

Sur `fort_occ4`, la plupart des doublons ne ressemblent pas à des boîtes
« haut du corps ». Ce sont **deux pistes aux boîtes quasi identiques** (IoU des
boîtes publiées : 0,75 à 0,95) sur la même personne assise : l'homme au t-shirt
vert (`technical_track_id` 50 et 55, s29 à s35) et la jeune femme en rouge (49
et 56, s31 à s35). Une NMS à 0,55 aurait dû en supprimer une. Hypothèse
examinée : une piste perdue maintenue par le filtre de Kalman pendant qu'une
nouvelle naît sur la même personne.

**Hypothèse réfutée.** Vérification en lecture seule
(`results/lot4/dup_origin.py`) : `seen_this_frame` des deux pistes, puis
inférence YOLO **brute** sur les mêmes frames, avec les réglages du pipeline
(`conf` 0,10, `iou` 0,55, `imgsz` 960, classe 0).

| Constat | Mesure |
|---|---|
| `seen_this_frame` des deux pistes de chaque doublon | **vrai pour les deux, dans les 16 cas** : aucune n'est une piste maintenue sans observation |
| Détections brutes de YOLO sur la personne | **deux par frame** : une boîte **entière** (hauteur ≈ 150 à 200 px) et une boîte **tête-épaules** (hauteur ≈ 70 à 85 px, une fois 123 px), dont le haut coïncide avec celui de la boîte entière |
| IoU entre ces deux détections | **0,31 à 0,50** : sous `nms_iou` 0,55, la NMS les conserve **toutes les deux** |
| Confiance des détections tête-épaules | 0,13 à 0,59, donc au-dessus de 0,45 pour une partie d'entre elles : elles créent une piste |
| Piste associée à la détection tête-épaules | identifiée par sa confiance (exemple à s30 : piste 50, confiance 0,591 = détection tête-épaules 0,59 ; piste 55, 0,367 = détection entière 0,37) |
| Boîte **publiée** pour cette piste | **pleine hauteur** (s30, piste 50 : `[988, 377, 1052, 537]`, alors que sa détection vaut `[988, 377, 1052, 448]`) |

**Ce que cela établit.** Le doublon **naît dans le détecteur** : deux boîtes
par personne assise, trop peu recouvrantes pour la NMS. Il **devient une
piste** parce que `new_track_thresh` est abaissé. Le tracker ne le crée pas, il
le **masque** : la boîte publiée pour la piste tête-épaules garde la hauteur
d'une personne entière. C'est pour cela que ces doublons apparaissent comme
« quasi identiques » dans le relevé, et non comme emboîtés.

**Déduit, non vérifié dans le code d'Ultralytics** : la boîte publiée semble
être l'état lissé du filtre de Kalman de la piste, et non la détection
associée. La hauteur lissée de la piste 50 conserverait ainsi une valeur
antérieure. Le constat ci-dessus ne dépend pas de cette explication.

**Conséquences pour une suite éventuelle, sans correctif ici :**

1. Remonter `nms_iou` n'aide pas, puisque les deux boîtes sont sous 0,55.
   L'abaisser jusqu'à 0,30 les fusionnerait, mais supprimerait aussi des
   personnes réelles côte à côte (le lot 1 a montré que la NMS agit
   directement sur ce cas, §14.6).
2. Sur les détections brutes, la boîte tête-épaules est incluse à 100 % dans
   la boîte entière et partage son bord haut. C'est une signature plus étroite
   que l'inclusion seule, mais **elle ne suffirait pas** à épargner le cas de
   l'enseignant : à s6 de `rare_occ2`, l'enfant placé derrière lui a le même
   bord haut que sa boîte (156 px pour les deux). *Correction : la première
   version de ce paragraphe affirmait le contraire sans l'avoir vérifié.*
   **Non mesurée.** Si elle devait l'être, ce serait au lot 5 (retouches
   locales), et après le lot 3.
3. Ces boîtes tête-épaules sont précisément ce que le §8 de `CLAUDE.md` met
   en réserve (« détection dédiée têtes / haut du corps »). Elles sont ici un
   sous-produit non voulu de `yolo11n`, pas une détection dédiée.

## 15.13 Décision du 2026-09-21 : `new_track_thresh` = **0,60**

> **Ce paragraphe remplace les §15.7 à §15.9 pour la valeur appliquée.** Ces
> sections restent en place parce qu'elles documentent la mesure à 0,45 ; la
> valeur en vigueur est 0,60 (`PROVISOIRE`).

La personne responsable a demandé deux valeurs de plus, 0,55 et 0,60, puis a
retenu **0,60** : le gain visible réel est acquis dès 0,60, et descendre plus
bas quadruple les doublons pour un gain marginal.

### 15.13.1 Mesure de 0,60 et 0,55

Relevés par seconde, base vidéo, lancés en parallèle (les compteurs sont
déterministes, §15.2). Le FPS de ces exécutions n'est pas utilisé ; celui de
0,60 est donné par la confirmation du §15.13.4. « Doublons emboîtés » : paires
de boîtes vues incluses à plus de 80 % l'une dans l'autre, classées à l'œil
comme deux boîtes sur **la même** personne (méthode du §15.12). « Net » : après
retrait de ces doublons.

**`fort_occ4`**

| | 0,7 (avant) | **0,60** | 0,55 | 0,45 |
|---|---|---|---|---|
| `erreur_comptage_max_visible` brute / nette | 9 / 9 | **8 / 8** | 7 / 7 | 7 / 7 |
| Somme des \|écarts visibles\|, brute / nette | 143 / 144 | **99 / 102** | 88 / 94 | 77 / 81 |
| Secondes où \|écart\| ≤ 1 (nettes, sur 35) | 12 | **13** | 14 | 17 |
| Surcompte visible maximal (brut) | +1 | **+1** | +1 | +2 |
| `visible_count` brut à s16 / s20 / s24 / s28 | 7 / 5 / 3 / 2 | **8 / 6 / 4 / 5** | 9 / 6 / 5 / 5 | 10 / 7 / 5 / 6 |
| `visible_count` net de s29 à s34 (10 à 11 visibles annotées) | 4–5 | **7–9** | 7–9 | 8–10 |
| `pistes_vues` min / max | 2 / 9 | **3 / 10** | 3 / 11 | 3 / 13 |
| **Doublons emboîtés** | 1 | **4** | 7 | 16 |
| Personnes réelles emboîtées (masquées derrière une autre) | 2 | 7 | 8 | 8 |
| Identités créées | 18 | **22** | 27 | 38 |
| `technical_ids` | 21 | **32** | 38 | 46 |
| Fragmentation max / pistes surnuméraires | 5 / 6 | **4 / 12** | 4 / 14 | 4 / 11 |
| `person_id` fragmentés / identifiants techniques partagés | 3 / 2 | **9 / 3** | 8 / 3 | 9 / 2 |
| `TECHNICAL_ID_CHANGED` | 6 | **14** | 15 | 12 |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 1 | **4** | 4 | 5 |
| `IDENTITY_SWAP_CORRECTED` | 0 | **0** | 0 | 0 |
| `REID_AMBIGUOUS` | 4 | **8** | 14 | 22 |
| `PURGE` | 14 | **15** | 18 | 28 |
| `OCCLUDED` | 52 | **70** | 78 | 84 |
| `POSSIBLE_MULTI_PERSON_BOX` | 55 | **66** | 66 | 52 |
| `INCONSISTENT_STATE` | 9 | **13** | 11 | 20 |
| `DETECTION_REJECTED_GEOMETRY` | 0 | **0** | 0 | 0 |
| *`IN` / `OUT` / `NEW` — non interprété* | *2 / 0 / 7* | ***6 / 0 / 5*** | *7 / 0 / 10* | *8 / 0 / 7* |
| *`occupancy_operational` / `observed` — non interprété* | *12 / 2* | ***14 / 7*** | *20 / 7* | *18 / 6* |
| *`erreur_comptage_max_operational` — non interprété* | *4* | ***4*** | *7* | *5* |

**`rare_occ2`** (pas de vérité terrain ; 5 personnes réellement visibles à s20,
comptées à l'œil)

| | 0,7 (avant) | **0,60** | 0,55 | 0,45 |
|---|---|---|---|---|
| Pistes visibles à s20, brutes / nettes | 4 / 4 | **7 / 5** | 7 / 5 | 7 / 5 |
| Pistes visibles au maximum, brutes / nettes | 5 / 4 | **7 / 5** | 7 / 5 | 7 / 5 |
| **Doublons emboîtés** | 1 | **7** | 9 | 11 |
| Identités créées / `technical_ids` | 7 / 7 | **10 / 10** | 10 / 11 | 12 / 12 |
| Pistes surnuméraires (relevé par seconde) | 0 | **0** | 1 | 0 |
| `TECHNICAL_ID_CHANGED` | 0 | **0** | 1 | 0 |
| `REID_AMBIGUOUS` | 0 | **4** | 4 | 6 |
| `PURGE` | 2 | **2** | 2 | 3 |
| `OCCLUDED` | 6 | **13** | 18 | 20 |
| `INCONSISTENT_STATE` | 0 | **1** | 1 | 1 |
| *`IN` / `OUT` / `NEW` — non interprété* | *4 / 1 / 2* | ***4 / 1 / 2*** | *6 / 3 / 2* | *5 / 2 / 2* |
| *`occupancy_operational` / `observed` — non interprété* | *5 / 3* | ***5 / 3*** | *5 / 3* | *5 / 2* |

### 15.13.2 Lecture

- **Le gain visible réel est acquis dès 0,60.** Une fois les doublons retirés,
  les trois valeurs abaissées voient les 5 personnes de `rare_occ2`, là où la
  production en voit 4. Sur `fort_occ4`, 0,60 fait passer la fin de séquence
  de 4–5 à 7–9 personnes nettes, et 0,45 n'ajoute qu'une personne nette de plus
  (8–10).
- **Le coût croît vite sous 0,60** : doublons emboîtés 4 → 7 → 16 sur
  `fort_occ4`, identités 22 → 27 → 38.
- **Sur `rare_occ2`, les doublons apparaissent dès 0,60** (1 → 7), toujours sur
  les deux mêmes garçons (boîte partielle, sans les pieds, dans la boîte
  entière). Aucune des trois valeurs ne les évite.
- **Écart résiduel** : `erreur_comptage_max_visible` vaut 8 à 0,60 pour une
  cible ≤ 1. Le creux se situe entre s20 et s28 (4 à 7 personnes vues pour 10 à
  12 visibles, écart maximal à s23–s24 : 4 pour 12). Ce lot ne l'atteint pas.

### 15.13.3 Réserve inscrite — non-régression de `rare_occ2`

**Acceptée en connaissance de cause par la personne responsable
(2026-09-21).** Au sens strict du §4 (« aucun compteur dégradé au-delà du
bruit », bruit nul en base vidéo), la non-régression de `rare_occ2` **n'est pas
tenue** à 0,60 :

| Ce qui se dégrade | 0,7 → 0,60 |
|---|---|
| Identités créées | 7 → 10 |
| Doublons emboîtés (pistes visibles en double) | 1 → 7 |
| Pistes visibles au maximum, brutes | 5 → 7, pour 5 personnes |
| `OCCLUDED` / `REID_AMBIGUOUS` | 6 → 13 / 0 → 4 |

| Ce qui s'améliore — la mesure qui compte | 0,7 → 0,60 |
|---|---|
| Personnes réellement présentes et suivies, à s20 | **4 sur 5 → 5 sur 5** |

**Condition de réouverture** : un correctif des doublons tête-épaules / boîte
partielle (§15.12.3, point 2), s'il est traité au lot 5, doit ramener les
doublons emboîtés de `rare_occ2` près de leur niveau de production (1) sans
reperdre la cinquième personne. Sinon, la réserve reste ouverte jusqu'à la
clôture et doit figurer dans `docs/rapport_final.md`.

### 15.13.4 Confirmation sur la configuration versionnée

`scripts/measure_corpus.py`, **sans** `--config`, donc `config/pipeline.yaml` et
`src/configs/custom_botsort.yaml` tels que commités, une vidéo à la fois, seul
sur la machine :

| Vidéo | Compteurs (`technical_ids` / identités / `TECHNICAL_ID_CHANGED` / `PURGE` / `OCCLUDED` / `STATE_TRANSITION` / `IN` / `OUT` / `NEW` / `occupancy_operational`) | Contre le relevé à 0,60 | FPS moyen |
|---|---|---|---|
| `fort_occ4` | 32 / 22 / 14 / 15 / 70 / 218 / 6 / 0 / 5 / 14 | **identiques** | 3,737 |
| `rare_occ2` | 10 / 10 / 0 / 2 / 13 / 91 / 4 / 1 / 2 / 5 | **identiques** | 3,191 ; rejoué : **4,151** (compteurs identiques) |

La configuration versionnée reproduit exactement la variante mesurée.

**FPS** : le 3,191 de la première exécution sur `rare_occ2` n'est qu'à 0,13 ips
du plancher relatif de 3,06. Le rejouage donne 4,151 avec les mêmes compteurs :
c'est le bruit de 0,5 à 1 ips déjà relevé au §15.2, pas un effet du seuil.
**Critère FPS tenu** sur toutes les exécutions du lot (minimum 3,191), mais
avec une marge qu'une exécution unique ne suffit pas à établir. Pour le
lot 6, où la marge est faible, il faudra plusieurs rejouages par mesure de FPS.

### 15.13.5 Seuils — état final du lot 4

| Paramètre | Avant | Après | Statut |
|---|---|---|---|
| `tracker.new_track_thresh` (`config/pipeline.yaml` **et** `src/configs/custom_botsort.yaml`) | 0.7 | **0.60** | `PROVISOIRE` — calibré sur l'ensemble de test ; valeurs 0,55 et 0,45 mesurées et écartées |
| `tracker.match_thresh` | 0.9 | 0.9 | 0,75 mesuré et écarté |
| `tracker.proximity_thresh` | 0.5 | 0.5 | 0,8 mesuré, **à reprendre après le lot 3** |
| Filtre géométrique, `timing.confirmation_seconds` | — | inchangés | report du garde-fou documenté, mesuré inefficace contre les doublons |
| Filtre d'inclusion (§15.12) | — | non implémenté | **écarté** |

### 15.13.6 Tests

Suite complète : **664 tests collectés, 663 passés, 1 ignoré** (`test_calibration_window.py:227`, serveur X, déjà ignoré), couverture **93,39 %**. Décompte réel (`grep -rc "^def test_" tests/`) : **469** fonctions sur **34** fichiers, inchangé.

Un seul test ajusté, le même qu'au §15.10.1 :
`test_config_validation.py::test_yolo_threshold_is_low_enough_for_the_tracker_low_stage`.
Son attente passe de 0,45 à **0,60**, pour la même raison (la valeur de
production change par décision). L'invariant `new_track_thresh >=
track_high_thresh` qu'il vérifie est inchangé (0,60 ≥ 0,4).

### 15.13.7 Ce qui reste non résolu (complète le §15.11)

1. **`erreur_comptage_max_visible` = 8** (cible ≤ 1) : creux entre s20 et s28.
2. **Doublons tête-épaules** : 4 sur `fort_occ4` et 7 sur `rare_occ2` à 0,60.
   Leur origine est établie (§15.12.3), aucun correctif n'est fait.
3. **Réserve de non-régression de `rare_occ2`** (§15.13.3), ouverte.
4. **`proximity_thresh`** à reprendre après le lot 3. **Métriques d'identité**
   toujours non publiées (rattachement à valider au lot 2).
5. **Correction apportée au §15.12.3, point 2**, dans le même commit que cette
   section : la première version affirmait sans vérification qu'une signature
   « bord haut partagé » épargnerait le cas de l'enseignant ; c'est faux (s6).

## 15.14 Diagnostic préalable au lot 2 — pourquoi les personnes ne sont pas redétectées

> Lecture seule, aucun correctif. Configuration en vigueur (`new_track_thresh`
> 0,60). Script et images : `results/lot4/diag_redetection.py`,
> `results/lot4/diag/` (non versionnés : personnes identifiables).

**Méthode.** Pour chaque seconde de `fort_occ4`, à l'image relevée :
inférence YOLO brute avec les réglages du pipeline, `conf` abaissé à 0,01 ;
retrait des détections déjà portées par une piste vue ; localisation à l'œil de
chaque personne visible annotée sans piste, et meilleure détection à son
emplacement. « Manquantes » = visibles annotées − pistes visibles, doublons
retirés.

| Cas | s20 à s28 (52 manquantes) | s29 à s34 (16) |
|---|---|---|
| Détectée ≥ 0,60 (pas bloquée par un seuil) | 0 | 1 |
| Détectée entre 0,40 et 0,60 : bloquée par `new_track_thresh` | 6 (12 %) | 5 (31 %) |
| Détectée entre 0,10 et 0,40 : bloquée **aussi** par `track_high_thresh` (0,40) | 27 (52 %) | 8 (50 %) |
| Détectée sous 0,10 seulement (écartée par `model.confidence`) | 11 (21 %) | 2 (12 %) |
| Aucune détection, même à 0,01 | 1 (2 %) | 0 |
| Non localisée à l'œil (s20 à s23, foule très dense) | 7 (13 %) | 0 |

**Lecture.**

- **Le premier frein est le seuil de création du tracker, pas le détecteur** :
  63 % des manquantes de s20 à s28, 81 % en fin de vidéo, sont détectées mais
  sous 0,60. La part du détecteur (sous 0,10 ou rien) est de 23 % puis 12 %.
- **La moitié des cas est sous 0,40**, donc hors de portée de
  `new_track_thresh` seul : la validation de la configuration impose
  `new_track_thresh ≥ track_high_thresh`.
- **Cas type du symptôme d'origine** : la femme en rouge assise à gauche. Elle
  est visible de s30 à s35 et détectée entre 0,34 et 0,47 à chaque seconde,
  mais n'a jamais de piste. Sa piste (`person_id` 19) a sauté sur l'enfant
  assis devant elle à s30.
- **Part du détecteur** : surtout la posture, par exemple la personne accroupie
  sous une table de s23 à s28 (0,04 à 0,06, ou rien).

**Limites** : une image par seconde ; attribution visuelle dans une foule dense
(incertitude d'environ ±1 personne par seconde) ; les boîtes couvrant deux
personnes ne sont attribuées à personne.

## 15.15 `track_high_thresh` 0,4 → **0,30** (décision du 2026-09-21)

> Complément du lot 4, décidé par la personne responsable après le diagnostic
> du §15.14. `PROVISOIRE`. **0,25 mesuré et écarté.**

### 15.15.1 Mécanisme visé

Vérifié dans le code d'Ultralytics 8.4.126 (`trackers/byte_tracker.py`) :

- les pistes **perdues** ne participent qu'à la **première association**
  (l. 282), qui ne reçoit que les détections ≥ `track_high_thresh` (l. 316) ;
- le **second étage** confronte les détections faibles aux seules pistes
  **encore suivies** (l. 427) ;
- une piste n'est **créée** que si la confiance est ≥ `new_track_thresh`
  (l. 479).

À 0,30, une détection entre 0,30 et 0,40 peut donc rejoindre sa piste perdue,
sans pouvoir créer d'identité : `new_track_thresh` reste à 0,60. L'invariant
`new_track_thresh (0,60) ≥ track_high_thresh (0,30) > track_low_thresh (0,1) ≥
model.confidence (0,10)` est tenu ; la configuration se valide sans
avertissement.

### 15.15.2 Mesure

Relevés par seconde, base vidéo, `new_track_thresh` 0,60 dans les trois
colonnes. Doublons et « net » : méthode du §15.12, classement visuel des
paires nouvelles.

**`fort_occ4`**

| | 0,40 (avant) | **0,30** | 0,25 |
|---|---|---|---|
| `erreur_comptage_max_visible` brute / nette | 8 / 8 | **7 / 7** | 8 / 8 |
| Somme des \|écarts visibles\| brute / nette | 99 / 102 | **95 / 100** | 96 / 105 |
| Secondes où \|écart\| ≤ 1 (nettes, sur 35) | 13 | 13 | 13 |
| `pistes_vues` min / max | 3 / 10 | 3 / 10 | 3 / 10 |
| Fragmentation max / pistes surnuméraires | 4 / 12 | **2 / 7** | 2 / 5 |
| `person_id` fragmentés | 9 | 7 | 5 |
| `TECHNICAL_ID_CHANGED` | 14 | **6** | 6 |
| `technical_ids` | 32 | 32 | 34 |
| Identités créées | 22 | **29** | 31 |
| … dont `reid_ambiguous_provisional` | 8 | **14** | 16 |
| Doublons emboîtés | 4 | **8** | 12 |
| `REID_AMBIGUOUS` / `PURGE` / `OCCLUDED` | 8 / 15 / 70 | 14 / 20 / 78 | 16 / 22 / 85 |
| *`IN` / `OUT` / `NEW` / `occupancy_operational` — non interprété* | *6 / 0 / 5 / 14* | *7 / 0 / 8 / 18* | *6 / 0 / 8 / 17* |

**Cas type du §15.14 — la personne assise à gauche, s26 à s35**

| | s26 à s29 | s30 à s35 |
|---|---|---|
| 0,40 | suivie | **perdue** : sa piste passe sur l'enfant devant elle |
| **0,30** | suivie | **suivie sans interruption, même piste technique**, détections entre 0,34 et 0,47 |
| 0,25 | suivie à s26 seulement | perdue dès s27 : les détections de l'enfant (0,25 à 0,37) entrent en première association et prennent sa piste |

**`rare_occ2`**

| | 0,40 | **0,30** | 0,25 |
|---|---|---|---|
| Pistes visibles nettes, max (5 personnes réelles) | 5 | 5 | 5 |
| Identités créées / `technical_ids` | 10 / 10 | 10 / 11 | 10 / 10 |
| Doublons emboîtés | 7 | 9 | 7 |
| Fragmentation max / pistes surnuméraires | 1 / 0 | 2 / 1 | 1 / 0 |

### 15.15.3 Lecture et décision

- **Continuité des pistes nettement meilleure** à 0,30 : les changements
  d'identifiant technique sont divisés par plus de deux, la fragmentation est
  presque divisée par deux, et le cas type est résolu.
- **Gain modeste sur `visible_count`** : 8 → 7. Seule la part du diagnostic
  entre 0,30 et 0,40 est concernée, et seulement quand une piste perdue est
  assez proche pour être retrouvée.
- **0,25 écarté** : pas de gain visible, 12 doublons, et le cas type redevient
  un échec, parce que les détections faibles d'une personne voisine volent la
  piste.

### 15.15.4 Réserve inscrite — identités provisoires et doublons

Acceptée par la personne responsable le 2026-09-21. **À traiter au lot 3.**

| Ce qui se dégrade (0,40 → 0,30, `fort_occ4`) | |
|---|---|
| Identités créées | 22 → 29 |
| … dont identités provisoires (`reid_ambiguous_provisional`) | 8 → 14 |
| Doublons emboîtés | 4 → 8 |
| `rare_occ2` : doublons / pistes surnuméraires | 7 → 9 / 0 → 1 |

**Attribution** : le nombre d'identifiants techniques est inchangé (32). Les
identités en plus ne viennent donc pas du tracker : ce sont surtout des
identités **provisoires** de la galerie, créées quand l'appariement est ambigu.
Interprétation, non vérifiée : les pistes vivent plus longtemps, la galerie
compte donc plus de candidats simultanés, et le descripteur HSV ne les départage
pas (§4.1.4 de `docs/diagnostic.md`). C'est exactement le périmètre du lot 3
(descripteur OSNet, puis galerie multi-échantillons).

### 15.15.5 Confirmation sur la configuration versionnée

`scripts/measure_corpus.py` sans `--config`, une vidéo à la fois, seul sur la
machine :

| Vidéo | `technical_ids` / identités / `TECHNICAL_ID_CHANGED` / `IN` / `OUT` / `NEW` / `occupancy_operational` / `PURGE` / `OCCLUDED` / `STATE_TRANSITION` / `REID_AMBIGUOUS` | Contre le relevé à 0,30 | FPS moyen |
|---|---|---|---|
| `fort_occ4` | 32 / 29 / 6 / 7 / 0 / 8 / 18 / 20 / 78 / 248 / 14 | **identiques** | 3,320 |
| `rare_occ2` | 11 / 10 / 1 / 4 / 0 / 2 / 6 / 2 / 15 / 104 / 4 | **identiques** | 3,821 |

Critère FPS (≥ 3,06) tenu. La marge de `fort_occ4` (0,26 ips) reste dans le
bruit de 0,5 à 1 ips relevé au §15.2 : non concluant sur une seule exécution,
à surveiller au lot 3, qui ajoute un réseau par personne suivie.

### 15.15.6 Tests

Suite complète : **664 tests collectés, 663 passés, 1 ignoré** (`test_calibration_window.py:227`, serveur X, déjà ignoré), couverture **93,39 %**. Décompte réel (`grep -rc "^def test_" tests/`) : **469** fonctions sur **34** fichiers, inchangé. Deux tests corrigés, aucun ajouté :

| Test | Correction | Raison |
|---|---|---|
| `test_config_validation.py::test_yolo_threshold_is_low_enough_for_the_tracker_low_stage` | attente `track_high_thresh` 0,4 → **0,30**, et ajout de l'invariant `track_low_thresh < track_high_thresh` | la valeur de production change par décision ; l'exigence de fond (ordre des seuils) est désormais vérifiée explicitement |
| `test_track_loss_recovery.py::test_threshold_matrix_on_the_same_blurred_scenario`, cas `(0,10 ; 0,30 ; 0,12 ; False)` | `track_low_thresh` d'ablation 0,30 → **0,20** | le cas simulait un plancher d'association au-dessus du flou (0,12) ; avec `track_high_thresh` à 0,30, la valeur 0,30 viole l'ordre strict `track_low < track_high` validé par `src/config.py`, et le test levait une `ConfigError` au lieu de vérifier son scénario. 0,20 garde l'intention (au-dessus de 0,12, sous 0,30). Aucun test supprimé ni ignoré |
