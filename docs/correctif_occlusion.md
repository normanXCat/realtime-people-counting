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
