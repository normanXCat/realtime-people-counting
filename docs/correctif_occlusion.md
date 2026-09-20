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
