# Correctif occlusion dense et assistance tête — lots A et B

Ce document répond au prompt `prompt_correctif_occlusion_et_tete.md` (lots A et
B ; lot C **non exécuté**). Il contient les mesures avant/après, la justification
de chaque seuil modifié, les tests ajoutés ou corrigés, et **ce qui reste
ouvert**.

> Règle appliquée ici : aucune conclusion n'est écrite si la mesure ne la
> soutient pas. Les points non mesurables faute de corpus annoté sont signalés
> comme tels plutôt que déclarés acquis.

---

## 0. Écart de protocole à signaler d'emblée

Le prompt impose « un lot à la fois : ne pas enchaîner le lot B avant que le lot A
ait été mesuré ». **Cet ordre n'a pas été respecté** : les lots A et B ont été
implémentés dans la même session de travail, à la demande explicite de portée
(« Lot A + Lot B »). Conséquence directe sur les mesures :

- les exécutions « après » reflètent **A + B.3.b + B.4** (les deux seuls
  changements B actifs avec `presence.head_assist.enabled: false`) ;
- il n'existe **pas** de tableau « lot A seul ». Le présent document ne prétend
  donc pas isoler l'effet du lot A ; il compare l'état **avant correction**
  (commit `4208951`) à l'état **après corrections A+B**.

Les changements B.1, B.2 et B.5 sont, par construction, **sans effet** tant que
`presence.head_assist.enabled` est `false` (branches conditionnées par ce
drapeau) : les runs « après » à assistance désactivée mesurent A + B.3.b + B.4.

---

## 1. Tableaux avant/après

### 1.1 Protocole principal — même vidéo, même ligne, 60 frames (§4.2)

- source : `test/video3.mp4` (1280×720) ;
- ligne **identique** : celle archivée de la session
  `results/20260917T180747Z/calibration.json`
  (`p1 = (0.190741, 0.996705)`, `p2 = (0.364815, 0.907743)`, intérieur
  `negative`) ;
- 60 frames, YOLO11n, `imgsz = 960`, CPU ;
- « avant » = `git archive HEAD` (commit `4208951`, correctifs absents) ;
- « après » = arbre courant ;
- la sélection manuelle de ligne est injectée par un pilote temporaire (aucune
  ligne par défaut n'existe ; c'est la même ligne validée qui est rejouée).

| Indicateur | Avant | Après |
|---|---:|---:|
| `TRACK_ID_APPEARANCE_DISCONTINUITY` (proxy de swap) | 0 | 0 |
| `IDENTITY_SWAP_CORRECTED` | 0 | 0 |
| `TECHNICAL_ID_CHANGED` | 0 | 0 |
| `REID_NEW` | 4 | 5 |
| Identités distinctes créées | 4 | 5 |
| `PURGE` | 0 | 0 |
| `POSSIBLE_MULTI_PERSON_BOX` | 6 | 6 |
| dont signal `width_growth` / `head_count` | 6 / 0 | 6 / 0 |
| `PRESENCE_MAINTAINED_BY_HEAD` | 0 | 0 |
| `PRESENCE_EXTENSION_EXPIRED` | 0 | 0 |
| Événements écrits | 62 | 59 |
| FPS moyen mesuré | 4,14 | 2,05 |

`[BILAN]` (identique dans les deux cas) :

```
[BILAN] OCCUPATION_CONFIRMEE=0 OBSERVEE=0 INCERTAINE=0 RANGE=[0, 0] IN=0 OUT=0 NEW=0
```

Latences P95 — avant : `capture=1082,5ms detection=730,2ms tracking=37,0ms
render=10,6ms reid=5,6ms fsm=3,8ms` ; après : `capture=796,1ms
detection=664,7ms tracking=21,9ms render=17,4ms reid=7,8ms fsm=3,7ms`.

**Lecture.** Sur ces 60 premières frames, aucun des quatre symptômes visés n'est
reproduit : **zéro** discontinuité d'apparence, **zéro** correction, **zéro**
changement d'identifiant technique. La comparaison ne montre donc **aucun effet
mesurable** du correctif sur ce segment. Le seul écart notable est le nombre
d'identités créées (4 → 5), non significatif sur 60 frames à une cadence de 2 à
4 FPS (l'écart de FPS entre les deux runs — 4,14 vs 2,05 — montre d'ailleurs que
ces deux exécutions ne sont pas reproductibles au frame près ; voir §4).

### 1.2 Mesure dédiée à la stabilité d'identité — 300 frames, salle de classe

`test/5121204_School_Classroom_1280x720.mp4`, 300 frames, via
`scripts/measure_tracker_swaps.py` (même chemin de code que le pipeline :
YOLO + BoT-SORT + `IdentityManager`, sans sélection de ligne). C'est la mesure
qui correspond réellement aux symptômes 1 et 3 du prompt (scène dense, personnes
assises).

| Indicateur | Avant | Après |
|---|---:|---:|
| Détections observées | 778 | 817 |
| `technical_ids` créés | 11 | **15** |
| `persons_created` (identités logiques) | 9 | 9 |
| `technical_id_changes` | 2 | **6** |
| `descriptor_rejections` | 37 | 42 |
| `TRACK_ID_APPEARANCE_DISCONTINUITY` | 0 | 0 |
| `IDENTITY_SWAP_CORRECTED` | 0 | 0 |
| `match_thresh` / `proximity_thresh` appliqués | 0,90 / 0,50 | 0,75 / 0,80 |

Distribution des similarités d'apparence **entre frames consécutives d'une même
piste** (proxy intra-identité, `continuity_similarity_summary`) :

| Statistique | Avant | Après |
|---|---:|---:|
| échantillons | 763 | 798 |
| min | 0,619 | 0,722 |
| p1 | 0,896 | 0,828 |
| p5 | 0,951 | 0,913 |
| médiane | 0,993 | 0,987 |
| sous le seuil (0,50) | 0 | 0 |

**Lecture — et c'est la conclusion importante de ce document.** Sur le clip
mesuré, le lot A **n'améliore pas** la stabilité d'identité : il crée **plus**
d'identifiants techniques (11 → 15) et **plus** de changements d'identifiant
technique (2 → 6). Le nombre d'identités **logiques** reste identique (9), donc
la galerie ReID absorbe la sur-création, mais au prix d'un brassage accru
d'identifiants techniques.

Deux effets agissent probablement en sens inverse et ne peuvent pas être
séparés par cette mesure :

- `match_thresh: 0.9 → 0.75` **resserre** l'appariement IoU : sous 0,75, une
  piste en mouvement peut être perdue plus tôt et repasser par la création d'un
  nouvel identifiant ;
- `new_track_thresh: 0.7 → 0.45` **abaisse** le seuil de création : là où une
  détection faible était auparavant ignorée, une piste est désormais créée.

Le second effet augmente mécaniquement `technical_ids` ; le premier augmente
`technical_id_changes`. Le prompt justifiait ces deux changements par des
symptômes (personne jamais ré-attribuée, appariement croisé) qui **n'ont pas été
reproduits** sur les clips disponibles : aucune `TRACK_ID_APPEARANCE_DISCONTINUITY`
dans les deux runs, donc **aucun swap observé** à corriger.

**Recommandation** : les trois seuils restent **PROVISOIRE**, et le lot A **ne
doit pas être considéré comme validé**. Une régression de stabilité
d'identifiant technique est un effet défavorable pour l'audit (plus de lignes
`TECHNICAL_ID_CHANGED`, plus de candidats ReID). Avant de figer ces valeurs, il
faut une mesure sur un clip où le symptôme se produit réellement (occlusion
dense avec échange effectif) ; à défaut, revenir à `match_thresh: 0.9` /
`new_track_thresh: 0.7` est le choix conservateur, ou tester les deux paramètres
**séparément** pour connaître le responsable.

### 1.3 Réserve méthodologique

Les deux runs de §1.2 ne sont pas bit-à-bit reproductibles : le nombre de
**détections** observées diffère (778 vs 817, soit +5 %) alors que le seuil
d'inférence YOLO est identique dans les deux configurations. L'écart
d'identifiants techniques (11 → 15, +36 %) dépasse cette variabilité mais ne
peut pas être attribué avec certitude à un paramètre précis sans une mesure
répétée (au moins 3 répétitions par configuration). Aucune répétition n'a été
faite : la conclusion de §1.2 est donc « non validé, probablement défavorable »,
pas « régression démontrée ».

---

## 2. Seuils et valeurs modifiés

| Paramètre | Avant | Après | Effet mesuré | Compromis accepté |
|---|---|---|---|---|
| `tracker.match_thresh` | 0,90 | 0,75 | `technical_id_changes` 2 → 6 | appariement plus strict : sous 0,75 une piste rapide peut être perdue et basculer sur la réassociation galerie |
| `tracker.proximity_thresh` | 0,50 | 0,80 | apparence native consultée presque toujours ; `descriptor_rejections` 37 → 42 | coût de calcul du descripteur natif, risque accru sur apparences très similaires |
| `tracker.new_track_thresh` | 0,70 | 0,45 | `technical_ids` 11 → 15 | davantage de pistes créées sur détections faibles ; garde-fou faux positifs reporté sur le filtre géométrique et `timing.confirmation_seconds` — **report documenté, non mesuré** |
| `reid.long_term.gallery_retention_seconds` | `null` (= grâce 5 s, fenêtre 10 s) | `12.0` (fenêtre 17 s) | non mesuré (aucune occlusion > 10 s dans les clips disponibles) | mémoire d'apparence conservée plus longtemps : plus de candidats comparés, coût et risque de fausse réassociation encadrés par `safety_margin` |
| `presence.head_assist.keypoints_used` | 3 points | 5 points (+ oreilles) | non mesuré (assistance désactivée) | aucun : correction d'une incohérence rapport ↔ config |
| `presence.head_assist.max_head_staleness_frames` | (absence) | `1` | couvert par tests unitaires seulement | un point tête périmé ne maintient plus la présence |
| `reid.long_term.feature_momentum` | 0,85 **codé en dur** | 0,85 **configurable** (`0 ≤ x < 1`) | identique (même valeur) | aucun : simple exposition d'une constante |
| `reid.appearance_continuity.freeze_gallery_frames` | (absence) | `3` | couvert par tests unitaires ; 0 discontinuité réelle observée sur les clips | une discontinuité prolongée (faux positif) gèle la mise à jour du descripteur |
| `reid.long_term.descriptor.*` (HSV par bandes) | 1 histogramme global 24×8 sur le crop complet | 3 bandes pondérées `[1, 1, 0.5]`, érosion 15 %, normalisation par bande | p1 des similarités 0,896 → 0,828 (distribution plus étalée) | dimension ×3 du descripteur (576 valeurs), léger surcoût de calcul |
| `geometry.multi_person_box.head_count_enabled` / `head_separation_ratio` | (absence) | `true` / `0.4` | non mesuré (assistance désactivée) | signal actif seulement si le modèle pose tourne |

`similarity_threshold: 0.40` **n'est pas modifié** : cf. §3.

---

## 3. Histogrammes de similarité et seuil proposé (B.4)

Le prompt demande deux distributions — **intra-identité** et **inter-identités** —
sur une session réelle, afin de justifier une nouvelle valeur de
`reid.long_term.similarity_threshold`.

- **Distribution intra-identité** (mesurée, cf. §1.2) : available sous la forme
  des similarités d'apparence **entre frames consécutives d'une même piste**
  (proxy). p1 = 0,83 (après) / 0,90 (avant), médiane ≈ 0,99, aucune valeur sous
  0,50.
- **Distribution inter-identités** : **non produite.** Elle exige une vérité
  terrain de type MOT (deux personnes distinctes annotées sur les mêmes frames),
  qui n'existe pas dans le dépôt (`config/protocol.yaml` : ensembles
  `datasets.*` vides). Un histogramme inter-identités fabriqué à partir de
  paires non annotées serait un chiffre inventé.

**Conséquence assumée** : `reid.long_term.similarity_threshold` **reste à 0,40
et reste marqué PROVISOIRE**. Aucune valeur n'est proposée, car l'écart
inter-personnes — la seule grandeur qui permettrait de la choisir — n'a pas été
mesurée. Le descripteur par bandes a été livré parce qu'il est structurellement
plus discriminant (érosion du fond, normalisation par bande), mais sa
discrimination n'a pas été quantifiée.

---

## 4. Tests ajoutés et tests corrigés

### 4.1 Tests ajoutés — `tests/test_correctif_occlusion.py` (21 tests)

| Correction | Tests |
|---|---|
| A.4 base de temps | avertissement uniquement en caméra live ; message chiffré au FPS mesuré (150 frames à 3,5 FPS ≈ 42,9 s) |
| B.1 personne assise | `personne_assise_stabilisee_redevient_fiable`, paramétré `head_assist` **activé et désactivé** |
| B.2 péremption du point tête | maintien **accepté** si le point est frais (1 frame) ; **refusé** s'il est périmé (3 frames) |
| B.3.a momentum | `feature_momentum` lu dans la config et propagé au gestionnaire d'identités ; rejet de `1.0` et `-0.1` |
| B.3.b gel du descripteur | descripteur inchangé pendant toute la fenêtre de gel ; correction croisée possible **après 5 frames de swap** (impossible sans gel) |
| B.4 descripteur | paramètres lus depuis la config ; érosion horizontale qui ignore le fond ; normalisation par bande + pondération ; `_blend` gère le changement de dimension |
| B.5 multi-personnes | signal par nombre de têtes (fusion détectée, têtes trop proches ignorées, deux points fiables exigés) ; signal **inactif** quand l'assistance est désactivée ; identité absorbée non purgée en silence |
| C.2 (prérequis) | extraction d'un point tête depuis un **vrai tenseur `keypoints`** `(N, 17, 2)` / `(N, 17)`, et chaîne complète tenseur → `Detection` → `PRESENCE_MAINTAINED_BY_HEAD` |

### 4.2 Tests corrigés (attente devenue fausse — jamais supprimés, jamais `skip`)

| Test | Raison |
|---|---|
| `test_config_validation.py::test_yolo_threshold_is_low_enough_for_the_tracker_low_stage` | affirmait `new_track_thresh == 0.7` ; la valeur voulue est 0,45 (A.1) |
| `test_config_validation.py::test_gallery_retention_defaults_to_the_occupancy_grace_period` → renommé `test_gallery_retention_is_explicit_and_covers_the_longest_measured_occlusion` | affirmait `gallery_retention_seconds is None` ; A.2 la rend explicite à 12 s, et le test vérifie désormais que la fenêtre totale couvre le maximum mesuré de 13,5 s |
| `test_long_occlusion_reid.py::test_retention_is_aligned_on_the_occupancy_grace_by_default` → renommé `test_retention_is_explicit_and_covers_the_longest_measured_occlusion` | même raison ; le test vérifie en plus que la rétention est **découplée** de la grâce |
| `test_long_occlusion_reid.py::test_occlusion_beyond_retention_is_explicit_and_never_silent` | utilisait 70 frames (7 s) comme « au-delà de la rétention » ; avec 12 s de rétention la fenêtre vaut 15 s. Le nombre de frames est désormais **dérivé de la configuration** au lieu d'être un littéral |
| `test_occupancy_synthetic.py::test_reappearance_after_release_creates_a_new_identity`, `test_pipeline_integration.py::test_reappearance_too_late_is_rejected`, `test_bbox_locker_integration.py::test_the_profile_is_released_once_the_person_is_gone` | mêmes littéraux de frames (40/50/22) devenus insuffisants pour franchir `grâce + rétention` ; dérivés de la configuration |
| `test_identity_manager.py::test_descriptor_is_blended_and_renormalized` | utilisait une seconde apparence **orthogonale** à la première (similarité 0) ; depuis B.3.b c'est une discontinuité qui **gèle** le descripteur. Le test vérifie maintenant le blend sur une apparence **proche** (continuité respectée) |
| `test_head_assist_presence.py::test_head_assist_extension_expires_after_max_seconds` | provoquait l'ancrage non fiable par une **chute de hauteur stabilisée**. Depuis B.1, une chute confirmée est réadaptée (ancre de nouveau fiable), donc ce motif n'est plus permanent. Le test utilise désormais un motif réellement indisponible : une boîte au **bord de l'image** |

---

## 5. Symptômes qui subsistent

| Symptôme du prompt | État après A+B | Relève de |
|---|---|---|
| 1. Swaps d'identifiant sous occlusion dense | **non reproduit** sur les clips disponibles (0 discontinuité mesurée) : ni corrigé, ni infirmé. La correction active (`IDENTITY_SWAP_CORRECTED`) n'a jamais été déclenchée | calibration sur un clip où le swap se produit ; lot C si les seuils ne suffisent pas |
| 2. Personne jamais ré-attribuée après perte longue | traité **par construction** (rétention 12 s > maximum mesuré 13,5 s ; `new_track_thresh` abaissé) mais **non mesuré** : aucune occlusion > 10 s dans les clips disponibles | calibration |
| 3. Deux personnes dans une seule boîte / un seul identifiant | signal de largeur opérationnel (6 événements mesurés) ; le **second signal par nombre de têtes** est livré mais **inactif** faute d'assistance tête activée ; l'identité absorbée n'est plus purgée en silence (testé) | mesure avec `head_assist` activé |
| 4. Assistance tête inopérante | cause racine confirmée : `presence.head_assist.enabled: false` **et** `models/yolo11s-pose.pt` absent. Toujours **non mesurée en conditions réelles** (poids non fournis, pas de corpus) | lot C.2 : télécharger les poids, renseigner le SHA-256, mesurer le coût CPU |

**Lot C : non exécuté.** Aucune ligne du lot C n'a été écrite. L'option (a)
« tête = ancre de comptage » n'est pas tranchée : elle reste à décider **après**
une mesure réelle du lot B, comme l'exige le prompt.

---

## 6. Corrections rapport ↔ code à répercuter

1. **§4.7 annonçait 5 points-clés, la configuration en déclarait 3.**
   Corrigé dans `config/pipeline.yaml` (`keypoints_used` = 5 points, oreilles
   incluses) ; le rapport est mis à jour.
2. **§4.1 annonçait « 352 passed », §4.7 « 585 tests ».** Les deux chiffres
   étaient ceux de deux états différents du dépôt. Le rapport porte désormais
   l'état courant de la suite (`606 passed`, couverture 93,5 %) et une note
   précisant que 352 et 585 étaient des états antérieurs.
3. **§4.7 « fait et testé » couvrait la logique de décision, pas l'intégration
   du modèle pose.** Le rapport est corrigé : l'intégration réelle n'était
   pas couverte (les tests injectaient `head_point` à la main), et un test
   d'intégration sur tenseur `keypoints` réel est ajouté (§4.1), sans que cela
   constitue une validation en conditions réelles — le poids n'étant pas
   versionné, ce chemin n'a **jamais tourné** sur une vidéo.

---

## 7. Fichiers modifiés

| Fichier | Nature |
|---|---|
| `config/pipeline.yaml` | seuils lot A, rétention, points-clés, nouveaux paramètres (momentum, descripteur, gel, péremption tête, multi-personnes) |
| `src/configs/custom_botsort.yaml` | seuils BoT-SORT synchronisés, report du garde-fou faux positifs documenté |
| `src/config.py` | `DescriptorConfig`, `feature_momentum`, `freeze_gallery_frames`, `max_head_staleness_frames`, `head_count_enabled`, `head_separation_ratio` + validation |
| `src/geometry.py` | `detect_multi_person_by_head_count` |
| `src/identity_manager.py` | descripteur HSV par bandes, gel du descripteur, `appearance_suspect_until_frame` |
| `src/occupancy_manager.py` | B.1 (retrait du terme `head_assist`), B.2 (fraîcheur du point tête), B.5 (signal têtes + absorption), câblage `feature_momentum` et `swap_correction` |
| `src/occupancy_types.py` | `head_point_frame_index`, `multi_person_absorption_reported` |
| `src/events.py` | champs optionnels du signal multi-personnes (`signal`, `head_count`, `head_separation`) |
| `src/main.py` | avertissement de base de temps (A.4) |
| `tests/*` | 1 fichier ajouté, 7 tests corrigés (raisons ci-dessus) |
