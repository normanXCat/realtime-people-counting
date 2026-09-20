# CLAUDE.md — Correctif occlusion & assistance tête (comptage de personnes)

Mémoire de travail du projet. **À lire intégralement avant toute modification
de code.** Ce fichier contient le périmètre, les règles, le protocole de mesure
et l'état d'avancement. Le détail des lots est dans `docs/`, à ouvrir seulement
au moment de traiter le lot concerné.

**Ordre de lecture en début de session :**

1. Ce fichier, en entier.
2. `docs/correctif_occlusion.md` — livrables déjà rendus, source de vérité sur
   ce qui est fait. En cas de contradiction avec le §0 ci-dessous, c'est
   `docs/correctif_occlusion.md` qui fait foi ; signaler l'écart.
3. `docs/diagnostic.md` — causes racines déjà identifiées (ne pas refaire ce
   diagnostic, ne pas le réécrire).
4. `docs/lot_1_*.md` … `docs/lot_7_*.md` — **uniquement le lot en cours.**

---

## 0. État d'avancement

> À mettre à jour à la fin de chaque lot, dans le même commit que le livrable.

| Étape | Fichier | Statut | Commit | Date |
|---|---|---|---|---|
| Catalogue du corpus (§2) | `docs/correctif_occlusion.md` §11.4 | mesuré, en attente d'acceptation | non commité | 2026-09-20 |
| Contrôle de déterminisme (§2.1) | `docs/correctif_occlusion.md` §11.7 | mesuré, en attente d'acceptation | non commité | 2026-09-20 |
| Vérité terrain (§3) | `tests/fixtures/ground_truth_fort_occ4.json` | **fournie** — 35 s annotées, 13 étiquettes, 4 segments d'occlusion (`docs/correctif_occlusion.md` §12). Métriques dérivées : **à outiller** (§12.3) | non commité | 2026-09-20 |
| Critères d'acceptation validés (§4) | — | à faire (§11.10.2) ; **baseline FPS figée à 3,825 ips au lot 0** (§13.4) | non commité | 2026-09-20 |
| Lot 0 — outillage de mesure | `docs/lot_0_outillage.md` | **accepté** le 2026-09-21 (`docs/correctif_occlusion.md` §13). Critère §0.3 non satisfait mais divergence expliquée et chiffrée (§13.5) ; ajout `line_origin` dans `src/events.py` **conservé** ; publication de la ligne dans `config_resolved.yaml` **reste ouverte** | `435804c` | 2026-09-20 |
| Lot 1 — base de temps + NMS | `docs/lot_1_base_temps_nms.md` | à faire ; §1.3 (filtre géométrique) **ramené à une investigation conditionnelle** après le lot 0 | — | — |
| Lot 4 — seuils BoT-SORT (**avancé**) | `docs/lot_4_seuils_botsort.md` | à faire | — | — |
| Lot 2 — rétention hors zone (+ bord du cadre) | `docs/lot_2_retention.md` | à faire | — | — |
| Lot 3 — ReID OSNet + galerie | `docs/lot_3_reid_osnet.md` | à faire | — | — |
| Lot 5 — retouches locales | `docs/lot_5_retouches.md` | à faire | — | — |
| Lot 6 — head_assist | `docs/lot_6_head_assist.md` | à faire | — | — |
| Lot 7 — prétraitement | `docs/lot_7_pretraitement.md` | à faire | — | — |
| Clôture — mise à jour de `docs/rapport_final.md` (§7.1) | — | à faire | — | — |

**Ordre imposé : 1 → 4 → 2 → 3 → 5 → 6 → 7.** Les numéros de lot sont
conservés (noms de fichiers, traçabilité avec `docs/correctif_occlusion.md`) ;
seul l'ordre d'exécution a changé après le lot 0. Chaque lot dépend de la
mesure du précédent dans cet ordre :

1. **Lot 1.** La base de temps corrigée conditionne l'interprétation de toute
   durée d'occlusion ; le seuil NMS agit en amont de tout le reste (une
   détection supprimée n'est récupérable par aucun réglage aval). Le §1.3
   n'est plus un correctif prescrit : le lot 0 a mesuré 52
   `POSSIBLE_MULTI_PERSON_BOX` sur `fort_occ4` avec le pipeline réel, ce qui
   invalide sa prémisse (`docs/correctif_occlusion.md` §13.5).
2. **Lot 4, avancé.** Le lot 0 localise le décrochage au niveau des **pistes**
   (s28 de `fort_occ4` : 2 pistes pour 10 personnes visibles), pas de la
   galerie : domaine du tracker. Trois valeurs de YAML, meilleur rapport
   effet/temps du plan, à épuiser avant le coût d'OSNet. Réserve conservée :
   `proximity_thresh` gouverne quand l'apparence est consultée ; avec le
   descripteur HSV, son effet peut être nul ou changer après le lot 3. Les
   seuils sont mesurés **isolément** pour rendre ce risque visible, et
   `proximity_thresh` sera à reprendre après le lot 3 s'il ne déplace rien.
3. **Lot 2.** La règle de rétention suppose la survie de piste stabilisée par
   les lots 1 et 4. Elle couvre aussi le **bord du cadre**, où la rétention
   est illimitée : `P9` sort du champ à s25 de `fort_occ4` mais reste dans la
   salle.
4. **Lot 3.** Le descripteur profond change l'échelle de
   `similarity_threshold` : mesuré seul, en deux étapes (3.a modèle, 3.b
   structure de galerie).
5. à 7. Inchangés.

Statuts possibles : `à faire` / `en cours` / `mesuré, en attente d'acceptation`
/ `accepté`. Un lot ne démarre que si le précédent **dans l'ordre ci-dessus**
est `accepté`.

---

## 1. Mission

Pipeline de comptage de personnes en temps réel (YOLO11 + BoT-SORT +
ré-identification + FSM d'occupation), caméra fixe, salle de classe /
amphithéâtre, personnes majoritairement assises, corps souvent occulté par les
tables et les chaises. Deux symptômes persistent sur vidéo à forte occlusion :

1. **Sous occlusion dense** : inversions d'identifiants (swaps), deux personnes
   proches réunies dans une seule boîte avec un seul identifiant, personnes
   perdues trop longtemps jamais redétectées.
2. **Corps masqué (chaise, table, autre personne)** : le système doit se
   référer à la **tête** pour maintenir la présence de la personne dans le
   comptage, plutôt que de la perdre.

**Ce que le projet ne demande PAS** : la tête ne devient **jamais** la
référence géométrique du comptage IN/OUT. La ligne virtuelle et le point bas de
la boîte (pieds) restent l'unique ancre de calcul des franchissements. Le rôle
de la tête est strictement de maintenir une identité **déjà suivie** dans le
compte de personnes visibles — jamais de recalculer une position, un côté, une
zone ou un franchissement.

---

## 2. Corpus de test

**Chemins à renseigner par la personne responsable avant le lot 1 :**

- Dossier des vidéos de test : **`test/`** (7 fichiers `.mp4`, non versionnés —
  `.gitignore` exclut `test/`). Catalogue complet : `docs/correctif_occlusion.md`
  §11.4.
- **Commande de lancement du pipeline sur un fichier** (établie en session 0,
  vérifiée par `--help`) :

  ```bash
  python src/main.py --mode comptage --source test/<video>.mp4
  ```

  `--source` est obligatoire pour un fichier (`source.default` vaut `"0"`,
  l'index caméra). **Cette commande exige un opérateur** : la ligne virtuelle se
  définit par deux clics puis « C », sans aucun repli automatique — voir
  `docs/correctif_occlusion.md` §11.1 et §11.1.1. Elle ne peut donc pas servir à
  une mesure automatisée. Depuis le lot 0, `--line` lève ce blocage (§13.2.1).
- **Ligne de mesure — FIGÉE pour tout le plan (lots 1 à 7).** Celle du lot 0,
  relevée dans `results/test_line_lot0/calibration.json` et identique sur les
  trois rejouages de la baseline (`results/baseline_lot0.jsonl`) :

  ```bash
  --line 0.05,0.6,0.95,0.6
  ```

  Coordonnées normalisées, `line_origin: argument`, `inside_side: negative` ;
  sur `fort_occ4` (1280×720), segment horizontal (64, 432) – (1216, 432).
  Commande de mesure de référence :

  ```bash
  python scripts/measure_corpus.py --videos test/ --line 0.05,0.6,0.95,0.6 \
      --repeat 3 --out results/<lot>.jsonl
  ```

  **Ne jamais changer cette ligne en cours de plan** : elle détermine zones,
  côtés et franchissements, donc tous les compteurs IN/OUT et de rétention. Une
  autre ligne rend les tableaux avant/après entre lots incomparables.
- **Prérequis d'installation** : `models/yolo11n.pt` n'est pas versionné et doit
  être téléchargé avant tout lancement, sinon `verify_weights` échoue (code 3).
  Source et SHA-256 : `docs/correctif_occlusion.md` §11.3.
- **Téléchargement des poids : autorisé.** Modèle pose `yolo11s-pose.pt`
  (téléchargement automatique Ultralytics) et poids OSNet-x0.25. Dans les deux
  cas, obligation de consigner dans `docs/` **la source exacte et le SHA-256**
  du fichier obtenu, et de renseigner `pose_model_expected_sha256` /
  `model_expected_sha256` — le but de ces champs est la reproductibilité, pas
  l'interdiction du téléchargement. Si un téléchargement échoue (le model zoo
  OSNet historique est sur Google Drive et casse souvent en automatisé) :
  signaler, appliquer le repli prévu au lot 3, ne pas bloquer le lot.

**Cataloguer le corpus avant le lot 1.** Deux sessions antérieures ont mesuré
leurs correctifs sur un clip **sans swap ni fusion**, ce qui a empêché toute
validation ou invalidation réelle.

1. Lister toutes les vidéos disponibles (extensions vidéo usuelles).
2. Pour chacune, faire tourner le pipeline **en l'état actuel** et noter :
   durée, nombre de frames, nombre approximatif de personnes, niveau
   d'occlusion (nombre de `POSSIBLE_MULTI_PERSON_BOX`, de
   `TRACK_ID_APPEARANCE_DISCONTINUITY`, ou à défaut inspection visuelle).
3. **Convention de nommage du corpus** : les vidéos sont préfixées `fort_occ`
   (occlusion supposée forte) et `rare_occ` (occlusion supposée faible). Cette
   étiquette est le point de départ, **pas le résultat** :
   - la vidéo de mesure principale est celle des `fort_occ` qui totalise le
     **plus grand nombre de phénomènes réellement comptés** à l'étape 2
     (swaps + fusions + pertes > 5 s) — pas la plus longue, pas la première
     par ordre alphabétique ;
   - la vidéo de non-régression est prise parmi les `rare_occ` ;
   - si le décompte contredit l'étiquette (une `fort_occ` sans phénomène, une
     `rare_occ` qui en contient), **le signaler dans le catalogue** et
     s'appuyer sur le décompte, jamais sur le nom du fichier. Ne pas renommer
     les fichiers.
4. Classer en **occlusion dense** (swaps, fusions ou pertes de piste
   réellement observables) et **occlusion faible/nulle** (garde-fou de
   non-régression).
5. Consigner le catalogue dans `docs/correctif_occlusion.md`, en nommant
   explicitement la vidéo retenue comme mesure principale et celle retenue
   comme contrôle de non-régression. Ce choix est figé pour tous les lots : ne
   pas changer de vidéo de référence en cours de plan, sinon les tableaux
   avant/après entre lots deviennent incomparables.

**Si le corpus contient peu ou pas d'occlusion dense** : ne pas bloquer. La
vidéo la plus occultée disponible devient la référence quelle que soit sa
densité réelle. Mais :

- le catalogue consigne **le décompte réel** des phénomènes observés dans cette
  vidéo (swaps, fusions, pertes > 5 s), pas une appréciation qualitative ;
- chaque livrable porte une **mention de portée** en tête :
  « mesuré sur un corpus contenant N occurrences du phénomène visé » ;
- un correctif qui ne déplace aucun compteur sur un corpus où N est petit est
  rapporté comme **non concluant**, jamais comme validé.

C'est l'erreur exacte des deux tentatives précédentes : mesurer contre un
phénomène absent et en conclure que le correctif fonctionne.

### 2.1 Contrôle de déterminisme (avant toute mesure avant/après)

Rejouer la baseline **deux fois** sur la vidéo à occlusion dense, sans rien
changer, et consigner l'écart entre les deux runs sur chaque compteur suivi.

- Si l'écart run-à-run dépasse la différence avant/après d'un correctif, cette
  différence ne prouve rien : l'indiquer dans le livrable au lieu de conclure.
- Si l'écart est non nul, chercher la source (seeds, threads, FPS variable) et
  figer ce qui peut l'être ; sinon, consigner l'amplitude du bruit comme seuil
  de signification pour tous les lots suivants.

---

## 3. Vérité terrain (bloquant — à établir avant le lot 1)

**Sans cette section, aucun lot n'est validable.** Les compteurs
`TRACK_ID_APPEARANCE_DISCONTINUITY`, `IDENTITY_SWAP_CORRECTED`,
`POSSIBLE_MULTI_PERSON_BOX` sont émis **par le pipeline lui-même** : ce sont
des proxys, pas des mesures. Baisser `match_thresh` peut faire baisser le
nombre de discontinuités *signalées* tout en augmentant les swaps *réels*. Une
mesure avant/après sur ces seuls compteurs peut donc être verte pendant que le
système régresse.

**Annotation minimale demandée** — un segment de 30 à 60 s de la vidéo à
occlusion dense, choisi parce qu'il contient effectivement au moins un swap,
une fusion ou une perte longue :

- fichier `tests/fixtures/ground_truth_<video>.json` (ou CSV), versionné ;
- pour chaque seconde (pas nécessairement chaque frame) : nombre de personnes
  réellement présentes dans la zone comptée, et l'identité stable de chacune
  (étiquettes humaines `P1`, `P2`, … attribuées une fois pour toutes) ;
- marquer les segments d'occlusion ≥ 3 s (utiles directement au lot 6, §7.5).

**Métriques dérivées, à rapporter dans chaque livrable de lot, à côté des
compteurs internes :**

| Métrique | Définition | Sens |
|---|---|---|
| `id_switches_reels` | changements d'identifiant technique pour une même étiquette humaine | symptôme 1 |
| `fragments_par_personne` | nb d'identifiants distincts attribués à une étiquette humaine sur le segment | pertes/redétections |
| `erreur_comptage_max_operational` | écart max, sur les secondes annotées, entre `occupancy_operational` et le nombre de personnes **présentes** | bilan officiel |
| `erreur_comptage_max_visible` | écart max entre `visible_count` et le nombre de personnes **visibles** annotées | effet visible |
| `pistes_vues` (diagnostic, sans cible) | nombre de pistes BoT-SORT observées à la seconde, contre les personnes visibles | localise le décrochage (tracker ou aval) |
| `fusions_reelles` | nb d'instants où 2 étiquettes humaines partagent un identifiant | symptôme 1 |

L'annotation est faite **une fois**, par la personne responsable ou par
inspection frame par frame assistée ; elle ne se refait pas à chaque lot. Si
elle n'est pas disponible au moment de démarrer, le dire et demander — ne pas
la remplacer par une estimation automatique produite par le pipeline testé.

---

## 4. Critères d'acceptation

> Valeurs **proposées**, à confirmer ou corriger par la personne responsable
> avant le lot 1. Tant qu'elles ne sont pas confirmées, marquer le tableau
> `NON VALIDÉ` dans chaque livrable.

Le projet est considéré comme abouti quand, sur le segment annoté de la vidéo à
occlusion dense :

| Critère | Cible proposée | Statut |
|---|---|---|
| `id_switches_reels` | ≤ 1 sur le segment (baseline à mesurer d'abord) | à valider |
| `fusions_reelles` non signalées | 0 — toute fusion réelle doit au minimum être publiée comme incertitude | à valider |
| `erreur_comptage_max_operational` : `occupancy_operational` contre les personnes **présentes** (`count`) | exact (0) à chaque seconde | à valider |
| `erreur_comptage_max_visible` : `visible_count` contre les personnes **visibles annotées** (§12.2.1 de `docs/correctif_occlusion.md`) | ≤ 1 personne | à valider |
| `occupancy_operational` | strictement exact (aucun IN/OUT fantôme) | à valider |
| Non-régression, vidéo à occlusion faible | aucun compteur dégradé au-delà du bruit mesuré en §2.1 | à valider |
| FPS moyen, modèle pose actif — plancher absolu | ≥ 3,0 ips | à valider |
| FPS moyen — contrainte relative | ≥ 80 % de la baseline, soit **≥ 3,06 ips** | à valider |

**Baseline FPS — FIGÉE (lot 0).** `3,825 ips`, médiane de **trois rejouages
complets** de `test/fort_occ4.mp4` par le pipeline réel
(`scripts/measure_corpus.py`, donc `src/main.py` en sous-processus, rendu
compris) : 3,701 / 3,826 / 3,825 ips. Machine de mesure : CPU seul
(`torch 2.13.0+cpu`, `cuda_available = False`), `imgsz: 960`, modèle pose
**inactif**. Détail et protocole : `docs/correctif_occlusion.md` §13.4.

Cette valeur est la référence de la contrainte relative du §4 : le plancher
relatif vaut **3,06 ips**, il est donc **plus contraignant** que le plancher
absolu de 3,0 ips. C'est lui qu'il faut opposer aux lots.

**Attention pour le lot 6** : la marge est de 0,8 ips seulement, et
`yolo11s-pose.pt` est sensiblement plus lourd que `yolo11n.pt`. Le critère FPS
a de fortes chances d'être violé dès l'activation du modèle pose — à trancher
avant le lot, pas pendant.

**Origine du plancher de 3,0 ips.** Il se dérive de la contrainte de
franchissement, il ne se choisit pas. Une personne marche à ≈ 1,3 m/s ; si la
zone de franchissement fait ≈ 1,5 m de profondeur au sol, elle y séjourne
≈ 1,15 s ; la FSM a besoin d'au moins 3 observations dans la zone pour
confirmer un franchissement, d'où FPS_min ≈ 3 / 1,15 ≈ 2,6 ips, arrondi à 3,0
avec marge. **À recalculer en session 0** avec la profondeur réelle de votre
zone, et à confronter à `timing.confirmation_seconds` : si celui-ci exige plus
de 1,15 s de présence, c'est lui qui devient la contrainte dominante, pas le
nombre de frames.

**Un lot qui améliore les compteurs internes sans progresser vers ces critères
n'est pas un succès** : le dire explicitement dans le livrable.

---

## 5. Règles absolues

- **Ne pas refactoriser.** Aucun déplacement de code, renommage ou
  réorganisation au-delà de ce que le lot en cours demande explicitement.
- **Un lot à la fois, mesuré avant le suivant.** Un commit par lot, message
  explicite. Jamais deux lots dans le même commit.
- **La suite de tests reste verte après chaque lot** (`python -m pytest`,
  couverture 85 % sur le cœur métier). Si un test échoue parce que son attente
  était erronée à la lumière du correctif : corriger le test **et expliquer
  pourquoi** dans le livrable — jamais le supprimer ni le passer en `skip`.
- **Aucune valeur métier codée en dur.** Tout nouveau paramètre va dans
  `config/pipeline.yaml`, validé par une dataclass dans `src/config.py`, publié
  dans `config_resolved.yaml`.
- **Tout changement de seuil est commenté dans le YAML** : valeur précédente,
  raison, compromis accepté. Garder le marquage `PROVISOIRE` tant que la valeur
  n'a pas été calibrée sur un corpus disjoint de l'ensemble de test.
- **Mesurer avant de conclure — sans rester bloqué.** « Ne conclus rien que la
  mesure ne soutient pas » ne veut pas dire « n'essaie rien sans corpus
  parfait ». Une valeur documentée `PROVISOIRE`, réversible, justifiée, peut
  être appliquée pour débloquer une mesure ultérieure. Une session précédente
  est restée bloquée là-dessus : des correctifs n'ont jamais été appliqués
  faute de corpus « suffisant », alors que la vidéo de test ne contenait même
  pas le phénomène à mesurer.
- **Dérive de périmètre.** Si à un moment vous vous apprêtez à faire de la tête
  une ancre géométrique (distance, côté, zone, franchissement) : refuser et
  relire le §1.
- **Ne rien inventer sur l'état du dépôt.** Citer les comptes réels
  (`grep -rc "^def test_" tests/`), jamais un chiffre mémorisé ou repris d'un
  rapport antérieur.

---

## 6. Architecture (carte du dépôt)

| Fichier | Rôle |
|---|---|
| `src/main.py` | Point d'entrée, boucle YOLO/BoT-SORT, extraction des keypoints pose |
| `src/occupancy_manager.py` | FSM par personne, comptage d'occupation, présence/occlusion |
| `src/identity_manager.py` | Ré-identification à deux niveaux (BoT-SORT natif + galerie d'apparence), correction de swaps |
| `src/config.py` | Dataclasses de configuration et validation |
| `src/fsm.py` | États de franchissement de ligne (IN/OUT/zone morte) |
| `src/geometry.py` | Calcul d'ancre, distance signée à la ligne, détection de fusion de boîte |
| `src/calibration.py` | Validation de la ligne virtuelle par l'opérateur |
| `src/events.py` | Définition et journalisation des événements |
| `src/track_diagnostics.py` | Instrumentation de diagnostic |
| `src/occupancy_types.py` | Types de données partagés |
| `src/metrics.py` | Bilan de session |
| `src/bbox_height_locker.py` | Verrouillage de hauteur de référence |
| `src/anchor_stabilizer.py` | Lissage de l'ancre |
| `src/configs/custom_botsort.yaml` | Configuration du tracker BoT-SORT |
| `config/pipeline.yaml` | Configuration principale du pipeline |
| `config/protocol.yaml` | Protocole d'évaluation |
| `tests/` | Suite de tests (compter avec `grep`, ne pas citer de mémoire) |

**Définitions à ne jamais confondre dans une mesure :**

- `occupancy_operational` = `initial + IN + NEW − OUT`. Bilan officiel (HUD,
  `SESSION_END`). Ne varie que sur une sortie confirmée par la FSM — jamais sur
  une occlusion, jamais sur une purge technique.
- `occupancy_observed` = pistes comptées (`inside_occupancy`) dont l'état
  n'est pas `ABSENTE` (`occupancy_manager.py:285-289`). **Il inclut l'état
  `OCCULTEE`** : une personne occultée pendant sa grâce y reste. Sa docstring
  affirme le contraire (« une personne comptée mais occultée n'y figure pas ») :
  elle est fausse, c'est le code qui fait foi.
- `visible_count` = pistes dans un état visible (`is_visible` :
  `INITIALISATION`, `EXTERIEUR`, `PRESENTE`, `EN_ZONE_MORTE`,
  `ENTREE_EN_COURS`, `SORTIE_EN_COURS`), **hors `OCCULTEE` et `ABSENTE`**
  (`occupancy_manager.py:312-314`, `occupancy_types.py:125-133`). Attention :
  il n'est **pas** filtré sur `inside_occupancy`, donc il compte aussi les
  pistes `EXTERIEUR` et `INITIALISATION`. Publié dans `OCCUPANCY_SNAPSHOT`
  (`visible_count`). **C'est lui, et non `occupancy_observed`, qui se compare
  aux personnes visibles annotées** (§4), et c'est lui que l'assistance tête
  doit protéger.
- `occupancy_uncertain` / `occupancy_range` = écart entre `occupancy_observed`
  et `occupancy_operational`, publié comme incertitude quand une personne est
  perdue sans confirmation de sortie.

Trois compteurs distincts, pas deux : `occupancy_operational` ≥
`occupancy_observed` ≥ personnes comptées et visibles. `visible_count` n'entre
pas dans cette chaîne : il ajoute les pistes visibles non comptées
(`EXTERIEUR`, `INITIALISATION`) et peut donc dépasser `occupancy_observed`.

---

## 7. Protocole de mesure et livrable, à chaque lot

Mesure principale sur la **vidéo à occlusion la plus dense** du catalogue.
**En plus**, rejouer la même mesure sur au moins une vidéo à occlusion faible
(un `new_track_thresh` abaissé ne doit pas faire exploser les fausses créations
de piste sur une scène calme).

Livrable : `docs/correctif_occlusion.md`, **mis à jour** (jamais réécrit), avec
pour le lot en cours :

1. Vidéos utilisées (nom, durée, frames, niveau d'occlusion), en précisant
   laquelle sert de mesure principale et laquelle de contrôle de
   non-régression.
2. Tableau avant/après des compteurs internes pertinents pour ce lot.
3. **Tableau avant/après des métriques de vérité terrain (§3)** et position par
   rapport aux critères d'acceptation (§4).
4. Pour chaque seuil modifié : valeur avant, après, effet mesuré, compromis
   accepté, statut (`PROVISOIRE` ou figé).
5. Tests ajoutés ou corrigés, avec la raison.
6. Ce qui reste non résolu, et pourquoi (donnée manquante, décision en attente,
   implémentation non faite) — jamais une affirmation de résolution que la
   mesure ne soutient pas.

Mettre à jour le §0 de ce fichier dans le même commit.

**Ne pas passer au lot suivant sans avoir rendu ce livrable et sans
confirmation explicite qu'il est accepté.**

### 7.1 Clôture — `docs/rapport_final.md`

`docs/correctif_occlusion.md` est un **journal cumulatif** (un bloc par lot,
écrit pendant le travail). `docs/rapport_final.md` est la **synthèse** destinée
à un lecteur qui n'a pas suivi : ce n'est pas une concaténation du journal.

Une fois le dernier lot accepté, mettre à jour `rapport_final.md` — **le mettre
à jour, pas le réécrire de zéro** : il contient déjà l'historique du projet.
Structure attendue de la partie ajoutée :

1. État du système avant / après, en regard des critères d'acceptation (§4),
   avec les métriques de vérité terrain — pas les compteurs internes seuls.
2. Ce qui a été changé, par sous-système (détection, tracker, ReID, présence,
   prétraitement), avec le lot d'origine de chaque changement.
3. Table de tous les paramètres de configuration modifiés : valeur avant,
   après, statut `PROVISOIRE` ou figé, et pour les `PROVISOIRE` ce qu'il
   faudrait pour les figer.
4. Provenance et SHA-256 des poids de modèles ajoutés.
5. Ce qui reste non résolu, et les décisions en réserve (§8) avec leur
   condition de réouverture.
6. Correction de tout écart rapport ↔ code relevé en chemin (voir §9).

---

## 8. Journal des décisions

Décisions prises hors du code, à ne pas rouvrir sans instruction humaine.

| Décision | Statut | Raison |
|---|---|---|
| **Détection dédiée têtes / haut du corps** (YOLO CrowdHuman) | **en réserve** | Réponse la plus directe aux deux symptômes, mais elle invalide la calibration de tous les seuils (échelle et recouvrement des boîtes changent). À rouvrir uniquement si les lots 1 à 6 n'atteignent pas les critères du §4. Si adoptée : le crop ReID devra rester corps entier (OSNet est entraîné sur corps entier), donc association tête↔corps explicite à câbler, et point bas du corps conservé comme ancre de franchissement. |
| **Prior de position (carte statique de places)** | **écarté** | Incompatible avec les cas d'usage couloir / commerce à public mobile. Si un profil « scène assise » est un jour formalisé, c'est un flag de configuration, pas une réécriture. |
| **Soustraction d'arrière-plan (salle vide)** | **écarté** | Une personne assise immobile est absorbée dans la médiane temporelle : le signal s'éteint sur le cas qu'il devait couvrir. Incompatible également avec le warm-up initial. |
| **Warm-up initial** | **conservé** | Identifie les personnes déjà présentes au lancement. Ces identités sont hors zone de franchissement, donc couvertes par la rétention illimitée du lot 2. |
| **FAISS pour l'indexation ReID** | **écarté à ce stade** | Similarité cosinus NumPy suffit à 10–50 identités (< 0,1 ms). À rouvrir au-delà de plusieurs milliers d'empreintes. |
| **Tête comme ancre géométrique** | **interdit** | Voir §1. |

---

## 9. Hors périmètre

- **Ancre géométrique tête** : explicitement écartée (§1 et §8). Ne jamais
  l'implémenter sans instruction humaine explicite et documentée revenant sur
  cette décision.
- **Comptage redondant par différence de trame** et **priorité par ancienneté
  dans la correction de swap** : filets de sécurité à n'envisager que si, après
  les lots 1 à 6 mesurés, les symptômes persistent à un niveau inacceptable au
  regard du §4. Ne pas les implémenter par anticipation.
- **Écarts rapport ↔ code** : un rapport antérieur annonçait « 352 passed »
  puis « 585 tests » dans la même version. Toute divergence de ce type trouvée
  en cours de travail est corrigée aussi dans `docs/rapport_final.md`, pas
  seulement dans le nouveau rapport.
