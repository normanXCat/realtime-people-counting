# Real-Time People Counting — comptage de personnes par caméra fixe

Moteur vidéo d'un mémoire de Master STIC (2026) : comptage des entrées/sorties et
estimation d'occupation sur une caméra fixe. Le dépôt ne contient **qu'un seul
pipeline officiel**, déterministe, configurable, journalisé et testé.

## Pipeline officiel

```
Caméra / fichier vidéo
   ↓
YOLO11 (modèle configurable, poids + empreinte SHA-256 vérifiés)
   ↓
BoT-SORT natif (tracker court terme, persistance + ReID intra-buffer)
   ↓
Gestionnaire d'identités à deux niveaux (person_id stable)
   ↓
Ancre pieds (bas centre de boîte) avec garde-fou bord d'image
   ↓
Ligne virtuelle obligatoirement définie par l'opérateur
   (première image affichée, deux clics, « C » pour valider)
   ↓
Machine à états avec hystérésis relative (zone morte en fraction de hauteur de boîte)
   ↓
Comptage IN/OUT + occupation encadrée (confirmée / incertaine)
   ↓
Journal d'événements JSON Lines (écriture incrémentale)
```

Point d'entrée unique : **`src/main.py`** (lancé par `run.sh`). Le **verrouillage
de hauteur de boîte** (`stabilization.use_bbox_locker`) est actif dans la
configuration livrée : sans lui, une chute brutale de hauteur (jambes masquées,
personne assise, flou) fait remonter l'ancre pieds et produit une fausse
transition de zone. Cette hauteur **stabilisée** (indexée par `person_id`, fenêtre
`anchor.height_history_window_frames`) est aussi la source de l'échelle locale,
donc de la **zone morte** : celle-ci ne suit plus le bruit de détection frame par
frame. Les composants encore expérimentaux (stabilisation d'ancre, extracteur
d'apparence profond) restent désactivés par défaut et ne s'activent que par
configuration explicite, pour les expériences d'ablation.

### Stabilité de piste, warm-up et zone morte

Réglages issus des cinq symptômes observés sur les vidéos de test (détail et
compromis dans `docs/rapport_final.md` § 4.6) :

- **`tracker.track_buffer`** (150 frames ≈ 5 s à 30 ips) couvre le p90 des
  occlusions **mesurées** (médiane 2,5 s, p90 5,1 s) ; au-delà, la galerie ReID
  long terme prend le relais. `with_reid`, `match_thresh`, `proximity_thresh`,
  `appearance_thresh` et `gmc_method` (caméra fixe : `none`) sont désormais
  déclarés dans `config/pipeline.yaml` et vérifiés contre le YAML lu par
  Ultralytics ; `track_buffer` et les seuils d'apparence sont publiés dans
  `SESSION_START`.
- **Warm-up** : il se termine sur les deux bornes déclarées
  (`timing.warmup_min_frames` **et** `timing.warmup_seconds`), sans autre
  condition ; le compteur de stabilité est indexé par `person_id`, donc un
  changement d'identifiant technique réassocié ne le remet pas à zéro.
- **Réidentification après occultation longue** :
  `reid.long_term.gallery_retention_seconds: 12.0` (était `null`) **découple** la
  rétention de la galerie d'apparence de la grâce d'occupation : la fenêtre totale
  passe de 10 s à 17 s, ce qui couvre l'occlusion maximale mesurée (13,5 s) que
  l'ancienne valeur alignée laissait sortir de mémoire par construction.
  Une tolérance progressive du seuil, bornée, reste disponible mais désactivée par
  défaut. Le descripteur d'apparence est un histogramme HSV **par bandes** sur crop
  érodé (`reid.long_term.descriptor`), et le gel du descripteur pendant une
  discontinuité d'apparence (`reid.appearance_continuity.freeze_gallery_frames`)
  empêche la référence de dériver vers l'autre personne après un swap.
- **Assistance par détection de tête (additif, désactivé par défaut)** :
  `presence.head_assist.enabled: false`. Conçu pour les amphithéâtres où les pieds
  sont masqués par les tables : utilise `yolo11s-pose.pt` (17 points-clés COCO) pour
  maintenir la présence d'une personne sans basculer à tort en `OCCULTEE`.
  Le point tête ne participe **jamais** au calcul de franchissement de ligne ni aux
  décisions IN/OUT. `keypoints_used` retient les 5 points-clés de tête (oreilles
  incluses, elles seules survivent à une vue de profil) et
  `max_head_staleness_frames` exige un point tête **frais** (frame courante ou
  précédente) : les directions « poids du modèle pose », « SHA-256 du poids » et le
  détail des correctifs occlusion sont dans `docs/correctif_occlusion.md`.

> **`models/yolo11s-pose.pt` n'est pas versionné** (poids binaire). Il doit être
> téléchargé depuis les releases officielles Ultralytics
> (`https://github.com/ultralytics/assets/releases`), placé dans `models/`, et son
> SHA-256 renseigné dans `presence.head_assist.pose_model_expected_sha256`. Sans ce
> fichier, activer `presence.head_assist.enabled` fait échouer `verify_weights`
> (`FileNotFoundError`, `main` retourne 3).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # + requirements-dev.txt pour les tests
```

Les poids YOLO (`models/yolo11n.pt`) ne sont pas versionnés : ils sont téléchargés
au premier lancement par Ultralytics. L'empreinte attendue est déclarée dans
`config/pipeline.yaml` (`model.expected_sha256`) et vérifiée au démarrage —
remplacez-la si vous changez de poids.

## Utilisation

```bash
./run.sh                                   # caméra 0, sélection de ligne puis comptage
./run.sh --source video.mp4 --write-video  # fichier vidéo + vidéo annotée
./run.sh --source video.mp4 --inside-side positive
```

Options principales :

| Option | Rôle |
|---|---|
| `--config <yaml>` | Configuration du run (défaut : `config/pipeline.yaml`) |
| `--mode comptage` | Mode d'exécution — seul mode officiel, explicite et journalisé |
| `--source <valeur>` | Index caméra ou chemin de fichier vidéo |
| `--output-root <dossier>` | Racine des résultats (défaut : `results`) |
| `--session-id <id>` | Identifiant de session (défaut : horodatage) |
| `--inside-side positive\|negative` | Orientation **initiale** proposée dans la fenêtre de sélection (`I` inverse en direct) |
| `--no-show` | Aucune prévisualisation pendant le traitement (ne dispense **pas** de la sélection manuelle de la ligne) |
| `--write-video` | Écrire `results/<session>/annotated.mp4` |
| `--model`, `--conf`, `--iou`, `--imgsz` | Surcharges de détection |
| `--max-frames <n>` | Arrêt après N frames (diagnostic) |

Drapeaux d'ablation (protocole de la section 9) : `--no-long-term-reid`,
`--no-safety-margin`, `--no-spatial-constraint`, `--use-anchor-stabilizer`,
`--external-reid`. Le verrouillage de hauteur étant actif par défaut, il ne se
pilote plus par un drapeau mais par `stabilization.use_bbox_locker`
(`--use-bbox-locker` reste accepté, sans effet supplémentaire).

### Sélection de la ligne virtuelle (obligatoire)

Enchaînement exact, sans raccourci possible :

```
1. lancement du programme
2. mode comptage (seul mode officiel)
3. affichage de la PREMIÈRE image de la source
4. clic du point 1 -> il s'affiche
5. clic du point 2 -> il s'affiche et la ligne est tracée
6. « C » confirme la ligne
7. SEULEMENT ALORS : YOLO11 + BoT-SORT + comptage
```

La ligne séparant l'intérieur de l'extérieur n'est jamais déduite d'une valeur par
défaut (aucune ligne horizontale, aucun 60 %) : elle est définie par l'opérateur,
puis validée. Deux points exactement, pas quatre, et une seule ligne — pas de
double ligne L1/L2.

| Action | Effet |
|---|---|
| Clic gauche | Extrémité 1, puis extrémité 2 de la ligne (la ligne apparaît immédiatement) ; un troisième clic est refusé |
| `C` | Confirmer la ligne — uniquement si les deux points sont présents et la ligne valide — puis lancer YOLO11 + BoT-SORT + comptage |
| `G` | Réinitialiser la sélection et recommencer |
| `I` | Inverser le sens : quel côté de la ligne est l'intérieur (IN/OUT) |
| `Échap` | Annuler proprement le programme (message clair, aucun comptage lancé) |

- Les points sont convertis en **coordonnées normalisées** (0-1) : la ligne reste
  identique quelle que soit la résolution de la vidéo ou de la fenêtre.
- La ligne peut être **horizontale, verticale ou inclinée** ; c'est celle qui est
  tracée à l'écran qui est utilisée par `OccupancyManager`.
- Le côté intérieur et le côté extérieur sont **affichés en clair** sur l'image,
  avec une flèche vers l'intérieur ; `I` inverse les deux.
- Une ligne incomplète, à points confondus ou trop courte est **refusée** avec un
  message explicite : la fenêtre reste ouverte, rien n'est validé en silence.
- Cette ligne validée est la **seule** utilisée pour le dessin, la distance
  signée, la détection des franchissements et le comptage.
- Sans affichage disponible, le programme refuse de démarrer (code 4) avec un
  message expliquant que la calibration manuelle est impossible sans interface
  graphique : il n'existe aucun moyen de fabriquer une ligne automatiquement.
  `--no-show` ne change rien à cette règle : il ne supprime que la
  prévisualisation pendant le traitement.

Codes de sortie : `2` configuration invalide, `3` poids ou tracker introuvables,
`4` ligne non validée (annulation, affichage indisponible), `5` source inouvrable
ou erreur de lecture.

> **Noms de fenêtres OpenCV : ASCII uniquement.** Le backend Qt d'OpenCV ne
> retrouve pas une fenêtre dont le nom contient un caractère non ASCII :
> `cv2.setMouseCallback` échoue alors en `(-27:Null pointer) NULL window handler`
> et la sélection de ligne devient impossible. `display.window_name` est donc
> **refusé au chargement** s'il contient un accent ou un tiret typographique, et
> `calibration.sanitize_window_name()` translittère tout nom fourni par ailleurs.
> Un titre purement décoratif ne doit jamais pouvoir bloquer le comptage.

### Sorties d'une session

```
results/<session_id>/
├── events.jsonl            # journal JSON Lines (ligne validée, franchissements, occupation, ReID…)
├── summary.json            # compteurs, FPS mesuré, latences par étape
├── environment.json        # versions exactes des dépendances
├── calibration.json        # ligne validée : points normalisés, clics bruts, côté intérieur, résolution
└── config_resolved.yaml    # configuration effective (dont l'orientation validée)
```

Aucune écriture dans le répertoire courant. `results/` est exclu du dépôt : les
vidéos annotées contiennent des personnes (voir `docs/politique_confidentialite.md`).

## Configuration

Tout paramètre métier vit dans `config/pipeline.yaml` : aucun seuil, aucune durée
et aucune marge géométrique n'est codé en dur.

- les **durées sont en secondes** et converties en frames avec le FPS *mesuré* ;
- les **marges sont relatives** à la hauteur médiane des boîtes observées, jamais
  en pixels absolus ;
- les valeurs marquées « PROVISOIRE » ne sont pas encore calibrées : elles
  doivent être fixées sur l'ensemble de calibration du protocole d'évaluation.

Points de configuration à connaître :

- **`model.confidence` (0,10) et `tracker.track_low_thresh` (0,1)** : le seuil
  d'inférence YOLO s'applique *avant* BoT-SORT, il doit donc rester au plus égal
  au seuil bas du tracker, sinon la gamme
  `[track_low_thresh, model.confidence[` n'atteint jamais l'association et les
  personnes floues sont perdues. Une incohérence est journalisée en
  `CONFIG_WARNING` (elle n'empêche pas un run d'ablation) ;
- **`timing.warmup_seconds` + `timing.warmup_min_frames`** : le warm-up exige les
  **deux** bornes (durée écoulée *et* frames observées). La dernière frame de
  warm-up est celle qui atteint les deux ; la suivante est la première frame
  comptée, et `WARMUP_END` publie `warmup_frames_observed`,
  `warmup_elapsed_seconds` et l'`initial_occupancy` réellement retenu ;
- **`reid.long_term.gallery_*`** : qualité minimale d'une apparence avant
  écriture en galerie (confiance, taille de crop, netteté). Une apparence
  refusée n'écrase jamais une apparence connue et le refus est journalisé
  (`REID_DESCRIPTOR_REJECTED`) ;
- **`geometry.occlusion_ambiguity_ratio`** : rayon (fraction de la hauteur
  médiane) dans lequel une apparition intérieure peut être confondue avec une
  personne déjà comptée dont l'observation est perdue. Dans ce rayon, `NEW` est
  différé et signalé, jamais deviné ;
- **`diagnostics`** : journalisation des pertes d'association
  (`DETECTION_ABSENT`, `TRACK_ID_ABSENT`, `DETECTION_LOW_CONFIDENCE`) avec
  limitation de débit — le front montant est toujours écrit, les compteurs
  complets restent dans `summary.json`.

Occupation publiée — trois valeurs, un invariant :

- `occupancy_confirmed` **est** l'effectif opérationnel, c'est-à-dire le bilan
  `initial_occupancy + IN + NEW - OUT`. Il ne diminue **que** sur une sortie
  confirmée par la machine à états (ou une réapparition cohérente côté
  extérieur, qui vaut confirmation de sortie). Ni une occultation, ni
  l'expiration de la période de grâce, ni une absence de détection (`boxes`
  absentes ou sans `track_id`), ni une purge technique, ni un changement
  d'identifiant technique, ni la libération de la mémoire de galerie ne le font
  baisser ;
- `occupancy_observed` est la part de cet effectif actuellement observable ;
- `occupancy_uncertain` est la part dont l'observation est perdue. Elle n'est
  jamais retirée de l'occupation : une personne occultée reste comptée jusqu'à
  une sortie réellement observée ;
- l'invariant `observed + uncertain == confirmed` est vérifié à chaque
  snapshot publié, et `occupancy_range` vaut `[occupancy_observed,
  occupancy_confirmed]`. Une divergence est journalisée
  (`INCONSISTENT_STATE`, kind `occupancy_ledger_mismatch`), jamais corrigée par
  un `max(0, …)` : un test vérifie l'absence de tout clamp silencieux.

`PURGE` est toujours une purge **technique** (`purge_kind: technical`,
`is_exit: false`) : elle ne produit jamais de sortie. Seul un événement `OUT`
(issu des actions FSM `COMPTE_SORTIE` / `CONFIRME_SORTIE_DIFFEREE`) décrémente
l'occupation — un test d'audit verrouille le fait que `total_out` n'est
incrémenté nulle part ailleurs.

## Tests

```bash
python -m pytest tests -q                # suite complète (niveaux 1 à 3)
python -m py_compile src/*.py            # vérification de compilation
python -m pytest tests -q --cov=src      # avec couverture du cœur métier
```

Quatre niveaux de tests (section 8 du cahier des charges) :

1. **unitaires** — géométrie (`test_geometry.py`), machine à états
   (`test_fsm.py`), identités et ReID long terme (`test_identity_manager.py`),
   configuration (`test_config_validation.py`), sélection de la ligne et de son
   côté intérieur (`test_calibration.py`, `test_calibration_window.py`),
   stabilisation et son câblage (`test_stabilization_flags.py`,
   `test_bbox_locker_integration.py`) ;
2. **trajectoires synthétiques** — compteurs et occupation attendus, scénarios
   d'hésitation, de franchissement rapide et de scène initialement peuplée
   (`test_occupancy_synthetic.py`), frontière du warm-up et effectif initial
   (`test_warmup.py`), occupation opérationnelle/incertaine, purge technique et
   demi-tours (`test_occupancy_operational.py`) ;
3. **intégration sans caméra** — croisements, occultations, réapparitions,
   situation ambiguë de ReID (`test_pipeline_integration.py`), perte de piste
   par flou et réapparition avec un nouvel ID technique
   (`test_track_loss_recovery.py`), diagnostics de détection
   (`test_track_diagnostics.py`), et parcours complets du point d'entrée avec
   détection simulée : ligne validée utilisée partout (horizontale et inclinée),
   annulation, `--no-show` sans écran (`test_main_e2e.py`) ;
4. **vidéos réelles annotées** — exécutées par le protocole d'évaluation, avec
   les métriques de la section 9 (`scripts/run_protocol.py`).

## Évaluation et calibration

```bash
python scripts/run_protocol.py --dataset calibration --dry-run   # vérification
python scripts/run_protocol.py --dataset calibration             # réglages
python scripts/run_protocol.py --dataset test --variant baseline_1 --variant variant_1
python evaluate_system.py --events results/.../events.jsonl --ground-truth data/gt/clip01.json
```

Le dispositif (ensembles disjoints, critère d'acceptation chiffré, baselines et
ablations) est décrit dans `docs/protocole_evaluation.md` et déclaré dans
`config/protocol.yaml`. Les ensembles de vidéos y sont volontairement vides : les
métriques ne peuvent être produites qu'après acquisition et annotation des vidéos.

## Organisation du dépôt

```
config/pipeline.yaml        configuration unique du pipeline
config/protocol.yaml        protocole d'évaluation (ensembles, seuils, variantes)
src/main.py                 point d'entrée unique
src/calibration.py          sélection manuelle obligatoire de la ligne (clics, C/G/I/Échap)
src/config.py               chargement et validation stricte de la configuration
src/events.py               schéma versionné + journal JSON Lines incrémental
src/geometry.py             géométrie canonique (distance signée, zone morte relative, bord)
src/fsm.py                  machine à états, table de transition explicite
src/identity_manager.py     identités à deux niveaux (BoT-SORT + galerie long terme)
src/occupancy_manager.py    orchestration du comptage et de l'occupation
src/occupancy_types.py      types métier (états, zones, pistes logiques)
src/metrics.py              FPS mesuré et latences instrumentées
src/anchor_stabilizer.py    stabilisation d'ancre (expérimental, désactivé par défaut)
src/bbox_height_locker.py   verrouillage de hauteur de boîte (actif, indexé par person_id,
                            source de l'échelle locale / zone morte)
src/configs/custom_botsort.yaml   configuration BoT-SORT
scripts/run_protocol.py     exécution des baselines et ablations
evaluate_system.py          évaluation F1 / occupation / FPS d'une session
docs/audit_livrable0.md     audit du code existant et décisions
docs/protocole_evaluation.md    protocole de calibration et d'évaluation
docs/politique_confidentialite.md  données, conservation, accès
```

## Limites connues

- Les valeurs de seuils encore marquées « PROVISOIRE » ne sont pas calibrées.
- La ligne étant obligatoirement définie à la main, **chaque exécution exige un
  écran et une validation** : le protocole d'évaluation (`run_protocol.py`) doit
  donc être piloté par un opérateur, une session à la fois.
- La boucle de la fenêtre de sélection est testée avec une fenêtre OpenCV
  **simulée** (`tests/test_calibration_window.py`) ; seule une session avec un
  vrai serveur graphique n'est pas automatisable. Elle est en revanche vérifiée
  manuellement contre le backend Qt réel (ouverture, `setMouseCallback`,
  `getWindowProperty`, fermeture), y compris le refus d'un nom non ASCII.
- Le rapport final de la section 9 n'est pas encore produit : il exige les vidéos
  annotées (ensemble de calibration et ensemble de test disjoints).
- La vérité terrain MOT dense (IDF1, ID switches) est à produire et à évaluer
  séparément avec `py-motmetrics`.
