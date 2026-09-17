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

Point d'entrée unique : **`src/main.py`** (lancé par `run.sh`). Tout composant
expérimental (stabilisation de boîtes, extracteur d'apparence profond) est
désactivé par défaut et ne s'active que par configuration explicite, pour les
expériences d'ablation.

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
`--no-safety-margin`, `--no-spatial-constraint`, `--use-bbox-locker`,
`--use-anchor-stabilizer`, `--external-reid`.

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
   stabilisation (`test_stabilization_flags.py`) ;
2. **trajectoires synthétiques** — compteurs et occupation attendus, scénarios
   d'hésitation, de franchissement rapide et de scène initialement peuplée
   (`test_occupancy_synthetic.py`) ;
3. **intégration sans caméra** — croisements, occultations, réapparitions,
   situation ambiguë de ReID (`test_pipeline_integration.py`), et parcours
   complets du point d'entrée avec détection simulée : ligne validée utilisée
   partout (horizontale et inclinée), annulation, `--no-show` sans écran
   (`test_main_e2e.py`) ;
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
src/bbox_height_locker.py   verrouillage de hauteur de boîte (expérimental, désactivé)
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
