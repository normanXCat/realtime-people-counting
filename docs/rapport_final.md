# Rapport de correction — état des livrables et vérifications

Ce document résume ce qui a été fait, ce qui a été vérifié, et ce qui reste à
faire. Il complète `docs/audit_livrable0.md` (état des lieux initial et décisions).

## 1. Livrables

| # | Livrable | Statut | Emplacement |
|---|---|---|---|
| 0 | Audit préalable module par module | fait | `docs/audit_livrable0.md` |
| 1 | Code consolidé autour du chemin unique | fait | `src/main.py` + 11 modules |
| 2 | Configuration unique validée | fait | `config/pipeline.yaml`, `src/config.py` |
| 3 | Gestionnaire d'identités à deux niveaux | fait et testé | `src/identity_manager.py` |
| 4 | Machine à états complète, table explicite | fait et testé | `src/fsm.py` |
| 5 | Journalisation JSON Lines incrémentale validée | fait et testé | `src/events.py` |
| 6 | Suite de tests niveaux 1–3, une commande, couverture | fait | `tests/`, `pytest.ini` |
| 7 | Rapport d'évaluation (protocole section 9) | outillé, non mesuré | `config/protocol.yaml`, `scripts/run_protocol.py`, `evaluate_system.py`, `docs/protocole_evaluation.md` |
| 8 | Politique de confidentialité | fait | `docs/politique_confidentialite.md` |
| 9 | Assistance tête en cas d'occlusion des pieds (additif) | fait et testé | `src/geometry.py`, `src/occupancy_manager.py`, `src/config.py`, `docs/handoff_head_assist.md` |

Niveau 4 (vidéos réelles annotées) : **outillé mais non exécuté**, faute de corpus
annoté et d'ensemble de test tenu à l'écart. Les métriques de la section 9.3 ne
peuvent pas être publiées sans ces vidéos ; le dispositif qui les produira est en
place et testé.

## 2. Suppressions (spec « ce qu'il faut supprimer franchement »)

Supprimés du dépôt : `src/detection.py`, `src/tracker.py`, `src/visualizer.py`,
`src/calibration.py` (doublon de la calibration de `detection.py`), `src/reid.py`,
`src/hybrid_tracker.py`, `src/configs/bytetrack_custom.yaml`,
`bbox_height_locker.py` (racine, doublon octet pour octet),
`predictions_systeme.json` (artefact de run), et les tests hérités qui ciblaient
ces modules.

> `src/calibration.py` a ensuite été **recréé** (voir §4.4) avec un rôle
différent : unique point de définition de la ligne virtuelle, obligatoirement
manuelle. Ce n'est pas un retour en arrière : le doublon supprimé déduisait la
ligne avec un repli automatique, celui-ci l'exige de l'opérateur.

Aucune référence à ByteTrack ou DeepSORT ne subsiste dans le code, la
configuration, le README ou les scripts : les seules occurrences restantes sont
dans l'audit, qui documente leur suppression.

Morts retirés du cœur métier : états `EN_ZONE_LIGNE` / `A_RETOURNE` jamais
atteints, 11 champs de piste jamais lus ni écrits, double ligne
(`line2_px`, `gate_width_px`, `is_zenithal`), arguments CLI sans effet
(`--line2-p1`, `--line2-p2`, `--gate-width`, `--zenithal`), et les dépendances
`scipy` / `scikit-learn` (plus aucun import dans le dépôt).

## 3. Composants conservés et adaptés (règles 2, 3 et 5)

| Élément | Traitement |
|---|---|
| Persistance BoT-SORT (`persist=True`, `custom_botsort.yaml`) | conservée telle quelle |
| `AnchorStabilizer`, `BBoxHeightLocker` | logique interne inchangée ; `BBoxHeightLocker` est **actif par défaut** (historique indexé par `person_id`, purge sur les personnes suivies), `AnchorStabilizer` reste désactivé par défaut ; traçabilité ajoutée |
| Extracteur d'apparence profond (ex-`reid.py`) | fusionné dans `identity_manager.py`, activé seulement par `reid.external_reid.enabled` |
| Contrainte spatio-temporelle et qualité de descripteur (ex-`hybrid_tracker.py`) | fusionnées dans le niveau 2 du gestionnaire d'identités |
| `evaluate_system.py` | corrigé sur place : lit le JSON Lines validé, raisonne en secondes, calcule F1, occupation et FPS |
| `OccupancyManager`, `occupancy_types.py` | gardent leur nom et leur rôle ; logique de transition déplacée dans `fsm.py`, géométrie dans `geometry.py`, identités dans `identity_manager.py` |
| `.gitignore` | corrigé : la suite de tests n'est plus ignorée, `results/` et `data/videos/` le sont |

## 4. Vérifications exécutées

### 4.1 Suite de tests — une seule commande

```
$ python -m pytest
352 passed in 8.72s
Required test coverage of 85% reached. Total coverage: 92.75%
$ python -m py_compile src/*.py     # compilation vérifiée
```

Couverture du cœur métier (le seuil de 85 % porte sur ces modules) :

| Module | Couverture |
|---|---:|
| `src/calibration.py` | 99 % |
| `src/geometry.py` | 98 % |
| `src/occupancy_types.py` | 97 % |
| `src/fsm.py` | 96 % |
| `src/metrics.py` | 95 % |
| `src/config.py` | 92 % |
| `src/occupancy_manager.py` | 92 % |
| `src/bbox_height_locker.py` | 90 % |
| `src/events.py` | 88 % |
| `src/identity_manager.py` | 87 % |
| **total cœur métier** | **93 %** |

`src/main.py` est exclu du seuil (`.coveragerc`) car c'est la boucle
d'entrée/sortie : ses fonctions pures sont testées unitairement (surcharges CLI,
résolution de source, empreinte des poids, préparation et écriture de la
configuration de session, paramètres transmis à la sélection de ligne) et ses
parcours complets — ligne validée, annulation, affichage indisponible, source
inouvrable — sont testés de bout en bout avec une détection simulée
(`tests/test_main_e2e.py`), plus un cas en sous-processus pour la source
inouvrable.

La sélection de ligne est couverte par `tests/test_calibration.py` (22 tests :
normalisation des clics, machine d'états de la sélection, refus d'une ligne
incomplète/dégénérée/trop courte, rendu des points, de la ligne et des côtés
IN/OUT) et par `tests/test_calibration_window.py` (10 tests) qui rejoue la boucle
d'événements avec une fenêtre OpenCV **simulée** : « C » trop tôt refusé, « G »,
« I », « Échap », fermeture de fenêtre, backend graphique en échec, image très
grande réduite. Seule une session avec un vrai serveur graphique n'est pas
automatisable.

### 4.2 Exécution réelle sur vidéo (validation de bout en bout)

`test/video3.mp4`, ligne validée à la main dans la fenêtre de sélection (deux
clics + « C »), 60 frames, YOLO11n, `imgsz=960`, CPU :

```
[BILAN] OCCUPATION_CONFIRMEE=4 INCERTAINE=0 RANGE=[4, 4] IN=0 OUT=0 NEW=1
[LATENCE P95] capture=503.5ms detection=393.3ms tracking=13.3ms render=12.3ms fsm=2.7ms reid=2.0ms
FPS moyen mesuré = 3.48 ; occupation initiale (warm-up) = 3 ; phase WARMUP -> RUNNING
```

Événements écrits : 30 (`SESSION_START`, `WARMUP_END`, 16 `OCCUPANCY_SNAPSHOT`,
6 `STATE_TRANSITION`, 4 `REID_NEW`, 1 `NEW`, `SESSION_END`), tous validés par le
schéma. Les sorties sont dans `results/<session_id>/` et aucune écriture n'a lieu
dans le répertoire courant.

### 4.3 Cas d'erreur et modes dégradés

| Cas | Comportement vérifié |
|---|---|
| Source inouvrable | `SOURCE_ERROR` journalisé, code de sortie 5, aucun comptage inventé (test en sous-processus) |
| Fin de flux | `SOURCE_END_OF_STREAM` distinct d'une erreur (spec 5.5) |
| Absence prolongée de détections | `SOURCE_NO_DETECTION` après `no_detection_seconds` |
| Ligne non validée (Échap ou fermeture de la fenêtre) | `LINE_CALIBRATION_CANCELLED` journalisé, code 4, aucun comptage (test de bout en bout) |
| Affichage indisponible | `LINE_CALIBRATION_UNAVAILABLE` journalisé, code 4 : le programme refuse de démarrer au lieu de fabriquer une ligne par défaut |
| Ligne incomplète, à points confondus ou trop courte | refus annoncé dans la fenêtre (« C » sans effet utile), jamais validée en silence |
| Config invalide / `line.p1` ou `line.p2` présents dans le YAML / poids absent ou hash incorrect | refus au démarrage avec message explicite (codes 2, 3) |
| Nom de fenêtre non ASCII | Le backend Qt d'OpenCV ne retrouve plus la fenêtre (`NULL window handler`) : le nom est refusé au chargement (code 2) et translittéré ailleurs |
| Rejouabilité | `results/<session>/config_resolved.yaml` se recharge tel quel et redonne une configuration identique (test dédié) |

### 4.4 Ligne virtuelle : sélection manuelle obligatoire

Enchaînement imposé : lancement → mode `comptage` (seul mode officiel, journalisé
dans `SESSION_START`) → affichage de la **première image** → clic du point 1 →
clic du point 2 (la ligne est tracée) → « C » confirme → **seulement alors**
YOLO11 + BoT-SORT + comptage. Aucune ligne par défaut (ni horizontale, ni à
60 %), exactement deux points, une seule ligne (pas de L1/L2).

« G » réinitialise, « I » inverse intérieur/extérieur, « Échap » annule avec un
message clair et un code 4 sans lancer le traitement. `--no-show` ne dispense pas
de la sélection : sans interface graphique, le démarrage est refusé avec un
message explicite. Les points sont normalisés entre 0 et 1, le côté intérieur est
annoncé à l'écran, et la ligne validée est la seule utilisée pour le dessin, la
distance signée, les franchissements et le comptage. Elle est archivée dans
`results/<session>/calibration.json` et journalisée (`LINE_VALIDATED`) ;
l'orientation retenue est écrite dans `config_resolved.yaml`.

Vérifications dédiées : `tests/test_calibration.py` et
`tests/test_calibration_window.py` (logique, touches, rendu, lignes horizontale /
verticale / inclinée), `tests/test_occupancy_synthetic.py` (franchissements sur
ligne inclinée et verticale), `tests/test_main_e2e.py` (la ligne tracée est celle
validée, horizontale comme inclinée, et l'ancienne ligne par défaut à 60 %
n'apparaît plus) et `tests/test_main_units.py` (mode, `--no-show`, absence de
tout résidu de double ligne).

Un défaut bloquant a été corrigé après la première exécution réelle : le titre de
la fenêtre contenait un tiret cadratin, or le backend Qt d'OpenCV 5.0 ne retrouve
pas une fenêtre au nom non ASCII — `cv2.setMouseCallback` échouait alors en
`(-27:Null pointer) NULL window handler`, et **aucun** comptage ne pouvait
démarrer. `calibration.sanitize_window_name()` garantit désormais des noms ASCII
et `display.window_name` est validé au chargement. Vérifié contre le backend Qt
réel : nom accentué → échec, nom assaini → fenêtre ouverte, callback installé,
boucle complète et sortie propre (`Échap`).

### 4.5 Protocole d'évaluation

```
$ python scripts/run_protocol.py --dataset calibration --dry-run
[PROTOCOLE] Aucune vidéo déclarée dans datasets.calibration ...
```

Le dispositif vérifie la disjonction calibration / test, génère une configuration
revalidée par variante, refuse de conclure sans corpus et sans critère chiffré.
Tests dédiés : `tests/test_evaluate_system.py`, `tests/test_protocol.py`.

### 4.6 Stabilité de piste, warm-up et zone morte (corrections ciblées)

Cinq symptômes avaient été observés sur les vidéos de test : identifiants
techniques instables sous occlusion, personnes jamais réassociées après une
longue occultation, personnes visibles au warm-up mais non comptées, warm-up
anormalement long, zone morte qui grandit et rétrécit. L'hypothèse de départ —
les symptômes d'identifiant, de comptage au warm-up et de zone morte partagent
une cause commune (une grandeur dérivée de la **hauteur de boîte brute
instantanée**), les deux autres étant distincts — a été vérifiée composant par
composant avant toute correction.

**1. Stabilité des identifiants BoT-SORT.** Les paramètres de persistance et de
ReID natif sont désormais **déclarés** dans `config/pipeline.yaml` (section
`tracker`) et non plus seulement dans le YAML lu par Ultralytics : un test
vérifie que les deux sources ne divergent pas (`tracker.config_path`,
`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`,
`with_reid`, `match_thresh`, `proximity_thresh`, `appearance_thresh`,
`gmc_method`). `track_buffer` n'est pas une valeur par défaut arbitraire : la
durée des occlusions a été **mesurée** sur les sessions réelles
(`results/*/events.jsonl`, événement `OCCLUDED` → réapparition) — médiane 2,5 s,
p90 5,1 s, maximum 13,5 s. 150 frames ≈ 5 s à la cadence source de 30 ips
couvrent donc le p90 ; au-delà, la galerie ReID long terme prend le relais.
`with_reid: true` fait tenter à BoT-SORT une réassociation par apparence **avant**
de créer un nouvel identifiant technique, en complément de `IdentityManager`
(qui opère, lui, après expiration de la piste technique). `gmc_method: none` est
imposé pour une caméra fixe : toute compensation de mouvement y serait une
source d'instabilité gratuite.

**2. Comptage pendant le warm-up.** Le compteur « frames stables en
`ZONE_INTERIEURE` » vit sur la piste **logique** (`person_id`), jamais sur
l'identifiant brut de BoT-SORT, et `IdentityManager` est actif **pendant** le
warm-up (aucune phase parallèle). Un changement d'identifiant technique réassocié
durant le warm-up ne remet donc plus le compteur à zéro : la personne, visible à
chaque frame, est comptée normalement. Un test dédié rejoue ce scénario ; un
second vérifie qu'un `person_id` réellement nouveau reste une entité distincte
(pas de fusion).

**3. Durée du warm-up.** Le code se termine bien sur les **deux** bornes
déclarées (`warmup_min_frames` **et** `warmup_seconds`), sans condition
supplémentaire : la stabilité individuelle ne détermine que l'entrée dans
`initial_occupancy`, jamais la durée. Les sessions réelles se clôturent
exactement à 15 frames observées (4,4 s à 6,1 s selon le FPS mesuré de 2,3 à
3,2), soit `max(15 frames, 0,5 s)` — conforme à la configuration. Aucune valeur
concurrente codée en dur : un test fait varier `warmup_min_frames` et
`warmup_seconds` et vérifie que la durée **et** les valeurs publiées dans
`WARMUP_END` suivent la configuration.

**4. Stabilité de la zone morte.** Elle reste une fraction de l'échelle locale
(`geometry.dead_zone_ratio`), mais l'échelle locale est désormais alimentée par
la hauteur **stabilisée** du `BBoxHeightLocker` (moyenne glissante par
`person_id`, fenêtre `anchor.height_history_window_frames`) — soit la **même**
source que celle de l'ancre, sans seconde logique de lissage. Alimentée par la
hauteur brute instantanée, la zone morte suivait le bruit de détection (une
chute de 40 % la faisait rétrécir) ; désormais elle reste quasi constante pour
une personne immobile, ne suit pas une occlusion basse, et s'adapte
progressivement — sans à-coup — quand la personne se rapproche réellement. Un
defaut de câblage a été corrigé au passage : le verrouillage est construit depuis
la configuration dans `main.py` (seuil **et** fenêtre) et non plus avec la
fenêtre codée en dur de 15 frames du module.

**5. Ré-identification après occultation longue.** La rétention de la galerie
d'apparence est désormais une valeur **explicite**
(`reid.long_term.gallery_retention_seconds: null` = alignée sur
`timing.grace_period_seconds`). La mémoire n'est donc jamais libérée avant que la
personne n'ait eu une chance réaliste de réapparaître (fenêtre totale = grâce +
rétention, soit 2 × grâce par défaut) ; la libérer plus tôt supprimerait
l'événement `PURGE` et le passage en occupation incertaine. Le seuil de
similarité reste **constant** par défaut : un assouplissement automatique serait
un risque de fausse réassociation. Une tolérance **progressive et bornée** est
toutefois disponible (`long_absence_seconds`,
`long_absence_similarity_floor`) : le seuil descend linéairement vers le plancher
déclaré au bout de la durée d'absence, jamais en dessous, et la marge de
sécurité vis-à-vis du second candidat reste appliquée telle quelle. Elle est
désactivée par défaut et testée dans les deux sens (une apparence légèrement
changée est récupérée, une personne différente reste rejetée). Au-delà de la
rétention, le comportement reste explicite : la personne comptée demeure
`incertaine` (borne haute), aucune sortie n'est inventée, et la réapparition
devient une identité distincte dont l'ambiguïté est publiée dans l'encadrement
`[observée, opérationnelle]`.

| Valeur retenue | Choix | Compromis documenté |
|---|---|---|
| `tracker.track_buffer` = 150 | couvre le p90 mesuré (5,1 s) | plus long = moins de changements d'ID, mais plus de mémoire/calcul et davantage de réassociations erronées en forte densité |
| `tracker.with_reid` = true, `appearance_thresh` = 0,25 | réassociation par apparence avant création d'un ID technique | coût de calcul du descripteur natif ; complète, sans le remplacer, le ReID long terme |
| `tracker.gmc_method` = none | caméra fixe | aucune estimation de mouvement à stabiliser |
| `warmup_min_frames` = 15 / `warmup_seconds` = 0,5 | deux bornes exigées | à faible FPS la borne de frames domine (≈ 5 s réelles) ; c'est le prix de la robustesse aux FPS élevés |
| `geometry.dead_zone_ratio` = 0,15 sur hauteur **stabilisée** | zone morte quasi constante | adaptation plus lente à un vrai changement de distance (bornée par la fenêtre de l'ancre) |
| `gallery_retention_seconds` = null (≈ grâce) | mémoire jamais libérée avant la réapparition possible | galerie conservée plus longtemps |
| `long_absence_similarity_floor` = null | aucun assouplissement subi | l'option reste disponible et bornée, testée, pour les longues occultations |

Tests ajoutés ou étendus : `tests/test_dead_zone_stability.py`,
`tests/test_long_occlusion_reid.py`, `tests/test_warmup.py`,
`tests/test_config_validation.py` (synchronisation BoT-SORT, validation des
nouveaux paramètres), `tests/test_bbox_locker_integration.py` (câblage du
verrouillage depuis la configuration), `tests/test_stabilization_flags.py`.

### 4.7 Assistance par détection de tête en cas d'occlusion des pieds (additif)

Dans les scènes d'amphithéâtre et de salle de cours, les pieds des personnes sont fréquemment occultés par les tables et les chaises alors que le haut du corps et la tête restent parfaitement visibles.

1. **Architecture et séparation stricte des responsabilités** :
   - Un modèle d'estimation de pose (`yolo11s-pose.pt`, 17 points-clés COCO) remplace le modèle de détection standard lorsque `presence.head_assist.enabled: true`.
   - Seuls 5 points-clés de la tête sont considérés : `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear` (indices COCO 0 à 4).
   - **Garde-fou fondamental** : le point tête ne participe **jamais** au calcul de distance signée, à la détection de franchissement de ligne (`line.crossing`) ni aux décisions `IN` / `OUT` / `NEW`. Le comptage des flux reste à 100 % adossé à l'ancre pieds.
   - Le point tête intervient uniquement comme signal de maintien de présence pour empêcher un basculement prématuré vers l'état `OCCULTEE` lorsque les pieds sont occultés.

2. **Garde-fou temporel et traçabilité** :
   - Paramètre `max_presence_extension_seconds` (défaut 10,0 s) : si l'ancre pieds reste indisponible au-delà de cette durée continue, l'extension expire, l'événement `PRESENCE_EXTENSION_EXPIRED` est émis, et la piste bascule vers `OCCULTEE` (fenêtre de grâce normale puis purge).
   - Événement `PRESENCE_MAINTAINED_BY_HEAD` émis à chaque maintien avec la confiance tête et le statut d'indisponibilité des pieds (`edge`, `height_drop`, `no_detection`).

3. **Garantie de non-régression** :
   - Le flag `presence.head_assist.enabled` est `false` par défaut.
   - Tant qu'il est désactivé, aucun calcul de pose ni extraction de points-clés n'a lieu, et les 564 tests initiaux passent sans modification.
   - 21 nouveaux tests dédiés ont été ajoutés (`tests/test_extract_head_point.py`, `tests/test_head_assist_disabled_by_default.py`, `tests/test_head_assist_presence.py`, et compléments dans `tests/test_config_validation.py`), portant la suite à 585 tests réussis.

## 5. Limites et reste à faire

1. **Seuils non calibrés** : `model.confidence`, `model.nms_iou`,
   `model.image_size`, `geometry.dead_zone_ratio`,
   `timing.confirmation_seconds`, `timing.grace_period_seconds`,
   `reid.long_term.similarity_threshold`, `safety_margin`, `v_max_ratio`,
   `spatial_margin_ratio` restent marqués « PROVISOIRE » et doivent être figés
   sur l'ensemble de calibration. Il en va de même pour `tracker.track_buffer`,
   `tracker.appearance_thresh`, `tracker.proximity_thresh` et
   `tracker.match_thresh` : les valeurs retenues sont justifiées par les
   occlusions mesurées, mais la calibration finale (F1 / IDF1) doit les
   reconfirmer. `long_absence_similarity_floor` reste `null` tant qu'aucune
   mesure ne prouve qu'un assouplissement est nécessaire.
2. **Corpus absent** : les ensembles `datasets.*` de `config/protocol.yaml` sont
   vides. Sans vidéos annotées, les métriques de la section 9.3 (F1, MAE
   d'occupation, FPS) ne peuvent pas être produites.
3. **Vérité terrain MOT dense** : à annoter sur 2–3 clips courts puis évaluer
   avec `py-motmetrics` (IDF1, ID switches) ; hors du périmètre de
   `evaluate_system.py`.
4. **Matériel de référence** : `torch 2.13.0+cpu` — le seuil de FPS acceptable
   (`acceptance.min_fps`) reste `null` et devra être fixé sur le matériel
   documenté du mémoire.
5. **Sélection de ligne obligatoire** : chaque exécution exige un écran et deux
   clics validés, y compris pour le protocole d'évaluation (les sessions sont
   donc lancées par un opérateur, une à la fois). C'est la conséquence assumée de
   l'exigence « aucune ligne par défaut » ; la ligne validée est archivée
   (`calibration.json`, `LINE_VALIDATED`) pour que chaque run reste auditable.
6. **Latence de capture en headless** : la boucle `stream=True` d'Ultralytics
   mêle capture et inférence, la latence de capture mesurée inclut donc
   l'attente de l'inférence précédente. Une mesure séparée exigerait de sortir du
   mode `stream` ; le point est signalé plutôt que masqué.
