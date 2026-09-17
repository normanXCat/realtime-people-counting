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
| `AnchorStabilizer`, `BBoxHeightLocker` | logique inchangée, désactivés par défaut, traçabilité ajoutée |
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

## 5. Limites et reste à faire

1. **Seuils non calibrés** : `model.confidence`, `model.nms_iou`,
   `model.image_size`, `geometry.dead_zone_ratio`,
   `timing.confirmation_seconds`, `timing.grace_period_seconds`,
   `reid.long_term.similarity_threshold`, `safety_margin`, `v_max_ratio`,
   `spatial_margin_ratio` restent marqués « PROVISOIRE » et doivent être figés
   sur l'ensemble de calibration.
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
