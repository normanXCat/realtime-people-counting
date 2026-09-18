# Livrable 0 — Audit préalable du code existant

**Dépôt :** `normanXCat/realtime-people-counting` (branche `main`, HEAD `5c8694a`)
**Périmètre :** moteur vidéo uniquement (source → détection → tracking → identités → ligne → comptage → occupation → évaluation).
**Méthode :** lecture intégrale de `src/*.py`, graphe d'imports statique (`grep` sur les imports), recherche des champs/états/méthodes jamais lus, exécution de la suite de tests existante.

État initial mesuré (`.venv/bin/python -m pytest tests -q`) : **7 tests passent, 4 échouent**, les 4 échecs étant des `TypeError` de constructeur (tests écrits contre des signatures qui n'existent plus). Aucune modification n'a été faite avant cet audit.

---

## 1. Graphe d'imports réel — un seul chemin actif, mais six modules orphelins

`run.sh` → `src/main.py`. Fermeture transitive des imports de `main.py` :

```
main.py → anchor_stabilizer.py
        → bbox_height_locker.py
        → occupancy_manager.py → occupancy_types.py
```

Tout le reste est **hors du chemin d'exécution** :

| Module | Lignes | Importé par le chemin actif | Importé par |
|---|---:|---|---|
| `src/detection.py` | 355 | non | personne (second point d'entrée manuel) |
| `src/tracker.py` | 143 | non | `detection.py` seulement |
| `src/visualizer.py` | 201 | non | `detection.py` seulement |
| `src/calibration.py` | 136 | non | `detection.py` seulement |
| `src/hybrid_tracker.py` | 222 | non | tests locaux seulement |
| `src/reid.py` | 276 | non | personne (mort) |
| `bbox_height_locker.py` (racine) | 73 | non | doublon **octet pour octet** de `src/bbox_height_locker.py` |

`src/detection.py` constitue un **second pipeline complet et concurrent** (YOLO + `LineCrossingTracker` + `Visualizer`), avec son propre comptage (`entries`/`exits`), sa propre calibration et sa propre politique d'ID. Il n'est jamais lancé par `run.sh` et n'est pas testé de bout en bout.

---

## 2. Audit module par module et décisions

### 2.1 `src/main.py` (589 l.) — point d'entrée réel — **CORRIGER (conservé)**

Contient déjà l'orchestration à peu près correcte (YOLO → BoT-SORT → IDManager → OccupancyManager → rendu). Problèmes relevés :

| # | Problème | Réf. spec |
|---|---|---|
| a | `IDManager` (l. 60-262) est une **troisième** implémentation de la réassociation d'apparence, en parallèle de `reid.py` (`ReIDGallery`) et de `hybrid_tracker.py` (`HybridAssociation`). Trois galeries, trois seuils (0.78 / 0.45 / 0.40), trois extracteurs. | §0.1, §3, « code dupliqué » |
| b | `ask_counting_mode()` appelle `input()` **inconditionnellement** : le mode `--no-show` n'est pas réellement headless. | §5.1 |
| c | `calibrate_two_lines()` (l. 310-400) et les arguments `--line2-p1/--line2-p2/--gate-width` existent alors que `main()` force `line2_p1 = line2_p2 = None` (l. 424) : **arguments CLI sans effet**. | « arguments CLI qui n'ont plus d'effet » |
| d | Le FPS de sortie est obtenu en **réouvrant la source** (`cv2.VideoCapture(args.source)`, l. 545) au lieu d'être mesuré. | §5.3 |
| e | L'index caméra est converti différemment selon les sites : `int(args.source) if args.source.isdigit()` pour YOLO (l. 449) mais `cv2.VideoCapture(args.source)` (chaîne) pour le writer (l. 545) et pour la calibration (l. 288). | §5.2 |
| f | Tous les paramètres métier sont codés en dur (`0.25/0.55/960`, `dead_zone=20 px`, `warmup=15 frames`, `confirm=15 frames`, `grace=300 frames`, `confirmation_confidence=0.75`, `gallery_ttl=300`). | §2, §0.2, §0.3 |
| g | Écrit `predictions_systeme.json` dans le **répertoire courant** à la fin du run. | §6.4 |
| h | Les événements ne portent pas toujours `direction` (les événements `NEW` n'en ont pas), et il n'y a aucun schéma ni `schema_version`. | §6.2 |
| i | Deux boucles de traitement distinctes (`counting_enabled` / sinon) dupliquent la logique de rendu et de comptage. | §0.1 |
| j | `verify_reid_weights()` échoue si `with_reid` est absent du YAML du tracker : couplage rigide, non configurable. | §2 |

### 2.2 `src/occupancy_manager.py` (634 l.) — cœur métier — **CORRIGER / FUSIONNER**

Ce module garde sa place et son nom : c'est le cœur du système. Défauts constatés :

| # | Problème | Réf. spec |
|---|---|---|
| a | **Distance en pixels absolus** : `dead_zone_margin` / `dead_zone_inside` (20 px / 10 px) et `door_zone = max(35.0, dead_zone_inside + 15.0)`. Aucune échelle locale. | §0.3 |
| b | **Durées en frames** : `init_duration_frames=15`, `confirmation_threshold=6`, `grace_period_frames=300`, `DEFAULT_TRAJECTORY_MAXLEN=60`, `frame_index - last_seen_frame < 3` dans `IDManager`. | §0.2 |
| c | **Machine à états implicite** : la logique de transition est dispersée dans `process_frame` (l. 245-395) sous forme de cascades `if/elif` sur `pending_direction`, `side`, `dist`. Aucune table de transition, aucun test par transition. | §4.2 |
| d | **Occupation unique et monotone** : `occupancy_count = max(0, initial + in + new - out)` (l. 179). Pas de distinction *confirmée* / *incertaine*, pas de borne haute, pas de snapshot périodique. Le `max(0, ...)` masque silencieusement une occupation négative. | §4.4, §4.5 |
| e | **Aucun garde-fou `INCONSISTENT_STATE`** : ni OUT sans IN préalable, ni double réassociation simultanée. | §4.5 |
| f | **Purge silencieuse** : `_handle_missing_tracks` supprime un `LogicalTrack` (`self.tracks.pop(lid)`) **sans émettre d'événement**. | §3.4 |
| g | **Disparition en cours de franchissement non traitée** : `_handle_missing_tracks` bascule vers `OCCULTEE` sans distinguer le cas où la piste était en cours de franchissement ; l'état `pending_direction` est conservé mais jamais résolu ni journalisé à l'expiration. | §4.3 |
| h | **États et champs morts** (vérifiés par recherche exhaustive des lectures) : `TrackState.EN_ZONE_LIGNE` n'est **jamais assigné** ; `TrackState.A_RETOURNE` n'est jamais lu (couleur identique à `PRESENTE`) ; `LogicalTrack.previous_zone`, `exterior_streak`, `interior_streak`, `exit_l1_crossed`, `entry_l2_crossed`, `previous_bottom_l1`, `previous_top_l2`, `previous_top_l1`, `previous_bottom_l2`, `crossing_in_streak`, `crossing_out_streak` ne sont **jamais ni lus ni écrits** ; `OccupancyManager.is_zenithal` est stocké mais jamais lu. | « champs d'état jamais lus ni écrits, états jamais atteints » |
| i | **Résidu double ligne** : `line2_p1`, `line2_p2`, `gate_width_px` et la méthode `line2_px()` ne sont appelés de nulle part (main.py ne les alimente plus). | « arguments CLI sans effet » |
| j | **Distances signées dupliquées** : `signed_perpendicular_distance` (ici) et `check_line_crossing` (ici) recouvrent `cross_product` / `check_intersection` de `tracker.py`, avec des conventions de signe différentes. | « plusieurs façons de calculer la même distance signée » |
| k | **Ancre non fiable au bord non gérée** : aucune notion de marge de bord ; une boîte tronquée peut franchir sur une ancre fantôme. | §5.4 |
| l | Wording métier trompeur : `origin`, `pending_direction`, `is_counted_out` mélangent comptage et état, et l'identifiant utilisé est `display_id` (produit par `IDManager`) et non un `person_id` propre au gestionnaire d'identités. | §3.1 |

### 2.3 `src/occupancy_types.py` (72 l.) — **CORRIGER**

`TrackState` ne contient pas les états demandés (`INITIALISATION`, `EXTERIEUR`, `EN_ZONE_MORTE`, `ENTREE_EN_COURS`, `SORTIE_EN_COURS`, `ABSENTE`) et contient des états morts (§2.2 h). `Zone` (INTERIEURE / MORTE / EXTERIEURE) est conservé : il est cohérent et utilisé. `LogicalTrack` doit être réduit aux champs effectivement utilisés, et étendu (position + hauteur de bbox, horodatage **en secondes**, `person_id`, statut d'occupation).

### 2.4 `src/hybrid_tracker.py` (222 l.) — **FUSIONNER puis SUPPRIMER**

Docstring et README présentent explicitement **ByteTrack comme tracker principal** et **DeepSORT** comme mode de secours, alors que BoT-SORT est le tracker actif depuis longtemps. Le module n'est branché sur aucun chemin.

Ce qui a de la valeur et sera **fusionné** dans le gestionnaire d'identités (`identity_manager.py`) :
- la contrainte **spatio-temporelle** (IoU sur boîte prédite par vélocité + `max_age`),
- la règle de **qualité de la feature** (`_feature_is_trustworthy` : vue non tronquée + confiance suffisante) avant écriture en galerie,
- le **couple (score d'apparence, seuil d'apparence)** et l'idée de filtre préalable par seuil.

Ce qui est **abandonné** : le déclenchement « à l'occlusion imminente », le coût hongrois, l'état `raw_to_stable` parallèle, `scipy.optimize.linear_sum_assignment`, les références ByteTrack/DeepSORT.

### 2.5 `src/reid.py` (276 l.) — **FUSIONNER puis SUPPRIMER**

`ReIDGallery` n'est importé par personne : **code mort**. Sa logique (galerie `id → {embedding, frame_archived}`, purge TTL, similarité cosinus, `id_remap`) est exactement le niveau 2 demandé par la spec §3.3, mais en frames et sans contrainte spatiale ni marge de sécurité.

À **fusionner** : l'extraction d'embedding profond (`MobileNetV3 Small` tronqué, L2-normalisé, `@torch.no_grad`, cadre `transforms` 128×64) — c'est le seul extracteur d'apparence réellement « personne » du dépôt, réutilisable pour l'expérience d'ablation « ReID externe » (`reid.external_reid.enabled`).

À **supprimer** : la classe `ReIDGallery` elle-même (sa galerie est remplacée par celle du gestionnaire d'identités), `occlusion_threshold`, `_prev_active_ids`, `_known_ids`, `_resolve_real_id` (boucle linéaire sur `id_remap`).

### 2.6 `src/tracker.py` (143 l.) — **SUPPRIMER**

`LineCrossingTracker` est un second moteur de franchissement (produit vectoriel sur le centre bas, `counted_in`/`counted_out` par track_id, cooldown). Il est plus permissif que la FSM (`sign_prev < 0 < sign_curr` : aucun franchissement ne peut être perdu, mais aucune zone morte, aucune gestion d'occlusion, aucun `person_id`). Le garder entretient exactement le « double chemin incohérent » visé par la spec. Ses alias `check_intersection` / `_segments_intersect` et `cross_product` sont remplacés par la géométrie canonique unique de `geometry.py`.

### 2.7 `src/detection.py` (355 l.) — **SUPPRIMER**

Second point d'entrée, non lancé par `run.sh`, non testé, qui réimplémente tout le pipeline avec d'autres seuils (`--occlusion-iou`, `--similarity-threshold`, `--gallery-ttl`, `--max-age`), sa propre politique d'ID (`conf > 0.6`) et son propre FPS (`30.0` codé en dur). Sa suppression entraîne celle de `calibration.py`, `tracker.py` et `visualizer.py` (voir ci-dessous) ; `main.py` possède déjà sa propre calibration par clics et son propre rendu.

### 2.8 `src/calibration.py` (136 l.) — **SUPPRIMER**

Doublon de `main.calibrate()` (même interaction 2 clics, même normalisation, même ligne de repli à 60 %). Importé uniquement par `detection.py`. `main.calibrate()` est conservé et devient le seul calibrateur (avec repli config).

### 2.9 `src/visualizer.py` (201 l.) — **SUPPRIMER**

Importé uniquement par `detection.py`. Son HUD titre littéralement « TRACKING (ByteTrack + ReID) » (référence ByteTrack à supprimer) et ses trois méthodes (`draw_crossing_line`, `draw_trails`, `draw_hud`, `draw_boxes`) recouvrent `OccupancyManager.draw_overlay` / `_draw_hud`. `draw_trails` n'est appelé nulle part (les « trails » sont désactivés d'après le log de `detection.py`).

### 2.10 `src/anchor_stabilizer.py` (56 l.) — **CONSERVER tel quel, passer derrière un flag**

Correct pour son rôle, sans dépendance. Devient `stabilization.use_anchor_stabilizer: false` par défaut. **Aucun changement de code**, seulement l'ajout de la traçabilité demandée par la spec §7 (boîte brute / corrigée / raison / seuil) au niveau de l'appelant.

### 2.11 `src/bbox_height_locker.py` (73 l.) — **CONSERVER, retirer le log, passer derrière un flag**

Logique OK ; deux défauts : `print()` non conditionnel dans `process_bbox` (pollution du flux headless, §5.1) et `purge_lost_tracks(active_ids, max_age=90)` dont le compteur d'âge est en frames (§0.2).

**Correction d'intégration (audit suivant).** Le composant avait d'abord été placé derrière `stabilization.use_bbox_locker: false`, ce qui le rendait inopérant dans le run officiel. Trois défauts de câblage ont été corrigés :

1. il est désormais **actif par défaut** (`config/pipeline.yaml`), avec son seuil lu depuis la configuration (`stabilization.bbox_locker_min_height_ratio`) ;
2. l'historique de hauteur est indexé par **`person_id`** (stable) et non par l'identifiant technique de BoT-SORT : un `TECHNICAL_ID_CHANGED` après occultation ne réinitialise plus la hauteur de référence (clé de repli négative `-1 - track_id` tant que l'identité n'est pas résolue) ;
3. la purge des profils suit les **personnes encore suivies** (`OccupancyManager.stabilization_keys()`), et non les identifiants présents dans la frame : une seule frame manquée ne détruit plus la correction.

En complément, la détection de chute (`check_anchor_height_drop`) est alimentée par la hauteur **brute** mesurée : alimentée par la hauteur corrigée, elle était structurellement muette et l'événement `ANCHOR_UNRELIABLE` (`reason="height_drop"`) — que la FSM consomme en priorité — ne pouvait plus être émis. Preuves : `tests/test_bbox_locker_integration.py` (dont un parcours complet écrivant l'événement dans `events.jsonl`).

### 2.12 `bbox_height_locker.py` (racine) — **SUPPRIMER**

Copie octet pour octet de `src/bbox_height_locker.py` (`diff` vide). Doublon pur : deux copies divergent au premier correctif.

### 2.13 `src/configs/bytetrack_custom.yaml` — **SUPPRIMER**

Configuration ByteTrack résiduelle, non référencée par `main.py` (`DEFAULT_TRACKER` pointe sur `custom_botsort.yaml`).

### 2.14 `predictions_systeme.json` (racine) — **SUPPRIMER**

Artefact de run (3 événements `NEW`) écrit dans le répertoire courant. Remplacé par le journal JSON Lines dans `results/{session_id}/`. Le format n'est pas celui attendu par `evaluate_system.py` (pas de champ `direction`), ce qui confirme que le contrat de sortie n'est pas stabilisé.

### 2.15 `evaluate_system.py` (132 l.) — **CORRIGER**

Attend un JSON **liste** avec `frame` + `direction` + `latency_ms`, et calcule TP/FP/FN par appariement temporel (tolérance en **frames**). Il est réutilisable mais doit lire le JSON Lines, raisonner en **secondes**, ignorer les événements non-franchissement (aujourd'hui il tenterait de les apparier) et rapporter les métriques de la §9 (F1 franchissements, erreur d'occupation, FPS).

### 2.16 `tests/` — **CORRIGER (réécrire) + réintégrer au dépôt**

`tests/` est dans `.gitignore` (`tests/` ET `*test*` ET `test_*`) : la suite de tests n'est **pas versionnée**, ce qui explique sa dérive complète. 4 des 11 tests échouent en `TypeError` contre des paramètres qui n'existent plus (`counted_grace_period_frames`, `enable_occupancy_reid`, `reconnect_max_speed_px_per_frame`, `max_reconnect_distance_px`, `new_presence_min_distance`, `id_swap_margin_ratio`, `band_margin_ratio`, `smoothing_window`, `min_cooldown_frames`).

Ces paramètres révèlent des fonctionnalités **écrites puis abandonnées** (garde spatiale de reconnexion, filtre de distance d'apparition, cooldown de franchissement, lissage). Décision : **ne pas les ressusciter** — la spec §3.3 définit une seule politique de réassociation (filtre spatio-temporel + seuil + marge de sécurité), qui couvre le besoin de la garde spatiale. Les 7 tests qui passent portent sur `HybridAssociation` et `LineCrossingTracker`, deux modules que l'on supprime : ils ne prouvent rien sur le chemin actif et seront remplacés par les tests de niveaux 1 à 3.

---

## 3. Synthèse « supprimer / garder en expérimental »

### 3.1 Suppression franche (code mort, résidu, doublon)

| Élément | Justification |
|---|---|
| `src/detection.py` | second pipeline concurrent, hors chemin, non testé |
| `src/tracker.py` | second moteur de franchissement, incohérent avec la FSM |
| `src/visualizer.py` | doublon de rendu + libellé « ByteTrack » |
| `src/calibration.py` | doublon de `main.calibrate()` |
| `src/reid.py` | code mort ; galerie remplacée par `identity_manager` |
| `src/hybrid_tracker.py` | non branché ; docstring ByteTrack/DeepSORT ; logique spatio-temporelle fusionnée |
| `bbox_height_locker.py` (racine) | doublon octet pour octet |
| `src/configs/bytetrack_custom.yaml` | configuration ByteTrack résiduelle |
| `predictions_systeme.json` | artefact de run écrit au mauvais endroit |
| `occupancy_types.TrackState.EN_ZONE_LIGNE`, `A_RETOURNE` | jamais atteint / jamais lu |
| 11 champs morts de `LogicalTrack` | jamais lus ni écrits |
| `OccupancyManager.line2_px()`, `line2_p1/p2`, `gate_width_px`, `is_zenithal` | jamais appelés / jamais lus |
| CLI `--line2-p1`, `--line2-p2`, `--gate-width`, `--zenithal` (partiel), `--no-stabilizer` (devient config) | sans effet ou remplacés par la configuration |
| Mentions ByteTrack / DeepSORT dans le README et les docstrings | ne correspondent plus au comportement réel |

### 3.2 Conservé en expérimental, derrière un flag de configuration (ablation §9)

| Élément | Flag | Test qui prouve qu'il fonctionne si activé |
|---|---|---|
| `BBoxHeightLocker` | `stabilization.use_bbox_locker` (**true** dans la référence) | test unitaire niveau 1 + `test_bbox_locker_integration.py` (clé `person_id`, purge, chute signalée, parcours complet) |
| `AnchorStabilizer` | `stabilization.use_anchor_stabilizer` | test unitaire niveau 1 |
| Extracteur d'apparence profond (ex-`reid.py`) | `reid.external_reid.enabled` | test unitaire niveau 1 (embedding normalisé, déterministe) |
| ReID long terme lui-même | `reid.long_term.enabled` | niveaux 2 et 3 (baseline 1 vs variante 1) |

### 3.3 Conservé tel quel

- `AnchorStabilizer` et `BBoxHeightLocker` : logique interne inchangée (seul le câblage de ce dernier a été corrigé, cf. §2.11).
- Persistance BoT-SORT (`persist=True`, `custom_botsort.yaml`) : correct, conforme §3.2.
- Lecture YAML de la configuration du tracker : conservée, déplacée derrière le chargeur de configuration.
- `compute_anchor` : conservée, déménagée dans `geometry.py` (suppression du paramètre `is_zenithal` mort).

---

## 4. Questions soumises à arbitrage (spec §0.1 et §10)

1. **`src/anchor_stabilizer.py` / `src/bbox_height_locker.py` restent dans `src/`** plutôt que déplacés dans `src/experimental/`. Choix retenu : les laisser en place (le dépôt « reste reconnaissable ») et les neutraliser par configuration `false`, avec un test qui prouve leur activation. À confirmer.
2. **`evaluate_system.py` est corrigé sur place** (JSON Lines, secondes, F1 franchissements + erreur d'occupation + FPS) plutôt que réécrit sous `tools/`. À confirmer.
3. **`tests/` doit sortir du `.gitignore`.** Actuellement `tests/`, `*test*` et `test_*` sont ignorés, donc toute suite de tests est invisible du dépôt : c'est la cause racine de la dérive constatée en §2.16. Je vais ajouter des exceptions (`!tests/`, `!tests/**`) — à confirmer, car cela change le contenu versionné du dépôt.
4. **Vidéos de test** : `test/*.mp4|avi` sont ignorés et absents du dépôt. Les niveaux 1 à 3 (géométrie, trajectoires synthétiques, intégration sans caméra) sont donc les seuls réalisables ici ; le **niveau 4** (vidéos réelles annotées) et le **protocole de calibration §9.1-9.3** (ensembles disjoints, vérité terrain, critères chiffrés) ne peuvent être que *outillés* (formats de fichiers, script d'évaluation, tableau de résultats vide), pas exécutés sans les vidéos et les annotations. Confirmé comme périmètre de cette passe.
5. **Pas de matériel GPU** : `torch 2.13.0+cpu` est installé. Les baselines GPU de la §9.4 (YOLO11n vs YOLO11s, FPS matériel de référence) ne peuvent pas être mesurées de façon représentative ici ; le harnais sera fourni et les mesures marquées comme « à exécuter sur le matériel de référence ».

### 4.1 Décisions finalement appliquées

1. `src/anchor_stabilizer.py` et `src/bbox_height_locker.py` **restent dans `src/`**, neutralisés par `stabilization.use_*: false`, avec des tests qui prouvent leur activation (`tests/test_stabilization_flags.py`).
2. `evaluate_system.py` est **corrigé sur place** : JSON Lines validé par le schéma, appariement en secondes, F1 des franchissements, erreur absolue moyenne d'occupation, FPS et latence, verdict explicite contre le critère d'acceptation.
3. `tests/`, `*test*` et `test_*` ont été **retirés du `.gitignore`** ; `results/` et `data/videos/` y ont été ajoutés. La suite de tests est désormais versionnée.
4. Niveau 4 et protocole §9.1-9.3 : **outillés** (`config/protocol.yaml`, `scripts/run_protocol.py`, `docs/protocole_evaluation.md`), non exécutés faute de corpus annoté.
5. Aucune mesure de FPS de référence n'est publiée : le harnais fournit FPS mesuré et latences par étape, et `acceptance.min_fps` reste `null` en attendant le matériel documenté.

Les dépendances `scipy` et `scikit-learn`, déclarées dans `requirements.txt` mais plus importées nulle part après suppression de `hybrid_tracker.py`, ont également été retirées.

### 4.2 Mise à jour — sélection manuelle obligatoire de la ligne

Décision ultérieure, qui **remplace** le repli décrit en §2.8 (« `main.calibrate()`
est conservé et devient le seul calibrateur (avec repli config) ») :

- `line.p1` / `line.p2` sont **retirés du schéma de configuration** et activement
  refusés s'ils réapparaissent dans un YAML (une ligne écrite dans le fichier
  laisserait croire qu'elle est utilisée) ;
- les arguments `--line-p1`, `--line-p2` et `--calibrate` sont **supprimés** : ils
  permettaient de contourner la validation manuelle ;
- `src/calibration.py` est recréé, non plus comme doublon de `detection.py` mais
  comme **unique** point de définition de la ligne (`LineSelection` testable sans
  écran + `select_line` pour la fenêtre OpenCV) ;
- `OccupancyManager` exige désormais la ligne validée : plus aucun repli sur la
  configuration, donc aucune divergence possible entre la ligne affichée et celle
  qui compte ;
- `--inside-side` remplace ces arguments comme simple **orientation initiale**
  proposée dans la fenêtre, inversable en direct avec « I » ;
- `--mode comptage` rend explicite l'étape « l'utilisateur choisit le mode
  comptage » (seul mode officiel, journalisé dans `SESSION_START`) ;
- `--no-show` ne dispense pas de la sélection manuelle : sans interface
  graphique, le démarrage échoue en code 4 avec un message explicite au lieu de
  fabriquer une ligne.

Aucun résidu de l'ancienne double ligne L1/L2 n'est réintroduit : une régression
vérifie l'absence de `calibrate_two_lines`, `line2*`, `gate_width` et
`line_pos_ratio` dans `src/`, et le schéma `LineConfig` ne porte que
`inside_side`, `on_line_policy` et `min_length_ratio`.
### 4.3 Mise à jour — audit des compteurs d'occupation et pertes de détection sur flou

Audit demandé par le prompt de correction ciblée (« occupancy_confirmee ne doit
décrémenter que sur un OUT confirmé »). Méthode : balayage de `src/*.py`
(recherche des écritures sur `total_in` / `total_out` / `total_new` /
`initial_occupancy`, des purges et timeouts, des branches `except`, et de tout
`max(0, occupancy)` masquant un décompte).

**Écritures trouvées — toutes dans `src/occupancy_manager.py` :**

| Compteur | Ligne | Déclencheur | Verdict |
|---|---:|---|---|
| `initial_occupancy += 1` | 733 | action FSM `EMPLACE_INITIAL` (warm-up), une seule fois par `person_id` | conforme |
| `total_in += 1` | 864 | `_count_in` ← `COMPTE_ENTREE`, `CONFIRME_ENTREE_DIFFEREE` | conforme |
| `total_out += 1` | 900 | `_count_out` ← `COMPTE_SORTIE`, `CONFIRME_SORTIE_DIFFEREE`, `RECUPERE_EXTERIEUR` (réapparition extérieure cohérente) | conforme |
| `total_new += 1` | 979 | `_count_new`, gardé par les conditions `NEW` du FSM | conforme |

Aucun timeout, aucune purge, aucun `except` et aucune branche `else` ne touche
ces compteurs ; aucun `max(0, occupancy)` ne porte sur l'occupation
(`max(0, …)` n'apparaît que sur des indices de frame et des coordonnées de
crop). Les purges passent par `identity_manager.py:852` et émettent `PURGE`,
dont le schéma (`src/events.py:138`) **exclut** toute interprétation comme
sortie ; `_count_out` n'est jamais atteint depuis ce chemin.

**Cause réelle du décrément observé.** La formule `initial + IN + NEW - OUT`
était correcte, mais la valeur **publiée** par frame (`occupancy_confirmed`,
celle lue par le tableau de bord, les snapshots et `evaluate_system.py`)
était recalculée en excluant les pistes non visibles : une personne occultée, ou
dont la piste était purgée après la grâce, disparaissait donc de la valeur
affichée **sans** `total_out` — d'où l'impression d'une décrémentation par
purge/timeout.

**Correction.** `occupancy_confirmed` est désormais un alias explicite de
`occupancy_operational` (`occupancy_manager.py:183-208`), bilan
`initial_occupancy + total_in + total_new - total_out`, jamais recalculé à partir
des pistes visibles et jamais borné : une occultation, une absence de détection
ou une purge n'y changent rien. `occupancy_uncertain` compte les personnes
comptées mais non observées, et l'invariant
`occupancy_observed + occupancy_uncertain == occupancy_operational` est vérifié à
chaque frame puis journalisé en cas d'écart, sans correction silencieuse.
`evaluate_system.py` mesure donc la MAE sur l'occupation **opérationnelle**.

**Pertes de détection sur flou.** `model.confidence` passe de 0,25 à 0,10
(configurable, `config/pipeline.yaml`), avec les seuils tracker distincts et
validés `track_high_thresh = 0,4`, `track_low_thresh = 0,1`,
`new_track_thresh = 0,7`. L'invariant `model.confidence <= track_low_thresh` est
vérifié au chargement et toute divergence est journalisée en `CONFIG_WARNING`.
Résultats de la comparaison `model.confidence` × `track_low_thresh` sur le
scénario synthétique flou : `docs/protocole_evaluation.md` §1.1.

**Tests ajoutés** (niveaux 1-3, sans caméra) : `tests/test_warmup.py`,
`tests/test_track_loss_recovery.py` (dont la matrice de seuils),
`tests/test_occupancy_operational.py`, `tests/test_track_diagnostics.py`.

## 5. Plan d'exécution retenu (étapes revuables)

1. Suppression des modules morts + nettoyage des mentions ByteTrack/DeepSORT + `tests/` réintégré.
2. `config/pipeline.yaml` (schéma §2) + `src/config.py` (chargement, validation §5.6, copie résolue §6.3).
3. `src/events.py` (JSON Lines incrémental, schéma versionné §6.1-6.2, `session_id` §6.3).
4. `src/geometry.py` (géométrie canonique unique, zone morte **relative**, garde-fou bord, validation de ligne §5.7).
5. `src/fsm.py` (table de transition explicite §4.2 + politique de disparition §4.3).
6. `src/identity_manager.py` (fusion de `IDManager` + `ReIDGallery` + `HybridAssociation` → deux niveaux §3).
7. Refonte `src/occupancy_manager.py` + `src/occupancy_types.py` (occupation encadrée §4.4, garde-fous §4.5).
8. Refonte `src/main.py` (chemin unique, headless réel, FPS mesuré, latences §6.5, fin de flux §5.5).
9. Tests niveaux 1-3 + schéma + headless + source défaillante, `run.sh`, `requirements.txt` figées, `evaluate_system.py`, note de confidentialité (livrable 8).
