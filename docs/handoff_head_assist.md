# Handoff — Assistance par détection de tête en cas d'occlusion des pieds

Ce document détaille la conception, l'implémentation, la validation et les consignes d'exploitation de la fonctionnalité d'**assistance de présence par détection de tête**.

---

## 1. Contexte et objectif

Dans les environnements d'amphithéâtre ou de salle de classe, les personnes assises voient très fréquemment la partie inférieure de leur corps (jambes et pieds) masquée par le mobilier (tables, chaises, pupitres).
Dans le pipeline standard reposant sur l'ancre pieds (bas-centre de la boîte englobante) :
- Une chute de hauteur brutale est détectée (`check_anchor_height_drop`).
- Ou la boîte est tronquée et touche le bord du mobilier.
- Sans signal complémentaire, la personne risque d'être considérée comme perdue (`OCCULTEE`), puis purgée après la période de grâce, entraînant une sous-estimation de l'occupation opérationnelle.

La présente fonctionnalité fournit un signal complémentaire purement additif : **l'assistance de tête via un modèle d'estimation de pose**.

---

## 2. Principes architecturaux stricts

1. **Additif uniquement** :
   - Désactivé par défaut via `presence.head_assist.enabled: false`.
   - Tant que le drapeau est à `false`, aucun code spécifique ne s'exécute (pas de modèle pose, pas d'inférence de points-clés, aucune régression sur le pipeline de référence).
2. **Découplage absolu entre présence et franchissement** :
   - Le point tête **ne participe jamais** au calcul de la distance signée (`line.distance`), ni à la détection de franchissement de ligne (`line.crossing`), ni aux décisions `IN` / `OUT` / `NEW`.
   - La confirmation d'un franchissement continue d'exiger le passage effectif de l'ancre pieds.
3. **Garde-fou temporel borné** :
   - Le maintien de présence par la tête ne peut excéder `max_presence_extension_seconds` (défaut : 10,0 s).
   - Au-delà de ce délai sans retour d'une ancre pieds fiable, la piste bascule normalement vers `OCCULTEE` (grâce puis purge standard).

---

## 3. Détail des composants modifiés et créés

### 3.1 `src/geometry.py`
- Constante `COCO_HEAD_KEYPOINTS` associant les 5 points de tête aux indices COCO :
  - `nose` : 0
  - `left_eye` : 1
  - `right_eye` : 2
  - `left_ear` : 3
  - `right_ear` : 4
- Fonction `extract_head_point(keypoints_xy, keypoints_conf, keypoints_used, min_confidence)` :
  - Filtre les points demandés selon le seuil de confiance.
  - Calcule le barycentre `(mean_x, mean_y)` des points fiables et renvoie la confiance minimale observée.
  - Renvoie `(None, None)` si aucun point n'atteint le seuil.

### 3.2 `src/config.py` & `config/pipeline.yaml`
- Nouvelles dataclasses figées :
  - `HeadAssistConfig` : `enabled`, `pose_model_path`, `pose_model_expected_sha256`, `min_keypoint_confidence`, `keypoints_used`, `max_presence_extension_seconds`.
  - `PresenceConfig` : englobe `head_assist`.
- Ajout de `device: str = "cpu"` dans `ModelConfig`.
- Validation stricte dans `build_config` :
  - `min_keypoint_confidence` borné dans $[0.0, 1.0]$.
  - `max_presence_extension_seconds > 0.0`.
  - `keypoints_used` non vide et restreint aux 5 points de tête COCO autorisés.
- Non-régression `to_dict()` et `write_resolved_config()` :
  - La section `presence` est omise de la sérialisation lorsque `enabled: false`, garantissant une stricte conformité aux schémas existants.

### 3.3 `src/events.py`
- Ajout des schémas d'événements :
  - `PRESENCE_MAINTAINED_BY_HEAD` : `person_id`, `reason`, `head_confidence`, `feet_anchor_status`, optionnels `frame`, `head_point`.
  - `PRESENCE_EXTENSION_EXPIRED` : `person_id`, `duration_s`, `max_extension_seconds`, optionnels `frame`, `reason`.

### 3.4 `src/identity_manager.py` & `src/occupancy_types.py`
- Ajout de `head_point` et `head_confidence` (optionnels, par défaut `None`) dans `Observation` et `Assignment`.
- Transmission systématique dans les 4 points de création de `Assignment`.
- Ajout de `head_presence_first_s`, `head_presence_expired`, `current_head_point`, `current_head_confidence` dans `PersonTrack`.

### 3.5 `src/occupancy_manager.py`
- `Detection` : ajout de `head_point` et `head_confidence`.
- Transmission dans `step()` vers les `Observation`.
- Dans `_update_track()` :
  - Si l'ancre pieds devient non fiable (`height_drop`, bord ou absence) et que la piste est en état `PRESENTE` :
    - Si `head_point` et `head_confidence >= min_keypoint_confidence` :
      - Maintien en `PRESENTE`.
      - Émission de `PRESENCE_MAINTAINED_BY_HEAD`.
      - Gel de l'ancre de franchissement.
    - Si la durée continue dépasse `max_presence_extension_seconds` :
      - Émission de `PRESENCE_EXTENSION_EXPIRED`.
      - Bascule vers `OCCULTEE`.
    - Si `head_confidence` sous le seuil :
      - Bascule vers `OCCULTEE`.
  - Si l'ancre pieds redevient fiable : réinitialisation du compteur d'extension.
- Dans `_handle_missing()` :
  - Même logique d'évaluation si une tête est identifiée sur la frame manquante.

### 3.6 `src/main.py`
- Chargement conditionnel du modèle :
  - Si `presence.head_assist.enabled: true` -> chargement et vérification de `pose_model_path`.
  - Sinon -> chargement standard de `model.path`.
- Paramètre `device` transmis à `model.track(..., device=config.model.device)`.
- Extraction des points-clés depuis `result.keypoints` lors du tracking et instanciation des `Detection`.

---

## 4. Validation et suite de tests

La suite de tests comprend 4 fichiers dédiés :
1. `tests/test_extract_head_point.py` (6 tests) :
   - Extraction sur point unique fiable.
   - Barycentre de plusieurs points fiables.
   - Exclusion des points sous le seuil.
   - Renvoi de `(None, None)` si aucun point n'est valide.
   - Ignorance des points hors sélection.
2. `tests/test_head_assist_disabled_by_default.py` (4 tests) :
   - Vérification de `enabled: false` par défaut.
   - Champs `head_point` / `head_confidence` initialisés à `None`.
   - Absence totale d'événements d'assistance tête quand inactif.
3. `tests/test_head_assist_presence.py` (4 tests) :
   - Maintien en `PRESENTE` lors d'une chute de hauteur avec tête confiante + émission de `PRESENCE_MAINTAINED_BY_HEAD`.
   - Bascule vers `OCCULTEE` si tête non confiante.
   - Expiration après `max_presence_extension_seconds` + émission de `PRESENCE_EXTENSION_EXPIRED`.
   - Garde-fou de franchissement : un `head_point` franchissant la ligne ne déclenche aucun événement `IN` ni `OUT`.
4. `tests/test_config_validation.py` (7 tests additionnels) :
   - Rejet des confiances hors $[0, 1]$, durées négatives ou nulles, points-clés inconnus ou vides.

**Résultat global de la suite de tests** :
```bash
./.venv/bin/python -m pytest --no-cov tests/
585 passed in 9.78s
```
Zéro régression, 100 % des tests passants.

---

## 5. Guide d'activation en production

Pour activer l'assistance par tête sur une caméra ou une vidéo d'amphithéâtre :

1. **Télécharger les poids de pose** :
   ```bash
   # Placer yolo11s-pose.pt dans models/
   # Calculer le hash sha256 :
   sha256sum models/yolo11s-pose.pt
   ```

2. **Éditer la configuration (`config/pipeline.yaml`)** :
   ```yaml
   presence:
     head_assist:
       enabled: true
       pose_model_path: models/yolo11s-pose.pt
       pose_model_expected_sha256: "<votre_hash_sha256>"
       min_keypoint_confidence: 0.5
       keypoints_used: [nose, left_eye, right_eye]
       max_presence_extension_seconds: 10.0
   ```

3. **Lancer le pipeline** :
   ```bash
   ./run.sh --source video_amphi.mp4 --write-video
   ```
   Les événements `PRESENCE_MAINTAINED_BY_HEAD` attesteront dans `results/<session>/events.jsonl` du maintien des personnes assises derrière les tables.
