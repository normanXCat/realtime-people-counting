# Handoff — Correctif occlusion dense et assistance tête (lots A et B)

Destinataires : opérateur / ingénieur qui reprend le sujet. Le livrable de
mesure est `docs/correctif_occlusion.md` ; ce handoff dit **ce qui a changé,
comment le vérifier, et ce qui reste ouvert**. Le lot C du prompt correctif
(`prompt/prompt_correctif_occlusion_et_tete.md`) **n'a pas été exécuté**.

---

## 1. Ce qui a changé (résumé opérationnel)

### Réglages (`config/pipeline.yaml`, `src/configs/custom_botsort.yaml`)

| Paramètre | Avant | Après |
|---|---|---|
| `tracker.match_thresh` | 0,90 | 0,75 |
| `tracker.proximity_thresh` | 0,50 | 0,80 |
| `tracker.new_track_thresh` | 0,70 | 0,45 |
| `reid.long_term.gallery_retention_seconds` | `null` (fenêtre 10 s) | `12.0` (fenêtre 17 s) |
| `presence.head_assist.keypoints_used` | 3 points | 5 points (oreilles incluses) |

Les trois fichiers YAML et `src/config.py` doivent rester **synchronisés** : un
test le vérifie (`tests/test_config_validation.py`).

### Nouveaux paramètres (tous dans `config/pipeline.yaml`, tous validés)

| Paramètre | Défaut | Rôle |
|---|---|---|
| `presence.head_assist.max_head_staleness_frames` | `1` | un point tête plus vieux qu'une frame ne maintient plus la présence dans `_handle_missing` |
| `reid.long_term.feature_momentum` | `0.85` | momentum du descripteur (était codé en dur) ; borne `0 ≤ x < 1` |
| `reid.appearance_continuity.freeze_gallery_frames` | `3` | frames de gel du descripteur après une discontinuité (borne min 3) |
| `reid.long_term.descriptor.*` | voir YAML | `horizontal_crop_ratio` 0,15, `bands` 3, `band_weights` `[1, 1, 0.5]`, `bins_hue` 24, `bins_saturation` 8 |
| `geometry.multi_person_box.head_count_enabled` | `true` | active le second signal (nombre de têtes dans la boîte) |
| `geometry.multi_person_box.head_separation_ratio` | `0.4` | séparation horizontale minimale entre deux têtes, en fraction de largeur de boîte |

### Code (aucun déplacement de module, aucun renommage)

- `src/occupancy_manager.py`
  - **B.1** : le terme « `not presence.head_assist.enabled` » qui neutralisait la
    réadaptation de hauteur d'une personne assise est retiré. Une chute
    confirmée sur `height_drop_confirm_frames` frames réadapte de nouveau
    l'historique de hauteur, **assistance activée ou non**.
  - **B.2** : `_handle_missing` exige un point tête frais
    (`frame_index - head_point_frame_index <= max_head_staleness_frames`).
  - **B.5** : signal multi-personnes par nombre de têtes (dès la première frame,
    sans historique de largeur) et événement
    `INCONSISTENT_STATE kind=identity_absorbed_by_multi_person_box` ; une purge
    consécutive porte la raison `absorbed_by_multi_person_box`.
  - câblage de `reid.long_term.feature_momentum` et de `reid.swap_correction`
    (ce dernier n'était pas transmis : `config` n'avait aucun effet).
- `src/identity_manager.py`
  - descripteur HSV **par bandes** sur crop érodé (`HistogramAppearanceExtractor`) ;
  - **B.3.b** : `IdentityRecord.appearance_suspect_until_frame` gèle `_blend`
    pendant `freeze_gallery_frames` après une discontinuité — la référence ne
    dérive plus vers l'apparence de l'autre personne après un swap.
- `src/geometry.py` : `detect_multi_person_by_head_count`.
- `src/main.py` : avertissement de base de temps (A.4) — `track_buffer` est en
  **frames**, la galerie ReID en **secondes**. Avertissement au démarrage pour
  une caméra live, puis complété par la durée réelle dès le FPS mesuré.
  **Aucune conversion automatique** n'est appliquée.

---

## 2. Comment vérifier

```bash
./.venv/bin/python -m pytest                 # 606 passed, couverture 93,5 % (seuil 85 %)
./.venv/bin/python -m py_compile src/*.py
./.venv/bin/python -m pytest tests/test_correctif_occlusion.py -q   # 21 tests du correctif
```

Les tests dédiés couvrent : la base de temps (A.4), la personne assise avec
assistance **activée et désactivée** (B.1), la fraîcheur du point tête (B.2),
le momentum configurable (B.3.a), le gel du descripteur et la correction croisée
après 5 frames de swap (B.3.b), le descripteur par bandes (B.4), le signal par
nombre de têtes et l'absorption (B.5), et l'extraction d'un point tête depuis un
**vrai tenseur `keypoints`** `(N, 17, 2)` / `(N, 17)`.

---

## 3. Ce qui n'est PAS validé (à lire avant toute conclusion)

1. **Le lot A n'est pas validé.** Sur
   `test/5121204_School_Classroom_1280x720.mp4` (300 frames, via
   `scripts/measure_tracker_swaps.py`), les identifiants techniques créés passent
   de **11 à 15** et les `technical_id_changes` de **2 à 6**, alors
   qu'**aucun swap** n'a été observé dans les deux configurations (0
   `TRACK_ID_APPEARANCE_DISCONTINUITY`). Le symptôme que le lot A devait corriger
   n'a donc pas été reproduit, et les seuils n'ont pas été figés. Les trois
   paramètres restent marqués PROVISOIRE.
2. **L'assistance tête n'a jamais tourné sur une vidéo réelle.**
   `models/yolo11s-pose.pt` est absent du dépôt et
   `presence.head_assist.pose_model_expected_sha256` vaut `null` : avec
   `enabled: true`, `verify_weights` lève `FileNotFoundError` et `main` retourne
   le code 3. C'était la cause immédiate du symptôme « aucun effet observable ».
3. **`similarity_threshold` (0,40) non recalibré.** L'histogramme
   **inter-identités** exige une vérité terrain MOT, absente du dépôt. Seule la
   distribution intra-piste est disponible (p1 ≈ 0,83 / 0,90, médiane ≈ 0,99,
   aucune valeur sous 0,50).
4. **Les deux runs de mesure ne sont pas reproductibles au frame près** : le
   nombre de détections diffère de +5 % à configuration de seuil YOLO identique.
   Une répétition (≥ 3 runs par configuration) est nécessaire avant de conclure
   fermement à une régression.

---

## 4. Prochaine étape recommandée — lot C, prérequis seulement

Avant toute décision d'architecture (option « tête = ancre de comptage » ou
« tête = garant de présence ») :

1. **Trancher le sort du lot A** : mesurer sur un clip où le swap se produit
   réellement, et tester `match_thresh`, `proximity_thresh` et
   `new_track_thresh` **séparément** (le prompt fournit les justifications, mais
   la mesure ne les confirme pas). Le choix conservateur — revenir à
   `match_thresh: 0.9` / `new_track_thresh: 0.7` — reste disponible en une ligne
   de YAML.
2. **Rendre l'assistance tête exécutable** : télécharger
   `yolo11s-pose.pt`, renseigner son SHA-256, mesurer le coût CPU (le pipeline
   tourne à ~2–4 FPS sur CPU ; un modèle pose à `imgsz: 960` sera plus lent).
   Ne pas l'activer par défaut avant cette mesure.
3. **Produire la vérité terrain inter-identités** (2–3 clips courts annotés) pour
   pouvoir proposer une valeur de `similarity_threshold` avec histogrammes à
   l'appui.
4. **Ne pas écrire de code du lot C** (ancre tête, ligne dans le plan des têtes,
   `LocalScale` remplacée par distance inter-oreilles, retrait de
   `BBoxHeightLocker` du chemin) tant que ces trois prérequis ne sont pas
   satisfaits.

---

## 5. Reproduire les mesures

Le protocole exact (vidéo, ligne archivée, nombre de frames, commandes) est
décrit dans `docs/correctif_occlusion.md` §1. La ligne de référence utilisée pour
le protocole à 60 frames est archivée dans
`results/20260917T180747Z/calibration.json` (la sélection de ligne étant
obligatoirement manuelle, un pilote temporaire a rejoué cette ligne en headless ;
il n'a pas été conservé). Les mesures étendues passent par
`scripts/measure_tracker_swaps.py`, qui écrit une ligne JSON par configuration
dans `results/swap_measurements*.jsonl`.
