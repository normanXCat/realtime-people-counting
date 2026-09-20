# Lot 4 — Seuils BoT-SORT

> Ex-« lot A ». Ne pas démarrer avant que le lot 3 soit mesuré et accepté.
> Lire d'abord `CLAUDE.md` et `docs/diagnostic.md`.
>
> **Amputé de deux sections lors de la révision du plan :** A.2 (mémoire
> d'identité) est remplacée par le lot 2 (rétention hors zone), A.4 (base de
> temps) par le lot 1.1 (`track_buffer` en secondes). Les intitulés A.1 à A.5
> sont conservés pour la traçabilité avec `docs/correctif_occlusion.md`.
>
> **Pourquoi après le lot 3 :** `proximity_thresh` gouverne le moment où
> l'apparence est consultée. Le régler pendant que le descripteur est un
> histogramme HSV peu discriminant reviendrait à calibrer une vanne sur un
> signal de bruit.

### A.1 Seuils BoT-SORT

Modifier `src/configs/custom_botsort.yaml` **et** la section `tracker` de
`config/pipeline.yaml` (les deux doivent rester synchronisés — un test existant
le vérifie) :

| Paramètre | Actuel | Cible |
|---|---|---|
| `match_thresh` | 0.9 | 0.75 |
| `proximity_thresh` | 0.5 | 0.8 |
| `new_track_thresh` | 0.7 | 0.45 |

**Mesurer chaque seuil isolément d'abord**, sur la vidéo à occlusion dense du
corpus (section 1), avant
de les combiner — ne pas les changer tous en même temps sans savoir lequel
contribue à quoi. Étendre `scripts/measure_tracker_swaps.py` si nécessaire pour
isoler `match_thresh` (il n'expose peut-être aujourd'hui que `track_buffer`,
`appearance_thresh`, `proximity_thresh`).

Le garde-fou contre les faux positifs que `new_track_thresh: 0.7` portait doit
être reporté sur le filtre géométrique existant (`min_aspect_ratio_wh`,
`max_aspect_ratio_wh`, `min_box_height_px`, `min_box_width_px`) et sur
`timing.confirmation_seconds`. Documenter ce report en commentaire YAML.

### A.2 Mémoire d'identité — SUPPRIMÉ, voir `docs/lot_2_retention.md`

`gallery_retention_seconds: 12.0` appliqué uniformément ne faisait que
repousser l'expiration. La règle métier du lot 2 (rétention illimitée hors zone
de franchissement) la remplace ; la valeur en secondes ne subsiste que pour les
identités perdues **dans** la zone.

### A.3 Points-clés de la tête

```yaml
presence:
  head_assist:
    keypoints_used: [nose, left_eye, right_eye, left_ear, right_ear]
```

Chercher toute occurrence codée en dur de `("nose", "left_eye", "right_eye")`
dans `src/config.py` (il peut y en avoir plusieurs, à des endroits différents
de repli par défaut) — corriger toutes les occurrences trouvées.

### A.4 Incohérence de base de temps — SUPPRIMÉ, voir `docs/lot_1_base_temps_nms.md`

L'ancienne consigne (« documenter, pas corriger automatiquement », log WARNING
au démarrage) est abandonnée : le lot 1.1 corrige l'incohérence en exprimant
`track_buffer` en secondes, converti au FPS mesuré.

### A.5 Vérification du lot 4

Sur la vidéo à occlusion dense du corpus (section 1), et en regard de non-
régression sur une vidéo à occlusion faible : tableau avant/après sur
`TRACK_ID_APPEARANCE_DISCONTINUITY`, `IDENTITY_SWAP_CORRECTED`,
`TECHNICAL_ID_CHANGED`, `REID_NEW`, nombre d'identités distinctes créées,
`[BILAN]` complet, FPS moyen.

Ajouter aux compteurs listés les métriques de vérité terrain (§3 de
`CLAUDE.md`) : `id_switches_reels`, `fusions_reelles`, `erreur_comptage_max`.

**Interaction à mesurer explicitement dans ce lot** : `new_track_thresh: 0.45`
combiné au seuil NMS relevé du lot 1.2. Les deux se renforcent vers le
sur-comptage (doublons survivants promus en identités). Mesurer la combinaison,
pas seulement chaque seuil isolément, et surveiller le nombre d'identités
créées sur la vidéo à occlusion faible.

**S'arrêter et rendre ce tableau avant de continuer vers le lot 5.**

---
