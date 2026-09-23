# Mesures avant/après — chemin de secours « galerie » (B3)

Sortie de la commande de mesure demandée, avec et sans `gallery_rescue`, sur les
vidéos de test disponibles dans l'arbre de travail.

## Protocole

Le même chemin de code que le pipeline est rejoué (YOLO11 + BoT-SORT +
`IdentityManager`), sans ligne virtuelle — le comptage de franchissements n'a
aucun effet sur la stabilité des identités :

```bash
python scripts/measure_tracker_swaps.py --source test/<video>.mp4 \
    --max-frames 300 --no-gallery-rescue --out results/lot_gallery_rescue_before.jsonl
python scripts/measure_tracker_swaps.py --source test/<video>.mp4 \
    --max-frames 300 --gallery-rescue    --out results/lot_gallery_rescue_after.jsonl
```

Configurations effectivement appliquées (identiques dans les deux colonnes) :
BoT-SORT `track_buffer=240`, `appearance_thresh=0.35`, `track_low_thresh=0.1`,
`new_track_thresh=0.7` ; correction de swap active, `margin=0.08`,
`group_proximity_ratio=0.5` ; par défaut `gallery_rescue_min_confidence` se replie
sur `track_low_thresh` (0,1).

« Nouvelles identités créées pour une personne déjà en galerie » = événements
`REID_NEW` précédés d'au moins un candidat plausible (`candidates_evaluated > 0`) :
c'est le proxy direct d'une piste perdue alors que la personne était encore
ré-associable.

## Résultats

| vidéo | frames | gallery_rescue | swap_corrections_applied | discontinuities | nouvelles identités (persons_created) | technical_id_changes | REID_GALLERY_RESCUE | nouvelles identités alors qu'un candidat galerie existait |
|---|---|---|---|---|---|---|---|---|
| `test/video2.avi` | 300 | **off** | 0 | 0 | 1 | 0 | 0 | 0 |
| `test/video2.avi` | 300 | **on**  | 0 | 0 | 1 | 0 | 0 | 0 |
| `test/fort_occ4.mp4` | 300 | **off** | 0 | 0 | 9 | 3 | 0 | 0 |
| `test/fort_occ4.mp4` | 300 | **on**  | 0 | 0 | 9 | 3 | 0 | 0 |

Lignes brutes : `results/lot_gallery_rescue_before.jsonl`,
`results/lot_gallery_rescue_after.jsonl`.

## Conclusion — décision d'activation

**Aucune amélioration nette, aucune hausse des fausses réassociations (les deux
colonnes sont identiques).** Sur ces deux vidéos, aucune observation non trackée de
la gamme `[track_low_thresh, new_track_thresh[` n'a été présentée à la galerie dans
une situation où une identité était ré-associable : `REID_GALLERY_RESCUE = 0`.

En conséquence, conformément à la consigne, **`gallery_rescue_enabled` reste à
`false` dans le YAML livré**. Le mécanisme est implémenté, testé (unitaire, intégré
et `main`) et mesurable séparément (`REID_GALLERY_RESCUE` + bloc
`gallery_rescue` de `summary.json`), mais son activation est une décision
explicite à prendre sur une vidéo **où le symptôme est reproduit**.

Deux réserves de mesure, documentées pour ne pas surinterpréter le tableau :

1. les vidéos qui ont produit les diagnostics d'occlusion antérieurs
   (`test/video3.mp4`, `test/5121204_School_Classroom_1280x720.mp4`) **ne sont pas
   présentes** dans l'arbre de travail : la scène de salle de classe dense n'a pas
   pu être rejouée ;
2. dans les runs mesurés, BoT-SORT a confirmé au moins une piste sur presque toutes
   les frames contenant des détections. Or `on_predict_postprocess_end`
   d'Ultralytics ne conserve **que** les pistes confirmées : une détection non
   associée n'apparaît dans `result.boxes` (sans `id`) que lorsqu'aucune piste n'a
   survécu sur la frame. Le chemin de secours ne peut donc rien faire quand une
   autre personne est trackée sur la même frame — c'est une limite connue du canal
   `result.boxes`, pas du mécanisme de rattachement lui-même.
