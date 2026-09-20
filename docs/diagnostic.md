# Diagnostic — causes racines (référence permanente)

> Extrait de l'ancien `CLAUDE.md` monolithique. Lire d'abord `CLAUDE.md`.
> Ce fichier ne se réécrit pas : il documente l'analyse qui a produit les lots A–D.

### 4.1 Symptôme 1 : swaps, fusions, personnes jamais redétectées

1. **`match_thresh: 0.9`** (`custom_botsort.yaml`) est un seuil de **coût**, pas
   de similarité — l'association accepte un appariement jusqu'à IoU 0,1. Deux
   personnes voisines en occlusion dense ont typiquement un IoU de 0,2–0,3
   entre elles : l'appariement croisé passe facilement. Défaut Ultralytics :
   0,8.
2. **`proximity_thresh: 0.5`** neutralise le ReID natif au mauvais moment.
   Dans `BOTrack.get_dists` (Ultralytics), l'apparence n'est consultée que si
   l'IoU est déjà ≥ 0,5. Une personne qui ressort d'occlusion a un IoU quasi
   nul avec sa boîte prédite → apparence ignorée exactement quand elle serait
   utile.
3. **`model: auto`** dans BoT-SORT réutilise les features du backbone de
   détection comme identité — peu discriminant entre deux personnes de même
   classe.
4. **Le descripteur de niveau 2 est un histogramme HSV global**
   (`HistogramAppearanceExtractor` dans `identity_manager.py`) sur le crop
   entier. En salle de classe, ça encode surtout la couleur des chaises/tables/
   mur. `similarity_threshold: 0.40`, `safety_margin: 0.10` — l'écart
   inter-personnes est souvent inférieur au bruit du descripteur.
5. **La correction de swap s'auto-empoisonne.** `_correct_cross_swaps` compare
   l'apparence courante à `rec.feature`, mis à jour à chaque frame via
   `_blend` avec `feature_momentum = 0.85` codé en dur. Après un swap non
   détecté, la référence contient déjà ~56 % de l'apparence de l'autre
   personne au bout de 5 frames — la correction devient aveugle juste au
   moment où elle serait utile.
6. **Deux verrous cumulés empêchent la ré-attribution après perte longue :**
   - `new_track_thresh: 0.7` exige une détection à confiance ≥ 0,7 pour créer
     une nouvelle piste. Une personne assise partiellement masquée sort à
     0,35–0,6 : détectée par YOLO, mais jamais identifiée.
   - La mémoire de galerie (`grace_period_seconds + purge_retention_seconds` =
     10 s avec `gallery_retention_seconds: null`) est plus courte que le
     maximum d'occlusion mesuré (jusqu'à 13,5 s sur sessions réelles).
   - Incohérence de base de temps : `tracker.track_buffer` est en **frames**
     (150), la galerie ReID en **secondes**. Sur fichier à 30 ips, 150 frames
     ≈ 5 s ; sur caméra live à ~3,5 FPS traités, 150 frames ≈ 43 s. Les deux
     mécanismes ne couvrent pas la même durée selon la source.
7. **Boîte fusionnée = simple journalisation, aucun effet.**
   `check_box_width_growth` (`geometry.py`) émet `POSSIBLE_MULTI_PERSON_BOX` et
   bloque l'écriture galerie, mais ne sépare rien, ne maintient rien. La
   personne absorbée disparaît du comptage comme si elle était sortie. Angle
   mort supplémentaire : le détecteur a besoin d'un historique de largeur
   *avant* la fusion — deux personnes assises côte à côte dès la première
   frame ne déclenchent jamais le signal.

### 4.2 Symptôme 2 : la tête n'est pas exploitée comme prévu

1. **`presence.head_assist.enabled: false`** par défaut. Tant que ce flag est
   à `false`, aucune ligne du chemin pose ne s'exécute.
2. **Le modèle pose est absent du dépôt** (`models/yolo11s-pose.pt`,
   `pose_model_expected_sha256: null`). L'activer sans ce fichier fait échouer
   `verify_weights` (`FileNotFoundError`, retour 3).
3. **Le point tête peut devenir périmé.** `_handle_missing`
   (`occupancy_manager.py`) relit `current_head_point`, figé à la dernière
   frame où la personne était observée — sans garde-fou, ça peut maintenir
   jusqu'à 10 s une personne réellement sortie du champ.
4. **Incompatibilité avec la personne assise.** La condition
   `not self.config.presence.head_assist.enabled and track.height_drop_streak
   >= ...` désactive, dès que le head_assist est actif, la réadaptation de
   hauteur pour une personne qui s'est assise et reste assise — la
   fonctionnalité censée aider les personnes assises dégrade justement ce cas.
5. **Divergence rapport/config sur les points-clés.** Un rapport antérieur
   annonçait 5 keypoints (`nose`, `left_eye`, `right_eye`, `left_ear`,
   `right_ear`), la config peut n'en déclarer que 3 (les oreilles manquantes
   sont celles qui survivent le mieux à un profil).
6. **Trou de test.** Les tests de `test_head_assist_presence.py` peuvent
   injecter `head_point` à la main sans jamais passer par le modèle pose réel
   — la suite est verte sans que la fonctionnalité ait jamais tourné en
   pratique.
7. **La sémantique de `occupancy_observed` doit être vérifiée avant d'activer
   quoi que ce soit.** Si le code actuel garde une personne `OCCULTEE` dans
   `occupancy_observed` jusqu'à expiration de la grâce (indépendamment de la
   tête), alors l'assistance tête n'a **structurellement aucun effet
   mesurable** sur ce compteur — voir la décision tranchée en section 7.1
   ci-dessous, à vérifier/appliquer avant toute mesure.

---
