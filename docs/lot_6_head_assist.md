# Lot 6 — Activation réelle du head_assist

> Ex-« lot C ». Ne pas démarrer avant que le lot 5 soit mesuré et accepté.
> Lire d'abord `CLAUDE.md` et `docs/diagnostic.md`. Contenu inchangé par la
> révision du plan ; §7.1 reste un prérequis bloquant à toute mesure de ce lot.

### 7.1 Décision tranchée sur `occupancy_observed` (à appliquer avant de mesurer quoi que ce soit)

Vérifier le comportement actuel de `occupancy_observed` dans
`occupancy_manager.py`. **Si** une piste `OCCULTEE` y reste jusqu'à expiration
de la grâce indépendamment de tout signal tête (comportement déjà rencontré
sur une tentative précédente), **alors** le head_assist n'a aucun effet
mesurable par construction — appliquer la correction suivante avant de
poursuivre :

- Une piste qui passe en `OCCULTEE` doit **immédiatement** sortir de
  `occupancy_observed` et basculer dans `occupancy_uncertain` — **sauf** si le
  head_assist la maintient (point tête frais, cf. B.2).
- `occupancy_operational` reste inchangé par cette décision dans tous les cas.
- Condition résultante : `inside_occupancy and state is not ABSENTE and
  (state is not OCCULTEE or presence_maintained_by_head)`.
- Mettre à jour la docstring de `occupancy_observed` pour qu'elle corresponde
  au code.
- Tests : piste `OCCULTEE` sans head_assist sort immédiatement de
  `occupancy_observed` ; piste `OCCULTEE` avec point tête frais y reste ;
  `occupancy_operational` ne varie dans aucun des deux cas (sauf sortie
  confirmée).

### 7.2 Obtenir le modèle pose

Télécharger `yolo11s-pose.pt` (poids officiel Ultralytics), calculer son
SHA-256, le renseigner dans `pose_model_expected_sha256` (ne pas laisser
`null`). Documenter le chemin et la source dans le `README` ou `docs/`.

### 7.3 Activer le flag

```yaml
presence:
  head_assist:
    enabled: true   # était false — PROVISOIRE, à confirmer après mesure
```

Vérifier le démarrage sans erreur, mesurer l'impact FPS (le modèle pose à
`imgsz: 960` est plus lourd que le modèle de détection seul).

### 7.4 Combler le trou de test

Ajouter un test d'intégration avec un vrai tenseur `keypoints` simulé de forme
`(N, 17, 2)` / `(N, 17)` (format réel Ultralytics), vérifiant l'extraction du
point tête et l'absence de valeur inventée quand une boîte n'a pas de
détection correspondante.

### 7.5 Mesure ciblée — le cas qui compte

Sur la vidéo à occlusion dense du corpus (section 1), identifier les segments où une personne est
assise/occultée par du mobilier ou une autre personne pendant au moins 3
secondes. Pour chaque segment, `head_assist` désactivé puis activé, en nommant
explicitement le compteur rapporté :

- reste-t-elle dans `occupancy_observed` pendant tout le segment, ou
  bascule-t-elle dans `occupancy_uncertain` ?
- `occupancy_operational` doit rester strictement identique dans les deux cas
  (sauf sortie réelle) — toute variation est une régression à signaler ;
- combien de temps avant un passage en `OCCULTEE` ;
- l'identifiant reste-t-il stable à la sortie du segment ?

Mesurer aussi l'effet du signal multi-têtes (lot 5, B.5) sur les fusions présentes
dans la vidéo, s'il y en a.

### 7.6 Ce qui doit rester vrai après ce lot

- Aucun événement `CROSSING_*` ne doit jamais être déclenché ou influencé par
  le point tête (`test_head_point_crossing_line_never_triggers_crossing`
  reste vert avec le modèle pose réel, pas seulement en injection manuelle).
- `max_presence_extension_seconds` continue de borner la durée maximale de
  maintien — l'activation ne doit pas le transformer en maintien indéfini.

---
