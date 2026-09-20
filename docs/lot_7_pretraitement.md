# Lot 7 — Prétraitement vidéo

> Ex-« lot D ». Dernier lot. Lire d'abord `CLAUDE.md` et `docs/diagnostic.md`.

> **Ne pas démarrer ce lot avant que le lot 6 soit mesuré et accepté.** Ce lot
> modifie les pixels que le descripteur OSNet du lot 3.a a déjà calibrés
> (distribution de similarité intra/inter-identité) — l'ordre compte, sinon
> toute mesure du lot 3 devient invalide sans qu'on sache pourquoi.

### 8.0 Pourquoi ce lot est cadré aussi strictement

Une première version de cette demande de prétraitement a été écrite sans tenir
compte de l'architecture réelle du pipeline. Trois choses qu'elle demandait
casseraient directement ce qui existe :

- **Le letterboxing est déjà fait par Ultralytics.** `main.py` appelle
  `model.track(source=source, imgsz=config.model.image_size, stream=True)` :
  le letterbox-resize et la normalisation sont internes à Ultralytics. Ajouter
  un letterbox dans un module de prétraitement séparé produit un double
  resize et deux référentiels de coordonnées à réconcilier.
- **Le frame skipping est incompatible avec `track_buffer`** (compté en
  frames, pas en secondes — voir A.4) et avec `persist=True`. Sauter des
  frames change le pas de temps effectif du filtre de Kalman du tracker sans
  le lui dire.
- **Rejeter une frame floue avant l'inférence** prive le tracker d'une
  observation précisément pendant les occlusions dont les lots A/B/C cherchent
  à limiter les dégâts. Le contrôle de flou existant (post-détection, par crop,
  pour la galerie, dans `identity_manager.py`) reste le bon niveau de
  granularité — il ne prive jamais le tracker, il protège seulement l'écriture
  d'identité.

Ces trois éléments sont donc **retirés** du périmètre de ce lot. Ce qui suit
est ce qui reste, corrigé pour s'intégrer sans casser l'existant.

### 8.1 Lecture asynchrone du flux — à ne faire qu'après mesure du besoin réel

`model.track(source=source, stream=True)` utilise déjà en interne les lecteurs
threadés d'Ultralytics. **Avant d'écrire une seconde couche de lecture
asynchrone**, profiler avec l'outil de profilage déjà présent dans `main.py`
pour vérifier si le goulot d'étranglement vient réellement de la capture ou de
l'inférence (probable, sur CPU). Si le profilage confirme un vrai goulot de
capture (source RTSP avec latence réseau, par exemple), implémenter une file
bornée en politique **drop-oldest** — jamais un backlog non borné.

### 8.2 Normalisation lumineuse (CLAHE)

- **Toujours actif ou toujours inactif**, jamais conditionné à un seuil de
  contraste mesuré frame par frame — un déclenchement conditionnel introduit
  une instabilité du descripteur ReID (lot 3.a) que rien ne
  distingue d'un vrai changement d'identité.
- Appliqué sur le canal **L** de l'espace LAB (ou V en HSV), jamais sur les
  trois canaux BGR séparément — un CLAHE canal par canal en BGR déplace la
  teinte, précisément ce que le descripteur du lot 3.a exploite.
- Doit s'appliquer **avant** que la frame soit transmise à la fois au modèle
  YOLO et à l'extracteur d'apparence, pour que détection et descripteur voient
  la même image.
- Si activé, **répéter la mesure de distribution de similarité du lot 3.a**
  (les deux histogrammes intra/inter-identité) avec CLAHE actif, et vérifier
  si `similarity_threshold` a besoin d'être réajusté. Ne pas supposer que le
  seuil calibré sans CLAHE reste valable.

```yaml
preprocessing:
  clahe:
    enabled: false        # PROVISOIRE — off par défaut tant que non mesuré
    clip_limit: 2.0
    tile_grid_size: [8, 8]
```

### 8.3 Masque ROI

Le masque **ne doit pas changer les dimensions de l'image ni son origine de
coordonnées**. On masque (pixels hors polygone mis à zéro ou couleur neutre),
on ne recadre pas — `geometry.py` et `calibration.py` travaillent en
coordonnées normalisées à l'image complète, un recadrage invaliderait
silencieusement la ligne déjà validée par l'opérateur.

```yaml
preprocessing:
  roi_mask:
    enabled: false
    polygon_normalized: []   # [[x, y], ...] en coordonnées 0-1, même repère que la ligne
```

### 8.4 Ce qui manque souvent dans ce genre de demande

- **Mesure avant/après obligatoire**, même protocole que les lots précédents
  (vidéo à occlusion dense du corpus, mêmes compteurs, plus les métriques de
  vérité terrain du §3 de `CLAUDE.md`), et la vérification de la distribution
  de similarité ReID pour le CLAHE (§8.2).
- **Budget de calcul réel** : utiliser le profileur déjà injecté dans
  `OccupancyManager` pour instrumenter chaque étape ajoutée, pas un
  chronométrage séparé.
- **Tests de non-régression** : le masque ROI ne déplace pas les coordonnées
  de la ligne validée ; le CLAHE ne s'applique que sur le canal de luminance ;
  activer/désactiver chaque étape ne change pas le nombre de frames traitées.

### 8.5 Livrable du lot 7

Un paragraphe dans `docs/correctif_occlusion.md` documentant, pour chaque
étape activée : l'effet mesuré sur les compteurs d'événements existants, et
pour le CLAHE spécifiquement, l'effet sur la distribution de similarité ReID
et la décision prise sur `similarity_threshold`.

---
