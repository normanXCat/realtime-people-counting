# Prompt pour Antigravity — Amélioration du système de comptage existant

## Contexte

J'ai un projet Python fonctionnel de comptage de personnes par caméra, structuré en trois fichiers :

- `detection.py` : point d'entrée principal, charge YOLO11 (`ultralytics`), lance `model.track()` avec `tracker="bytetrack.yaml"`, boucle sur les frames et appelle le tracker + le visualizer.
- `tracker.py` : classe `LineCrossingTracker`, qui suit les centroïdes par `track_id`, détecte le franchissement d'une ligne **horizontale** (définie par `line_pos_ratio`, une fraction de la hauteur), avec hystérésis, lissage et cooldown pour éviter les faux comptages.
- `visualizer.py` : classe `Visualizer`, dessine la ligne de franchissement, les trajectoires (trails), et le HUD (présents/entrées/sorties).

Le code actuel fonctionne mais présente trois limites à corriger, sans changer l'architecture générale ni casser les fonctionnalités existantes (hystérésis, cooldown, lissage, HUD, trails).

## Amélioration 1 — Calibration manuelle de la ligne par clic (2 points)

**Objectif** : remplacer `line_pos_ratio` (ligne horizontale fixe) par une ligne définie par **deux points cliqués manuellement** sur la première frame de la source vidéo, avant le démarrage du tracking.

### Comportement attendu
- Créer un nouveau module `calibration.py` avec une fonction `calibrate_line(source) -> tuple[tuple[float, float], tuple[float, float]]`.
- Lire uniquement la première frame de la source (webcam ou fichier).
- Afficher une fenêtre OpenCV avec un callback souris (`cv2.setMouseCallback`) :
  - Clic gauche : ajoute un point (maximum 2 points actifs à la fois).
  - Touche **`g`** : réinitialise les points cliqués (efface les points en cours, permet de recommencer la sélection).
  - Touche **`c`** : valide la calibration **uniquement si exactement 2 points ont été cliqués** ; sinon, ignorer l'appui (afficher un message indiquant qu'il manque un point).
  - Affichage en temps réel : cercles rouges sur les points cliqués, ligne provisoire entre les deux points dès que le second est posé, texte d'instructions à l'écran (« Cliquez 2 points | g: réinitialiser | c: valider »).
- Retourner les deux points en **coordonnées normalisées** (`x / largeur`, `y / hauteur`, entre 0.0 et 1.0), pour rester valides quelle que soit la résolution.
- Fermer la fenêtre de calibration une fois validé, avant de lancer la boucle principale de `detection.py`.

### Intégration dans `detection.py`
- Appeler `calibrate_line(source)` juste après la résolution de la source vidéo, avant l'initialisation du tracker.
- Remplacer l'argument `--line-pos` par deux arguments optionnels `--line-p1` et `--line-p2` (format `"x,y"` normalisé) permettant de **sauter la calibration manuelle** si des coordonnées sont déjà fournies en ligne de commande (utile pour des relances automatiques sans réafficher l'interface de clic).
- Passer les points obtenus (calibrés ou fournis en argument) au constructeur de `LineCrossingTracker`.

### Modifications dans `tracker.py`
- Remplacer `line_pos_ratio: float` par `line_p1: tuple[float, float]` et `line_p2: tuple[float, float]` (coordonnées normalisées) dans le constructeur de `LineCrossingTracker`.
- Généraliser `_zone_from_y` (actuellement basé sur une comparaison Y simple) en une méthode qui calcule le **côté de la ligne** via le signe du produit en croix (déterminant) entre le vecteur de la ligne et le vecteur (point − p1), au lieu d'une comparaison verticale. La bande d'hystérésis doit devenir une marge de distance perpendiculaire à la ligne, plutôt qu'une marge verticale.
- Adapter le lissage (`_smoothed_y`) pour lisser la **distance signée à la ligne** plutôt que la seule coordonnée Y, afin que la logique reste cohérente pour une ligne diagonale ou verticale.
- Garder toute la logique de cooldown, d'oubli des IDs perdus (`_cleanup_lost_tracks`) et de comptage entrées/sorties strictement identique — seule la géométrie de détection de franchissement change.

### Modifications dans `visualizer.py`
- Adapter `draw_crossing_line` pour dessiner un segment entre deux points arbitraires (`line_p1`, `line_p2` en pixels, dérivés des coordonnées normalisées et de la résolution de la frame courante), au lieu d'une ligne horizontale pleine largeur.

## Amélioration 2 — Tracking hybride : ByteTrack par défaut, réassociation par apparence (DeepSORT) en cas d'occlusion longue

**Contexte technique important** : `model.track()` d'Ultralytics gère ByteTrack en interne et attribue un **nouvel ID** à une personne si elle est perdue trop longtemps (occlusion prolongée), ce qui casse la continuité de la trajectoire dans `LineCrossingTracker` et peut fausser le comptage (double comptage ou trajectoire orpheline lors du franchissement).

### Solution attendue : couche de réassociation par apparence
- Créer un nouveau module `reid.py` avec une classe `ReIDGallery` :
  - Extrait un vecteur d'embedding d'apparence pour chaque `track_id` actif, à partir du crop de la bounding box (utiliser un extracteur léger, ex: `torchreid` OSNet si disponible, sinon un ResNet18 pré-entraîné tronqué avec pooling global comme solution de repli plus simple).
  - Maintient une « galerie » des IDs récemment perdus : quand un `track_id` disparaît des `active_ids` pendant plus de `occlusion_threshold` frames (mais moins de `gallery_ttl` frames), stocker son dernier embedding et son état de trajectoire (`track_states`, dernière position) dans la galerie.
  - Quand un **nouvel** `track_id` apparaît (jamais vu avant), calculer son embedding et le comparer par similarité cosinus à tous les embeddings de la galerie des IDs perdus.
  - Si la similarité dépasse `similarity_threshold` (ex: 0.65-0.7), considérer qu'il s'agit de la même personne réapparue : **remapper** ce nouvel ID vers l'ancien ID dans une table de correspondance (`id_remap: dict[int, int]`), et restaurer son état (`track_states`, position) dans `LineCrossingTracker` avant de poursuivre le traitement normal.
  - Si aucune correspondance n'est trouvée, traiter l'ID comme une nouvelle personne (comportement actuel inchangé).
  - Purger la galerie des entrées dépassant `gallery_ttl`.
- Exposer trois paramètres configurables (en argument CLI ou constante de config) : `occlusion_threshold` (frames avant qu'un ID soit considéré en occlusion longue, ex: 10), `gallery_ttl` (durée max de conservation en galerie, ex: 60 frames), `similarity_threshold` (seuil cosinus, ex: 0.65).
- Intégrer cette couche entre la sortie de `model.track()` et l'appel à `tracker.update()` dans `detection.py` : appliquer le remapping d'ID **avant** de transmettre les `track_ids` à `LineCrossingTracker`, de façon totalement transparente pour ce dernier.
- Important : ce mécanisme ne doit s'activer/calculer des embeddings que pour les IDs effectivement en occlusion prolongée ou nouvellement apparus, **pas à chaque frame pour tous les IDs actifs**, afin de préserver les performances temps réel.

## Amélioration 3 — Réduction des détections manquées

Le système rate parfois des personnes. Pistes à implémenter, par ordre de priorité :

- Ajouter un argument `--imgsz` (résolution d'inférence, défaut 640) exposé dans `parse_args()` et transmis à `model.track(imgsz=args.imgsz, ...)`. Une résolution plus élevée (960 ou 1280) améliore nettement la détection des personnes petites/éloignées, au prix d'un léger coût en vitesse.
- Ajouter un argument `--model-size` ou documenter le passage à `yolo11m.pt` (plus précis que `yolo11s.pt` actuellement utilisé par défaut) comme option pour les cas où la précision prime sur la vitesse.
- Exposer les seuils internes de `bytetrack.yaml` (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`) via un fichier de config YAML personnalisé copié dans le projet (`configs/bytetrack_custom.yaml`), pour permettre d'abaisser le seuil bas et mieux récupérer les détections faibles au lieu de les ignorer.
- Ajouter un prétraitement optionnel (flag `--enhance-contrast`) appliquant un CLAHE (`cv2.createCLAHE`) sur la frame avant inférence, utile pour les scènes en contre-jour ou faiblement éclairées.
- Documenter clairement dans le code (docstring) le compromis vitesse/précision de chaque option ajoutée, pour que l'utilisateur puisse ajuster selon son cas d'usage.

## Contraintes générales

- Ne pas casser les fonctionnalités existantes : hystérésis, lissage, cooldown, HUD, trails, mode `--no-show`, gestion `Ctrl+C`, téléchargement automatique du modèle.
- Conserver le style de code actuel (docstrings détaillées en français, typage Python, structure en classes/modules séparés).
- Toute nouvelle dépendance (torchreid, etc.) doit être ajoutée à un fichier `requirements.txt` si celui-ci n'existe pas encore, en le créant.
- Fournir des valeurs par défaut sensées pour tous les nouveaux paramètres, afin que le système reste utilisable sans configuration supplémentaire pour un usage basique.

## Fichiers à créer ou modifier

```
calibration.py     # NOUVEAU — interface de clic (2 points), touches g/c
reid.py             # NOUVEAU — galerie de réassociation par apparence
tracker.py          # MODIFIÉ — ligne générale (2 points), zone via déterminant
visualizer.py       # MODIFIÉ — dessin de ligne arbitraire
detection.py        # MODIFIÉ — appel calibration, intégration reid, nouveaux args CLI
configs/bytetrack_custom.yaml  # NOUVEAU — seuils ByteTrack ajustables
requirements.txt    # MODIFIÉ/CRÉÉ — nouvelles dépendances
```

## Critères d'acceptation

- Au lancement, une fenêtre de calibration s'ouvre sur la première frame ; cliquer 2 points, appuyer sur `c` valide et lance le tracking avec cette ligne exacte ; appuyer sur `g` avant `c` réinitialise la sélection.
- La ligne dessinée pendant le tracking correspond exactement aux deux points cliqués (diagonale, verticale ou horizontale selon le choix de l'utilisateur).
- Lors d'une occlusion simulée de longue durée (ex: une personne cachée manuellement pendant plusieurs secondes derrière un obstacle), l'ID est correctement restauré à la réapparition au lieu de générer un nouveau comptage erroné.
- Le taux de détections manquées diminue observablement sur une vidéo de test avec des personnes de petite taille ou éloignées, après activation de `--imgsz 1280` ou du modèle `yolo11m.pt`.
- Les performances temps réel restent acceptables (pas de chute drastique de FPS) grâce à l'activation sélective de la réassociation par apparence uniquement sur les IDs en occlusion prolongée.
