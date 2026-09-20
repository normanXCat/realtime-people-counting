# Lot 5 — Retouches locales

> Ex-« lot B ». Ne pas démarrer avant que le lot 4 soit mesuré et accepté.
> Lire d'abord `CLAUDE.md` et `docs/diagnostic.md`.
>
> **Amputé de deux sections lors de la révision du plan :** B.3 (gel du
> descripteur) et B.4 (descripteur HSV par bandes) sont remplacées par le
> lot 3. Les intitulés B.1 à B.6 sont conservés pour la traçabilité.

### B.1 Personne assise (`occupancy_manager.py`, ~ligne 800)

Supprimer le terme `not self.config.presence.head_assist.enabled` de la
condition qui gouverne la réadaptation de l'historique de hauteur. Les deux
mécanismes (réadaptation de hauteur et maintien par la tête) doivent coexister.
Ajouter un test qui vérifie qu'une personne assise stabilisée redevient
fiable, `head_assist` activé et désactivé.

### B.2 Fraîcheur du point tête (`_handle_missing`)

N'autoriser le maintien par la tête dans `_handle_missing` que si le point
tête a été observé sur la frame courante ou la précédente. Ajouter
`head_point_frame_index` sur la piste, renseigné en même temps que
`current_head_point`, et exiger `frame_index - head_point_frame_index <= 1`.
Ajouter `presence.head_assist.max_head_staleness_frames` (défaut 1) plutôt
qu'une constante codée en dur. Tests : maintien accepté quand frais, refusé
quand périmé de 2 frames ou plus.

### B.3 et B.4 — SUPPRIMÉS, voir `docs/lot_3_reid_osnet.md`

- **B.3** (exposer `feature_momentum`, geler `_blend` pendant une
  discontinuité) : sans objet. La galerie multi-échantillons du lot 3.b
  supprime le mélange de vecteurs, donc l'auto-empoisonnement qu'il fallait
  contenir. `feature_momentum` et `freeze_gallery_frames` disparaissent.
- **B.4** (histogramme HSV par bandes avec érosion du crop) : remplacé par
  OSNet-x0.25 au lot 3.a. Le HSV par bandes reste documenté comme **repli**, à
  n'implémenter que si OSNet s'avère impossible dans l'environnement — dans ce
  cas seulement, reprendre la spécification dans l'historique git de ce
  fichier.

### B.5 Boîte multi-personnes

1. Quand `POSSIBLE_MULTI_PERSON_BOX` est actif dans le voisinage d'une
   identité perdue, ne pas la purger silencieusement : la compter en borne
   haute de `[observée, opérationnelle]`, état publié comme incertain
   (réutiliser `occlusion_ambiguity_ratio`, `occluded_counted_neighbours`).
2. Ajouter un second signal indépendant de l'historique de largeur : nombre de
   têtes détectées dans la boîte (disponible une fois le modèle pose actif,
   lot C). Deux jeux de keypoints de tête confiants séparés de plus de
   `0.4 × largeur_boîte` dans une seule boîte ⇒ fusion, dès la première frame.
   Ce signal reste inerte tant que `head_assist` n'est pas activé (lot 6) —
   c'est normal, l'implémenter ici quand même.

### B.6 Vérification du lot 5

Même protocole qu'en A.5, plus : `POSSIBLE_MULTI_PERSON_BOX`,
`PRESENCE_MAINTAINED_BY_HEAD`, `PRESENCE_EXTENSION_EXPIRED`. La distribution
des similarités intra/inter a été mesurée au lot 3.a ; la reprendre ici
seulement si B.5 modifie les conditions d'écriture en galerie.

**S'arrêter et rendre ce tableau avant de continuer vers le lot 6.**

---
