# Lot 1 — Base de temps et seuil NMS

> Premier lot du plan. Prérequis : catalogue du corpus (§2 de `CLAUDE.md`),
> vérité terrain (§3), critères d'acceptation validés (§4), contrôle de
> déterminisme (§2.1). Lire `docs/diagnostic.md` avant.

**Pourquoi ce lot en premier.** Deux correctifs en amont de tout le reste :
l'un rétablit la cohérence temporelle sans laquelle aucune mesure de durée
d'occlusion n'est interprétable, l'autre empêche la destruction de détections
avant même que le tracker les voie. Aucun réglage en aval ne peut récupérer une
détection supprimée par la NMS ni une piste tuée trop tôt.

---

## 1.1 `track_buffer` exprimé en secondes

**Problème** (diagnostic §4.1.6) : `track_buffer: 150` est compté en **frames**.
Sur fichier à 30 ips, la piste meurt à 5 s ; sur caméra live à ~3,5 ips
traités, elle survit 43 s. La galerie ReID, elle, raisonne en secondes. Les deux
mécanismes ne couvrent pas la même durée selon la source, et les mesures faites
sur fichier ne transposent pas à la production.

**Ceci remplace A.4 de l'ancien lot A** (« documenter, pas corriger ») : on
corrige.

```yaml
tracker:
  track_buffer_seconds: 12.0   # nouveau — horizon de survie de piste, en secondes
  track_buffer: 150            # déduit au démarrage ; conservé pour compatibilité
  fps_warm_up_frames: 60       # frames de mesure du FPS avant conversion
```

**Implémentation :**

- Mesurer le FPS réel sur `fps_warm_up_frames` frames au démarrage, puis
  convertir **une seule fois** : `track_buffer = round(track_buffer_seconds × fps_mesuré)`.
- **Conversion unique, pas de réévaluation continue.** Sur caméra live le FPS
  dérive ; réévaluer en cours de session changerait l'horizon du tracker sans
  le prévenir, ce qui est exactement le défaut qu'on corrige.
- Journaliser en INFO au démarrage : FPS mesuré, valeur convertie en frames,
  horizon en secondes. Journaliser en WARNING si la valeur convertie sort de
  bornes raisonnables (< 15 ou > 1200 frames) — avertir, ne pas refuser.
- Publier la valeur résolue dans `config_resolved.yaml`.
- Si `track_buffer_seconds` est absent, retomber sur `track_buffer` en frames
  avec un WARNING explicite (compatibilité ascendante).

**Choix de la valeur.** 12 s couvre les occlusions jusqu'à 13,5 s mesurées en
session réelle… presque. Le diagnostic dit « jusqu'à 13,5 s » : retenir
**15,0 s**, marqué `PROVISOIRE`, et vérifier sur le corpus quelle est la plus
longue occlusion réellement observée avant de figer.

**Tests :** conversion correcte à 30 ips et à 3,5 ips ; repli sur la valeur en
frames quand la clé secondes est absente ; valeur publiée dans
`config_resolved.yaml` ; aucun appel de reconversion après le warm-up.

---

## 1.2 Seuil NMS (`iou`)

**Problème** : deux personnes assises côte à côte produisent deux détections
valides dont l'une est supprimée par la NMS avant d'atteindre le tracker. Aucun
réglage de `match_thresh` ne récupère une boîte qui n'existe plus. C'est la
cause la plus probable des `POSSIBLE_MULTI_PERSON_BOX` observées.

**Avant de changer quoi que ce soit** : relever la valeur **réellement** en
vigueur. 0,7 est le défaut Ultralytics, pas nécessairement le vôtre — chercher
`iou=` dans l'appel `model.track` de `src/main.py` et dans `config/pipeline.yaml`.
Consigner la valeur trouvée dans le livrable.

```yaml
model:
  nms_iou: 0.85   # était <valeur relevée> — PROVISOIRE
                  # raison : deux personnes assises côte à côte produisent des
                  # boîtes très recouvrantes, supprimées avant le tracker.
                  # compromis accepté : plus de doublons sur une même personne,
                  # filtrés en aval par les filtres géométriques et
                  # timing.confirmation_seconds.
```

**Mesurer 0,85 et 0,90 séparément.** Si 0,90 n'apporte rien de plus que 0,85
sur les fusions, garder 0,85 (moins de doublons).

**Risque à surveiller** : des doublons sur une même personne survivent la NMS
et peuvent être promus en identités distinctes → sur-comptage. À ce stade
`new_track_thresh` vaut encore 0,7, ce qui limite le risque : une détection
doublon a rarement une confiance ≥ 0,7. **L'interaction NMS × `new_track_thresh`
à 0,45 est le point sensible et se mesure au lot 4**, pas ici.

Surveiller aussi le FPS : plus de boîtes survivantes = plus de travail
d'association par frame.

---

## 1.3 Mesure du lot

Protocole standard (§7 de `CLAUDE.md`) : vidéo à occlusion dense en mesure
principale, vidéo à occlusion faible en non-régression, chaque changement
mesuré isolément puis combiné.

**Compteurs internes** : `POSSIBLE_MULTI_PERSON_BOX`,
`TRACK_ID_APPEARANCE_DISCONTINUITY`, `TECHNICAL_ID_CHANGED`, `REID_NEW`, nombre
d'identités distinctes créées, `[BILAN]`, FPS moyen.

**Métriques de vérité terrain** (§3) : `fusions_reelles`, `id_switches_reels`,
`fragments_par_personne`, `erreur_comptage_max`, plus **taux de
réidentification après perte** — c'est le chiffre que 1.1 doit déplacer.

**Mesure spécifique à 1.1** : durée de la plus longue occlusion du corpus, et
nombre de pistes détruites par expiration de `track_buffer` avant / après.

---

## 1.4 Ce qui doit rester vrai après ce lot

- `occupancy_operational` inchangé sur la vidéo de contrôle (aucun IN/OUT
  fantôme créé par des doublons NMS).
- Aucun test existant supprimé ou passé en `skip`.
- `track_buffer` en frames reste lisible dans `config_resolved.yaml` — les
  tests existants qui le vérifient doivent continuer à passer, éventuellement
  ajustés à la valeur déduite, avec la raison documentée.
