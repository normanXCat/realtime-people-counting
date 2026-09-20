# Lot 1 — Base de temps et seuil NMS

> Prérequis : **lot 0 accepté** (sans harnais versionné, ce lot n'est pas
> mesurable), catalogue du corpus (§2 de `CLAUDE.md`), vérité terrain (§3),
> critères d'acceptation validés (§4), contrôle de déterminisme (§2.1). Lire
> `docs/diagnostic.md` avant.
>
> **Élargi après la session 0.** Deux mesures ont changé le contenu de ce lot :
> l'écart entre base de temps vidéo et base murale (13 purges contre 0 sur la
> même vidéo) et les 235 rejets `aspect_ratio_out_of_range` sur `fort_occ`.
> Voir `docs/correctif_occlusion.md` §11.7 et §11.9.

**Pourquoi ce lot en premier.** Deux correctifs en amont de tout le reste :
l'un rétablit la cohérence temporelle sans laquelle aucune mesure de durée
d'occlusion n'est interprétable, l'autre empêche la destruction de détections
avant même que le tracker les voie. Aucun réglage en aval ne peut récupérer une
détection supprimée par la NMS ni une piste tuée trop tôt.

---

## 1.0 Base de temps selon la source

**Mesure de la session 0** : sur `fort_occ4`, en base de temps murale (celle
qu'emploie réellement `src/main.py`), `PURGE` tombe de 13 à **0**. Le mécanisme
est identifié : `grace_period_frames` est compté en frames (150) tandis que la
fenêtre de galerie est en secondes sur `timestamp_s`
(`identity_manager.py:1175-1179`) ; à ~4 ips de traitement, 17 s murales valent
≈ 2,4 s de vidéo, donc la galerie libère l'identité avant que la FSM puisse la
purger — `release_expired` retire la piste de `self.tracks` et `_handle_missing`
n'a plus rien à purger (`occupancy_manager.py:631-633`). Le commentaire « la
purge a déjà été journalisée » y est faux dans ce régime.

Second effet mesuré : le bruit run-à-run est **nul** en base vidéo (3 exécutions
concordantes) et **non nul** en base murale (±1 sur l'identité, ±37 sur
`INCONSISTENT_STATE`), parce que le FPS de traitement varie avec la charge
machine et que la base murale propage cette variation dans toutes les durées.

**Correctif** : l'horloge du pipeline suit la source.

```yaml
timing:
  time_base: auto     # auto | video | wall
                      # auto ⇒ horodatage vidéo si la source est un fichier,
                      #        temps mural si la source est une caméra
```

- Une seule horloge par exécution, consultée partout (FSM, galerie, snapshots) —
  pas de mélange frames/secondes/temps mural selon le module.
- La base retenue est journalisée au démarrage et publiée dans
  `config_resolved.yaml` ; elle figure dans chaque entrée du harnais (champ
  `time_base`).
- `video` et `wall` restent forçables explicitement, pour comparer les deux
  régimes lors d'une investigation.

**Pourquoi c'est dans ce lot** : sans cela, les lots 2 à 7 se mesureraient sur
fichier dans un régime qui n'existe pas en production, et aucun de leurs
résultats ne transposerait à la caméra.

**Tests** : sélection correcte de la base selon le type de source ; cohérence
grâce FSM / fenêtre galerie dans les deux bases ; sur `fort_occ4` en base
vidéo, `PURGE` retrouve 13.

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

## 1.3 Ordre du filtre géométrique

**Mesure de la session 0** : sur `fort_occ`, les 235 `DETECTION_REJECTED_GEOMETRY`
sont **tous** `aspect_ratio_out_of_range`, tous des boîtes trop larges (ratio
W/H jusqu'à 3,85). Or `check_detection_geometry` s'exécute **avant** la
détection de fusion (`occupancy_manager.py:365-399`). Une boîte large est
exactement la signature de deux personnes côte à côte : le filtre élimine donc
le cas que `POSSIBLE_MULTI_PERSON_BOX` doit attraper, avant qu'il puisse être
émis.

Conséquence directe sur le catalogue : les « 0 fusions » relevées en session 0
sont, au moins en partie, un artefact de mesure. Les vidéos les plus peuplées du
corpus (`fort_occ`, `fort_occ2`) sont précisément celles qui ne produisent aucun
phénomène — à revérifier après ce correctif.

**Correctif** : une boîte rejetée pour ratio W/H **trop grand** est routée vers
le chemin fusion avant d'être écartée, au lieu d'être éliminée en silence. Les
autres motifs de rejet (`box_too_small`, ratio trop petit) sont inchangés.

- Ne pas relâcher `max_aspect_ratio_wh` : le filtre garde son rôle, on corrige
  l'ordre des opérations, pas le seuil.
- Chaque rejet pour ratio trop grand émet son motif et ses dimensions, pour que
  la mesure reste vérifiable.

**Tests** : une boîte de ratio 3,5 contenant deux personnes déclenche
`POSSIBLE_MULTI_PERSON_BOX` au lieu d'un rejet silencieux ; une boîte trop
petite reste rejetée comme avant.

**À refaire après ce correctif** : rejouer le catalogue complet. Le classement
des vidéos du corpus peut changer.

---

## 1.4 Mesure du lot

Protocole standard (§7 de `CLAUDE.md`) : vidéo à occlusion dense en mesure
principale, vidéo à occlusion faible en non-régression, chaque changement
mesuré isolément puis combiné.

**Compteurs internes** : `POSSIBLE_MULTI_PERSON_BOX`,
`TRACK_ID_APPEARANCE_DISCONTINUITY`, `TECHNICAL_ID_CHANGED`, `REID_NEW`, nombre
d'identités distinctes créées, `[BILAN]`, FPS moyen.

**Métriques de vérité terrain** (§3) : `fusions_reelles`, `id_switches_reels`,
`fragments_par_personne`, `erreur_comptage_max`, plus **taux de
réidentification après perte** — c'est le chiffre que 1.1 doit déplacer.

**Mesures spécifiques** — 1.0 : `PURGE`, `INCONSISTENT_STATE` et
`OCCUPANCY_SNAPSHOT` dans les deux bases de temps, plus le bruit run-à-run en
base retenue. 1.1 : durée de la plus longue occlusion du corpus et nombre de
pistes détruites par expiration de `track_buffer`. 1.3 :
`DETECTION_REJECTED_GEOMETRY` ventilé par motif et `POSSIBLE_MULTI_PERSON_BOX`,
sur `fort_occ` et `fort_occ2` en particulier.

---

## 1.5 Ce qui doit rester vrai après ce lot

- `occupancy_operational` inchangé sur la vidéo de contrôle (aucun IN/OUT
  fantôme créé par des doublons NMS).
- Aucun test existant supprimé ou passé en `skip`.
- `track_buffer` en frames reste lisible dans `config_resolved.yaml` — les
  tests existants qui le vérifient doivent continuer à passer, éventuellement
  ajustés à la valeur déduite, avec la raison documentée.
