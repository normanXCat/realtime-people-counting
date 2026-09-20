# Lot 2 — Rétention d'identité hors zone de franchissement

> **Ordre révisé : ce lot vient après le lot 4 (seuils BoT-SORT), pas avant.**
> Les mesures du lot 0 montrent que le décrochage se produit au niveau des
> pistes ; les seuils du tracker sont plus déterminants et bien moins coûteux.
> Ce lot suppose donc les lots 1 **et 4** mesurés et acceptés. La base de temps
> corrigée (1.0 et 1.1) reste un prérequis strict : sans elle, on ne sait pas
> si une identité a été perdue par expiration de piste ou de galerie.

**Ceci remplace A.2 de l'ancien lot A** (`gallery_retention_seconds: 12.0`
appliqué uniformément). Allonger un délai ne fait que repousser le problème ;
la règle métier le supprime.

---

## 2.1 La règle

**Une personne ne quitte pas la salle sans franchir la ligne.**
`occupancy_operational` applique déjà cette logique — il ne décrémente que sur
sortie confirmée par la FSM. La galerie ReID la contredit en purgeant par
expiration temporelle : une personne occultée trop longtemps au milieu de la
salle perd son empreinte et revient comme nouvelle identité.

Nouvelle politique, selon la position de l'ancre **à la dernière frame
observée** :

| Position à la perte | Politique de rétention |
|---|---|
| Hors zone de franchissement | **illimitée sur la durée de session** |
| **Sur un bord du cadre** | **illimitée**, quelle que soit la distance à la ligne |
| Dans la zone de franchissement | expiration temporelle actuelle (`grace_period_seconds` + `gallery_retention_seconds`) |

**Le bord du cadre est un troisième lieu de disparition**, et la vérité terrain
en fournit le cas : `P9` sort du champ de la caméra à la seconde 25 de
`fort_occ4` et **reste dans la salle** jusqu'à la fin
(`tests/fixtures/ground_truth_fort_occ4.json`, occlusion `hors_champ_sans_retour`).
Si le bord du cadre coïncide avec la zone de la porte selon la géométrie de la
salle, la règle de distance seule traiterait `P9` comme une sortie probable
alors qu'elle est toujours présente. D'où une règle explicite plutôt qu'une
conséquence de la géométrie.

Attention à ce que ce cas **ne** permet **pas** de vérifier : `P9` ne revient
jamais dans le champ, donc il ne teste pas la redétection. Ce qu'il teste :
`occupancy_operational` doit rester à 13, `occupancy_observed` descendre
légitimement, et l'écart apparaître en incertitude.

La zone se définit à partir de la distance signée à la ligne, déjà calculée par
`geometry.py` — pas d'une nouvelle géométrie.

```yaml
reid:
  retention:
    exit_zone_half_width: 0.12    # distance normalisée à la ligne ; |d| <= seuil ⇒ "dans la zone"
    unlimited_outside_zone: true
    max_gallery_identities: 200   # plafond de sécurité, éviction des plus anciennes HORS zone
    similarity_penalty_per_minute: 0.0   # PROVISOIRE — exigence accrue avec la durée d'absence ; 0 = inactif
  long_term:
    gallery_retention_seconds: 12.0   # ne s'applique désormais QUE dans la zone de franchissement
```

`exit_zone_half_width` est à calibrer sur le corpus : assez large pour couvrir
la zone où une personne peut disparaître en sortant réellement, assez étroite
pour ne pas avaler la moitié de la salle. Valeur de départ `PROVISOIRE`.

---

## 2.2 Cas à trancher explicitement

**Identité perdue *dans* la zone, sans `CROSSING_OUT`.** Elle expire
normalement (la personne est probablement sortie par la porte sans que la FSM
confirme). Mais cette expiration doit émettre un événement dédié — proposition
`IDENTITY_EXPIRED_IN_EXIT_ZONE` — et alimenter `occupancy_uncertain` plutôt que
de disparaître en silence. À confirmer par la personne responsable avant
implémentation.

**Fin de session.** « Illimitée » signifie *sur la durée de la session*. Le
comportement au redémarrage est inchangé : pas de persistance disque.

**Warm-up initial.** Les identités créées au démarrage (personnes déjà
présentes) sont par construction hors zone : elles bénéficient de la rétention
illimitée. C'est le comportement voulu.

---

## 2.3 Garde-fous

Une galerie qui ne purge jamais voit sa probabilité de faux appariement croître
avec sa taille. En salle de classe la population est bornée (20–50 personnes,
session finie) et le risque est faible, mais les garde-fous s'implémentent
maintenant, pas plus tard :

1. **Plafond `max_gallery_identities`** avec éviction des plus anciennes
   entrées hors zone quand il est atteint, et un événement journalisé à chaque
   éviction (une éviction fréquente est le signe que le plafond est mal réglé).
2. **Seuil de similarité asymétrique** : plus l'absence est longue, plus
   l'exigence de correspondance est stricte —
   `seuil_effectif = similarity_threshold + similarity_penalty_per_minute × minutes_absence`,
   borné. Laissé à **0,0 (inactif) par défaut** : à activer seulement si la
   mesure montre des faux appariements sur les absences longues. Le paramètre
   existe pour que ce soit un réglage et non une réécriture.

---

## 2.4 Mesure du lot

- **Taux de réidentification après perte** (§3) : métrique centrale de ce lot,
  à ventiler par durée d'absence (< 5 s, 5–15 s, > 15 s).
  **Portée à mentionner dans le livrable** : le corpus ne contient qu'**un
  seul** intervalle long avec retour en vue — `P7`, s19 à s25, sept secondes.
  `P8` (3 s) et `P3` (2 s) sont trop courts ; `P9` et `P12` ne reviennent pas
  avant la fin. Un résultat fondé sur un seul cas se rapporte comme tel.
- Nombre d'identités créées (`REID_NEW`) : doit **baisser** si des personnes
  auparavant perdues sont maintenant retrouvées.
- `id_switches_reels` et `fragments_par_personne` : doivent baisser.
- **Faux appariements** : à compter manuellement sur le segment annoté — une
  identité rendue à la mauvaise personne est pire qu'une identité perdue.
  C'est le risque propre à ce lot, il doit apparaître dans le livrable même
  s'il vaut zéro.
- Taille maximale de la galerie en fin de session, nombre d'évictions.

---

## 2.5 Ce qui doit rester vrai

- `occupancy_operational` strictement inchangé : la rétention ne crée ni IN ni
  OUT.
- Une personne réellement sortie par la porte ne reste pas comptée : vérifier
  sur un segment annoté contenant une sortie confirmée.
- La zone de franchissement est dérivée de la ligne **déjà validée par
  l'opérateur** (`calibration.py`) — aucun nouveau référentiel de coordonnées.
