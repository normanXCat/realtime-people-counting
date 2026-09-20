# Lot 2 — Rétention d'identité hors zone de franchissement

> Ne pas démarrer avant que le lot 1 soit mesuré et accepté. Ce lot suppose la
> base de temps corrigée (1.1) : sans elle, on ne sait pas si une identité a
> été perdue par expiration de piste ou par expiration de galerie.

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
| Dans la zone de franchissement | expiration temporelle actuelle (`grace_period_seconds` + `gallery_retention_seconds`) |

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

- **Taux de réidentification après perte** (§3) : c'est la métrique centrale de
  ce lot, à ventiler par durée d'absence (< 5 s, 5–15 s, > 15 s).
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
