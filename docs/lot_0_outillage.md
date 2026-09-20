# Lot 0 — Outillage de mesure

> Préalable à tous les autres lots. Écrit après la session 0, qui a montré que
> le pipeline n'a aucun chemin non interactif : `perform_line_selection`
> (`src/main.py:356`) exige deux clics plus « C », et `--no-show` ne supprime
> que la prévisualisation.
>
> **Ce lot ne corrige aucun symptôme.** Il rend les mesures des lots 1 à 7
> possibles, reproductibles et représentatives de la production. Sans lui,
> chaque lot se mesure sur un harnais qui n'est pas le système livré.

---

## 0.1 Drapeau `--line` explicite

**Problème** : la calibration par clics rend impossible le catalogage, le
rejouage d'une baseline et toute mesure avant/après automatisée. La session 0 a
dû contourner par un harnais rejouant `OccupancyManager.process_frame` avec une
ligne de convention — ce qui a écarté du classement tous les compteurs
dépendant de la ligne (`IN`, `OUT`, `NEW`, `INCONSISTENT_STATE`).

**Ce qui ne change pas** : le repli automatique « ligne horizontale à 60 % » a
été délibérément supprimé du dépôt et **ne revient pas**. Une ligne implicite
est une source d'erreur silencieuse.

```
--line x1,y1,x2,y2     # coordonnées normalisées [0,1], séparées par des virgules
```

- Aucune valeur par défaut. Le drapeau est absent ⇒ comportement interactif
  actuel, strictement inchangé.
- `--no-show` **sans** `--line` ⇒ échec immédiat, code de sortie dédié, message
  explicite (« mode non interactif : --line est obligatoire »). Pas de repli,
  pas de devinette.
- La ligne fournie passe par la même validation que la ligne cliquée (mêmes
  contrôles dans `calibration.py`) — aucun chemin de validation parallèle.
- La ligne effectivement utilisée est journalisée au démarrage et publiée dans
  `config_resolved.yaml`, qu'elle vienne des clics ou du drapeau.

**Tests** : analyse syntaxique de `--line` (format valide, malformé, hors
bornes) ; refus en mode non interactif sans ligne ; identité du résultat entre
une ligne cliquée et la même ligne passée en drapeau.

---

## 0.2 Harnais de mesure versionné

**Le harnais doit appeler `src/main.py`, pas réimplémenter le pipeline.** C'est
le point central de ce lot. Un harnais qui rejoue `process_frame` de son côté
est un second système, qui diverge du premier sans prévenir — le contrôle de
déterminisme de la session 0 l'a montré de la pire manière (13 purges contre 0
selon la base de temps employée).

Une fois `--line` disponible, le harnais devient une enveloppe mince :

```
scripts/measure_corpus.py
  --videos <dossier>          # défaut : test/
  --line x1,y1,x2,y2          # ligne de convention, identique pour tout le corpus
  --repeat N                  # rejouages pour le contrôle de bruit
  --out <fichier.jsonl>
```

Il lance `src/main.py` par sous-processus, une entrée JSONL par vidéo et par
rejouage. Schéma **stable** — les lots 1 à 7 comparent leurs tableaux
avant/après sur ces champs, un changement de schéma en cours de plan rend les
comparaisons incohérentes :

- identifiants : `source`, `run_index`, `git_sha`, `timestamp`
- exécution : `frames_processed`, `processing_fps`, `time_base` (`video` ou
  `wall`)
- compteurs : le dictionnaire complet des événements, tel qu'émis
- identité : `technical_ids`, `persons_created`, `technical_id_changes`,
  `appearance_discontinuities`, `long_gaps_count`
- occupation : `occupancy_operational`, `occupancy_observed`, `occupancy_range`

Les scripts existants (`measure_tracker_swaps.py`, `diagnose_occlusion.py`)
restent en place ; ne pas les réécrire, ne pas les fusionner.

---

## 0.3 Vérification du lot

**Critère d'acceptation, unique et net** : rejouer `fort_occ4` avec le harnais
versionné, en base de temps vidéo, et **retrouver exactement** les compteurs
consignés en session 0 (`docs/correctif_occlusion.md` §11). Le bruit run-à-run
y était nul ; tout écart signale une divergence entre le harnais de session 0
et le chemin `main.py`, et doit être expliqué avant d'aller plus loin.

**À refaire après le lot 1** : celui-ci change la base de temps. Rejouer alors
le corpus complet pour établir la baseline définitive — celle sur laquelle tous
les lots suivants se compareront.

---

## 0.4 Ce qui doit rester vrai

- Le mode interactif est inchangé : mêmes clics, même touche, même validation.
- Aucun repli automatique de ligne, sous aucune condition.
- Aucune modification de `OccupancyManager`, `IdentityManager` ou du tracker
  dans ce lot. Il ne touche qu'au point d'entrée, à l'analyse des arguments et
  à `scripts/`.
