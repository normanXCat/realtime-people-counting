# Protocole d'évaluation et de calibration (section 9)

Ce document décrit *comment* les chiffres du mémoire sont produits. Il doit être
figé **avant** les mesures finales : des seuils choisis après avoir vu les
résultats ne démontrent rien.

## 1. Séparation stricte calibration / test (9.1)

Deux ensembles de vidéos **disjoints** sont déclarés dans `config/protocol.yaml`
(`datasets.calibration`, `datasets.test`). `scripts/run_protocol.py` vérifie la
disjonction et refuse de tourner si une vidéo apparaît dans les deux.

Seuils choisis **uniquement** sur l'ensemble de calibration :

| Paramètre de configuration | Rôle |
|---|---|
| `model.confidence`, `model.nms_iou`, `model.image_size` | détection |
| `geometry.dead_zone_ratio`, `timing.confirmation_seconds`, `timing.grace_period_seconds` | franchissement |
| `reid.long_term.similarity_threshold`, `safety_margin`, `v_max_ratio`, `spatial_margin_ratio` | réassociation long terme |

L'ensemble de test n'est exécuté **qu'une fois**, pour le rapport final. Toute
valeur encore marquée « PROVISOIRE » dans `config/pipeline.yaml` doit avoir été
justifiée par une courbe de calibration (ou rester explicitement annoncée comme
valeur provisoire non calibrée).

## 2. Vérité terrain à deux niveaux (9.2)

**Niveau événements** — sur l'ensemble complet des vidéos, format JSON :

```json
{
  "schema_version": 1,
  "video": "clip01.mp4",
  "fps": 30.0,
  "crossings": [
    {"timestamp_s": 3.24, "direction": "in"},
    {"timestamp_s": 18.90, "direction": "out"}
  ],
  "occupancy": [
    {"start_s": 0.0,  "end_s": 3.2,  "count": 0},
    {"start_s": 3.2,  "end_s": 19.0, "count": 1},
    {"start_s": 19.0, "end_s": 32.0, "count": 0}
  ]
}
```

- `crossings` : chaque franchissement réel, horodaté à la seconde près, avec sa
  direction ;
- `occupancy` : occupation vraie par intervalles (un relevé par changement) ;
- le niveau occupation est optionnel (le harnais le signale) mais requis pour
  publier l'erreur absolue moyenne.

**Niveau MOT dense** — sur 2 à 3 clips courts seulement : boîtes annotées frame
par frame (format MOTChallenge), évaluées avec `py-motmetrics` pour l'IDF1 et le
nombre d'ID switches. Ce niveau n'est pas calculé par `evaluate_system.py`.

## 3. Critère d'acceptation (9.3)

Déclaré dans `config/protocol.yaml` (`acceptance`), par défaut :

- F1 des franchissements ≥ **0.90** ;
- erreur absolue moyenne d'occupation ≤ **1.0** personne ;
- FPS moyen ≥ seuil à fixer sur le matériel de référence documenté
  (`acceptance.min_fps`, volontairement `null` tant que le matériel n'est pas
  arrêté).

Le harnais d'évaluation **refuse de conclure** sans critère : il compare les
mesures aux seuils passés en argument et rend un verdict explicite
(`acceptance.accepted`). Un run non conforme est signalé, jamais arrondi.

## 4. Baselines et ablations (9.4)

Déclarées dans `config/protocol.yaml`, exécutables sans intervention manuelle :

| Variante | Facteur étudié |
|---|---|
| `baseline_1` | YOLO11n + BoT-SORT, sans ReID long terme |
| `baseline_2` | YOLO11s + BoT-SORT, sans ReID long terme (coût/précision) |
| `variant_1` | configuration de référence : YOLO11n + BoT-SORT + ReID long terme |
| `variant_2` | variante 1 avec `safety_margin = 0` (apport de la marge de sécurité) |
| `variant_3` | variante 1 sans contrainte spatio-temporelle |
| `variant_4` | variante 1 + stabilisation (verrouillage de hauteur, ancre) |
| `variant_5` | variante 1 + ReID externe (comparaison d'extracteur d'apparence) |
| `variant_6` | YOLO11s + ReID long terme (justification du choix du modèle) |

Chaque variante diffère de la référence par **un seul** facteur : les surcharges
sont déclarées dans le protocole, revalidées par `src/config.py` et écrites dans
`results/protocol/<variante>/config.yaml` pour que chaque run soit rejouable.

## 5. Exécution

**Prérequis d'interaction.** La ligne virtuelle est obligatoirement définie à la
main : chaque session du protocole ouvre la première image et attend deux clics
validés par « C ». Les runs ne peuvent donc pas être entièrement automatisés ;
un opérateur doit confirmer la ligne, sur une configuration identique pour toutes
les vidéos d'un même ensemble (même caméra, même cadrage). Pour chaque session,
`calibration.json` et l'événement `LINE_VALIDATED` documentent la ligne retenue.

```bash
# 1. Sanity check sans lancer le pipeline
python scripts/run_protocol.py --dataset calibration --dry-run

# 2. Réglages sur l'ensemble de calibration
python scripts/run_protocol.py --dataset calibration

# 3. Rapport final : une seule exécution de l'ensemble de test
python scripts/run_protocol.py --dataset test --variant baseline_1 --variant baseline_2 --variant variant_1

# 4. Évaluation d'une session isolée (F1, occupation, FPS)
python evaluate_system.py \
    --events results/protocol/variant_1/variant_1__clip01/events.jsonl \
    --ground-truth data/gt/test/clip01.json \
    --min-f1 0.90 --max-occupancy-mae 1.0
```

Sorties : `results/protocol/<variante>/report.json` (détail + verdict),
`results/protocol/summary_<ensemble>.md` (tableau comparatif), et par session
`events.jsonl`, `summary.json`, `environment.json`, `calibration.json`,
`config_resolved.yaml` (qui fige l'orientation intérieur/extérieur validée).

## 6. Traçabilité exigée par le rapport

Pour chaque run publié dans le mémoire : identifiant de session, empreinte
SHA-256 des poids, version des dépendances (`environment.json`), configuration
résolue, débit et latence par étape (moyenne, médiane, minimum, P95). Sans ces
éléments, le résultat n'est pas reproductible et ne doit pas être publié.

## 7. Ce qui reste à faire côté mesures

Le dépôt fournit le dispositif complet, mais aucune campagne n'est lancée ici :
les listes `datasets.*` de `config/protocol.yaml` sont vides tant que les vidéos
annotées n'existent pas. Les sections « résultats » du mémoire ne peuvent être
remplies qu'après acquisition des vidéos, annotation de la vérité terrain et
exécution des étapes 2 à 3 du §5.
