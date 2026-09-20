# Lot 3 — ReID profond léger et galerie multi-échantillons

> Ne pas démarrer avant que le lot 2 soit mesuré et accepté.
> **Ce lot contient deux étapes mesurées séparément : 3.a puis 3.b.** Ne pas
> les appliquer dans le même commit ni dans le même tableau de mesure — 3.a
> change le descripteur, 3.b change la structure de mémoire qui le stocke, et
> les confondre rend les deux inattribuables.

**Ceci remplace B.3 et B.4 de l'ancien lot B.** Le descripteur HSV par bandes
(ex-B.4 option 1) devient un **repli documenté**, à n'implémenter que si OSNet
s'avère impossible dans l'environnement (voir 3.a). Le gel de `_blend` pendant
une discontinuité (ex-B.3) devient sans objet : la galerie multi-échantillons
supprime le mélange, donc l'auto-empoisonnement qu'il fallait contenir.

---

## 3.a — Descripteur OSNet-x0.25 (ONNX / INT8)

**Problème** (diagnostic §4.1.4) : l'histogramme HSV global sur le crop entier
encode surtout la couleur des chaises, des tables et du mur. L'écart
inter-personnes est souvent inférieur au bruit du descripteur, ce qui rend
`similarity_threshold: 0.40` arbitraire.

La classe `DeepAppearanceExtractor` existe déjà dans `identity_manager.py` :
la câbler, ne pas en écrire une nouvelle.

```yaml
reid:
  external_reid:
    enabled: true
    backend: onnxruntime
    model_path: models/osnet_x0_25.onnx
    model_expected_sha256: null   # À RENSEIGNER après export
    quantization: int8
    input_size: [128, 256]        # (l, h) — format ReID standard
    fallback_to_histogram: true   # si onnxruntime ou le poids est absent
```

**À figer avant d'écrire du code :**

- **Provenance du poids** : ONNX pré-exporté, ou export depuis `torchreid`.
  Documenter la source et le SHA-256 dans `docs/` — même exigence que pour le
  modèle pose (lot 6).
- **Crop utilisé** : **corps entier**, inchangé. La détection reste une
  détection de personnes à ce stade ; OSNet est entraîné sur des crops corps
  entier (Market-1501 / MSMT17) et se dégrade fortement hors de cette
  distribution. Si la détection de têtes est un jour adoptée, ce point devra
  être rouvert (voir le journal des décisions dans `CLAUDE.md`).
- **Repli** : si `onnxruntime` est indisponible ou le poids absent, retomber
  sur `HistogramAppearanceExtractor` avec un WARNING au démarrage — **ne pas
  planter**, ne pas non plus continuer en silence.

**Recalibrage obligatoire de `similarity_threshold`.** L'échelle de similarité
cosinus d'un descripteur profond n'a aucun rapport avec celle d'un histogramme
HSV : la valeur 0,40 actuelle devient sans objet. Procédure :

1. Mesurer, sur la vidéo à occlusion dense, les deux distributions de
   similarité : **intra-identité** (même personne, frames différentes) et
   **inter-identités** (personnes différentes), en s'appuyant sur les
   étiquettes de la vérité terrain (§3).
2. Proposer un seuil juste au-dessus du p95 inter-identité, et vérifier le taux
   de recouvrement des deux histogrammes.
3. Si le seuil ne peut pas être calibré sur un corpus disjoint dans le temps
   imparti, l'appliquer en **`PROVISOIRE`** documenté (valeur précédente,
   justification, limite connue) — pas de blocage, voir règle §5 de
   `CLAUDE.md`.
4. Reporter les deux histogrammes dans le livrable.

**Coût CPU** : instrumenter via le profileur **déjà injecté dans
`OccupancyManager`**, pas par un chronométrage séparé. Rapporter le coût par
crop et le coût total par frame au nombre de personnes observé. Ordre de
grandeur attendu ~0,2 s par seconde de vidéo à 20 personnes et 3,5 ips — à
confirmer, pas à supposer. Si le coût dépasse le budget FPS du §4 de
`CLAUDE.md`, essayer OSNet-x0.5 → x0.25 non quantifié → repli histogramme, dans
cet ordre, et documenter le choix.

**Indexation** : similarité cosinus directe en NumPy. À 10–50 identités
simultanées le coût est négligeable (< 0,1 ms) ; FAISS n'est pertinent qu'à
partir de plusieurs milliers d'empreintes et n'est **pas** dans ce lot.

---

## 3.b — Galerie multi-échantillons (remplace l'EMA)

**Problème** (diagnostic §4.1.5) : `rec.feature` est un vecteur unique mis à
jour à chaque frame par `_blend` avec `feature_momentum = 0.85` codé en dur.
Après un swap non détecté, la référence contient déjà ~56 % de l'apparence de
l'autre personne au bout de 5 frames — la correction de swap devient aveugle
exactement quand elle servirait.

```yaml
reid:
  long_term:
    gallery:
      max_samples: 6              # 5 à 8
      matching_strategy: top2_mean   # max | top2_mean
      min_detection_confidence: 0.6
      require_untruncated_crop: true   # crop non coupé par un bord d'image
      reject_when_multi_person_box: true
```

**Règles d'insertion** — un échantillon n'entre dans la galerie que si :

- confiance de détection ≥ `min_detection_confidence` ;
- le crop n'est pas tronqué par un bord de l'image ;
- aucun `POSSIBLE_MULTI_PERSON_BOX` actif sur cette boîte ;
- le contrôle de netteté existant (déjà présent dans `identity_manager.py`)
  passe.

**Appariement** : similarité **maximale** ou **moyenne des 2 meilleures**
(`matching_strategy`). Aucun mélange de vecteurs, donc aucun auto-empoisonnement
possible par construction.

**Éviction** quand la galerie est pleine : remplacer l'échantillon le plus
ancien, sauf s'il est le seul de sa « région » de similarité — préférer garder
la diversité. Une stratégie simple (remplacer le plus proche du nouvel
échantillon, ce qui maximise la diversité conservée) suffit ; documenter celle
retenue.

**Suppressions associées :** `feature_momentum`, `_blend` sur le descripteur, et
le paramètre `reid.appearance_continuity.freeze_gallery_frames` prévu en ex-B.3
n'ont plus d'objet. Retirer le code mort plutôt que le laisser inerte, et le
signaler dans le livrable. Position, hauteur et `last_seen_s` continuent d'être
mis à jour normalement — seule la logique du descripteur change.

---

## 3.c Mesure du lot

Deux tableaux avant/après distincts, l'un pour 3.a, l'autre pour 3.b, chacun
avec :

- les compteurs internes habituels (`TRACK_ID_APPEARANCE_DISCONTINUITY`,
  `IDENTITY_SWAP_CORRECTED`, `TECHNICAL_ID_CHANGED`, `REID_NEW`, `[BILAN]`,
  FPS) ;
- les métriques de vérité terrain (§3), en particulier `id_switches_reels` et
  le taux de réidentification après perte ;
- pour 3.a uniquement : les deux histogrammes de similarité et la décision
  prise sur `similarity_threshold` ;
- le coût CPU mesuré par le profileur.

**Ce qui doit rester vrai** : `occupancy_operational` inchangé ; le repli
histogramme reste fonctionnel et testé (un test qui force `enabled: false` doit
passer) ; aucune valeur de seuil codée en dur.
