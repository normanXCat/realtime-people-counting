# ENSEMBLE DE PROMPTS ARCHITECTURE : GESTIONNAIRE D'OCCUPATION ROBUSTE (YOLO11 + BOT-SORT)

---

## PROMPT 1 : MACHINE À ÉTATS, HYSTÉRÉSIS & BUFFER TEMPOREL (MONO-LIGNE)

# MISSION DE REFACTORISATION AVANCÉE : GESTIONNAIRE D'OCCUPATION ROBUSTE (STATE MACHINE & HYSTÉRÉSIS)

Tu es un expert en Vision par Ordinateur (YOLO11, BoT-SORT, ReID OSNet) et Architecte Logiciel Python.
Ta mission est de refactoriser le pipeline de traitement existant pour implémenter un **système de gestion de l'effectif d'une salle par machine à états finis (FSM)**.

Le système doit traiter la vidéo comme une **source d'information incomplète** et appliquer des règles strictes de tolérance aux erreurs d'occlusion, aux faux positifs, aux hésitations sur le seuil et aux apparitions soudaines.

---

## 1. RÈGLES ET PRINCIPES INCONTOURNABLES (NON-NÉGOCIABLES)

1. **Règle d'or de décrémentation :** La perte d'un tracker BoT-SORT (disparition de détection) ne doit **JAMAIS** diminuer l'effectif (`occupancy_count`).
2. **Dissociation stricte :** `visible_count` (détections instantanées YOLO) ≠ `occupancy_count` (estimation logique de l'effectif).
3. **Même personne déjà sortie :** Si une personne identifiée franchit la ligne `OUT`, l'effectif est décrémenté une seule fois. Sa détection continue du côté extérieur ne déclenche AUCUN nouveau comptage.
4. **Anti-rebond (Hystérésis) :** Un franchissement n'est confirmé que lorsque le centre de masse sort d'une **Zone Morte (Buffer)** entourant la ligne virtuelle.
5. **Phase de Warm-up :** Les événements `IN`/`OUT` sont verrouillés durant la phase d'initialisation automatique.

---

## 2. MODULE DE STRUCTURES DE DONNÉES (`occupancy_types.py`)

Crée ou mets à jour les structures de données suivantes :

```python
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import numpy as np


class TrackState(Enum):
    PRESENTE = auto()           # Connue dans la salle
    OCCULTEE = auto()           # Perdue temporairement (ex: sous une table)
    EN_ZONE_LIGNE = auto()      # Dans la zone morte autour de la porte
    SORTIE_CONFIRMEE = auto()   # A franchi OUT et s'est éloignée
    A_RETOURNE = auto()         # Personne sortie qui réentre dans la salle
    NOUVELLE_PRESENCE = auto()  # Apparue dans la salle sans franchissement observé


class OriginType(Enum):
    INITIALISATION = auto()
    ENTREE_LIGNE = auto()
    APPARITION_INTERIEURE = auto()


@dataclass
class LogicalTrack:
    logical_id: int
    technical_track_id: int
    state: TrackState
    origin: OriginType
    last_position: tuple[int, int]                               # (x, y)
    last_seen_frame: int
    feature_vector: np.ndarray | None = None                     # ReID embedding
    consecutive_interior_frames: int = 0
    counted_in_occupancy: bool = False
    is_counted_out: bool = False
    trajectory_history: list[tuple[int, int]] = field(default_factory=list)
```

---

## 3. ZONAGE ET POINTS DE RÉFÉRENCE (SUPPORT VUE ZÉNITHALE & PERSPECTIVE)

Implémente un sélecteur configurable pour le calcul de l'ancrage géométrique :

- **Perspective standard :** Point bas central `[x_center, y_max]` (les pieds).
- **Vue Zénithale / Plafond** (`is_zenithal=True`) : Center-box `[x_center, y_center]`.

Définis **3 Zones Spatiales** basées sur la ligne virtuelle :

1. **`ZONE_INTERIEURE`** — Côté salle.
2. **`ZONE_MORTE`** — Bande tampon d'épaisseur `dead_zone_margin` px (ex: 30px autour de la ligne).
3. **`ZONE_EXTERIEURE`** — Côté sortie.

---

## 4. LOGIQUE DÉTAILLÉE DE LA MACHINE À ÉTATS ET DE L'ASSOCIATION

### Phase 1 : Initialisation Automatique (Warm-up Buffer)

- `init_duration_frames` (ex: 90 frames / 3 secondes).
- Les _N_ premières frames servent exclusivement à enregistrer l'effectif de départ (`initial_occupancy`).
- Seules les détections dans la `ZONE_INTERIEURE` stables sur plus de _X_ frames sont comptabilisées dans l'effectif initial.
- **Aucun événement `IN`/`OUT` n'est généré pendant cette phase.**

### Phase 2 : Gestion des Nouvelles Détections à l'Intérieur (Priorité 3)

Lorsqu'un nouvel ID technique apparaît directement dans la `ZONE_INTERIEURE` :

1. **Recherche de correspondance ReID/Spatiale :** Compare son vecteur ReID et sa position avec les entités de l'état `OCCULTEE` (délai de grâce `grace_period_frames`, ex: 300 frames) et `SORTIE_CONFIRMEE`.
2. **Si correspondance trouvée avec une piste `OCCULTEE` :** Réassocie l'identité. L'état repasse en `PRESENTE`. `occupancy_count` inchangé.
3. **Si aucune correspondance (Vrai nouvel arrivant) :**
    - Incrémente `consecutive_interior_frames`.
    - Si `consecutive_interior_frames >= confirmation_threshold` (ex: 15 frames consécutives) :
        - Valide l'état `NOUVELLE_PRESENCE`.
        - `occupancy_count += 1`.
        - `counted_in_occupancy = True`.

### Phase 3 : Validation du Franchissement (Priorités 1 et 2)

Un franchissement n'est exécuté que si la trajectoire traverse la `ZONE_MORTE` et s'éloigne dans la zone opposée :

#### Règle OUT (Sortie)

- **Trajectoire :** `ZONE_INTERIEURE` → `ZONE_MORTE` → `ZONE_EXTERIEURE`.
- **Condition :** `state != SORTIE_CONFIRMEE` ET `counted_in_occupancy == True`.
- **Action :**
    - `total_out += 1`
    - `counted_in_occupancy = False`
    - `is_counted_out = True`
    - `state = SORTIE_CONFIRMEE`
- **Gestion des réimpressions YOLO :** Si YOLO continue de détecter cette personne dans `ZONE_EXTERIEURE`, maintenir `SORTIE_CONFIRMEE`. L'effectif ne bouge plus.

#### Règle IN (Entrée / Retour)

- **Trajectoire :** `ZONE_EXTERIEURE` → `ZONE_MORTE` → `ZONE_INTERIEURE`.
- **Action :**
    - `total_in += 1`
    - `counted_in_occupancy = True`
    - `is_counted_out = False`
    - `state = PRESENTE` (ou `A_RETOURNE`)

### Phase 4 : Gestion des Occultations / Pistes Perdues

- Si une piste BoT-SORT n'est plus mise à jour par YOLO :
    - **Ne pas supprimer** l'objet `LogicalTrack`.
    - Basculer son état vers `OCCULTEE`.
    - Conserver son dernier vecteur d'apparence et sa position théorique.
    - Purger la mémoire uniquement après expiration du `grace_period_frames` **sans décrémenter le compteur d'occupation**.

---

## 5. FORMULE GLOBALE ET TÉLÉMÉTRIE VISUELLE

### Formule d'Occupation

Implémente le calcul d'occupation selon l'équation :

```
occupancy_count = initial_occupancy + total_in + total_new_presences - total_out
```

### Panneau de Contrôle Visuel

Inscrute un panneau de contrôle clair sur la sortie vidéo avec OpenCV :

```
┌──────────────────────────────────────┐
│  OCCUPANCY : 5                       │
│  IN: 12  |  OUT: 8  |  NEW: 1       │
│  Visible: 4  |  Occluded: 1         │
│  Phase: RUNNING                      │
└──────────────────────────────────────┘
```
