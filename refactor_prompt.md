# ENSEMBLE DE PROMPTS ARCHITECTURE : GESTIONNAIRE D'OCCUPATION ROBUSTE (YOLO11 + BOT-SORT)

---

## PROMPT 1 : MACHINE À ÉTATS, HYSTÉRÉSIS & BUFFER TEMPOREL (MONO-LIGNE)

# MISSION DE REFACTORISATION AVANCÉE : GESTIONNAIRE D'OCCUPATION ROBUSTE (STATE MACHINE & HYSTÉRÉSIS)

Tu es un expert en Vision par Ordinateur (YOLO11, BoT-SORT, ReID OSNet) et Architecte Logiciel Python.
Ta mission est de refactoriser le pipeline de traitement existant pour implémenter un **système de gestion de l'effectif d'une salle par machine à états finis (FSM)**.

Le système doit traiter la vidéo comme une **source d'information incomplète** et appliquer des règles strictes de tolérance aux erreurs d'occlusion, aux faux positifs, aux hésitations sur le seuil et aux apparitions soudaines.

---

### 1. RÈGLES ET PRINCIPES INCONTOURNABLES (NON-NÉGOCIABLES)

1. **Règle d'or de décrémentation :** La perte d'un tracker BoT-SORT (disparition de détection) ne doit **JAMAIS** diminuer l'effectif (`occupancy_count`).
2. **Dissociation stricte :** `visible_count` (détections instantanées YOLO) $\neq$ `occupancy_count` (estimation logique de l'effectif).
3. **Même personne déjà sortie :** Si une personne identifiée franchit la ligne `OUT`, l'effectif est décrémenté une seule fois. Sa détection continue du côté extérieur ne déclenche AUCUN nouveau comptage.
4. **Anti-rebond (Hystérésis) :** Un franchissement n'est confirmé que lorsque le centre de masse sort d'une **Zone Morte (Buffer)** entourant la ligne virtuelle.
5. **Phase de Warm-up :** Les événements `IN`/`OUT` sont verrouillés durant la phase d'initialisation automatique.

---

### 2. MODULE DE STRUCTURES DE DONNÉES (`occupancy_types.py`)

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

### 3. ZONAGE ET POINTS DE RÉFÉRENCE (SUPPORT VUE ZÉNITHALE & PERSPECTIVE)

Implémente un sélecteur configurable pour le calcul de l'ancrage géométrique :

- **Perspective standard :** Point bas central `[x_center, y_max]` (les pieds).
- **Vue Zénithale / Plafond** (`is_zenithal=True`) : Center-box `[x_center, y_center]`.

Définis **3 Zones Spatiales** basées sur la ligne virtuelle :

1. **`ZONE_INTERIEURE`** — Côté salle.
2. **`ZONE_MORTE`** — Bande tampon d'épaisseur `dead_zone_margin` px (ex: 30px autour de la ligne).
3. **`ZONE_EXTERIEURE`** — Côté sortie.

---

### 4. LOGIQUE DÉTAILLÉE DE LA MACHINE À ÉTATS ET DE L'ASSOCIATION

#### Phase 1 : Initialisation Automatique (Warm-up Buffer)

- `init_duration_frames` (ex: 90 frames / 3 secondes).
- Les $N$ premières frames servent exclusivement à enregistrer l'effectif de départ (`initial_occupancy`).
- Seules les détections dans la `ZONE_INTERIEURE` stables sur plus de $X$ frames sont comptabilisées dans l'effectif initial.
- **Aucun événement `IN`/`OUT` n'est généré pendant cette phase.**

#### Phase 2 : Gestion des Nouvelles Détections à l'Intérieur (Priorité 3)

Lorsqu'un nouvel ID technique apparaît directement dans la `ZONE_INTERIEURE` :

1. **Recherche de correspondance ReID/Spatiale :** Compare son vecteur ReID et sa position avec les entités de l'état `OCCULTEE` (délai de grâce `grace_period_frames`, ex: 300 frames) et `SORTIE_CONFIRMEE`.
2. **Si correspondance trouvée avec une piste `OCCULTEE` :** Réassocie l'identité. L'état repasse en `PRESENTE`. `occupancy_count` inchangé.
3. **Si aucune correspondance (Vrai nouvel arrivant) :**
   - Incrémente `consecutive_interior_frames`.
   - Si `consecutive_interior_frames >= confirmation_threshold` (ex: 15 frames consécutives) :
     - Valide l'état `NOUVELLE_PRESENCE`.
     - `occupancy_count += 1`.
     - `counted_in_occupancy = True`.

#### Phase 3 : Validation du Franchissement (Priorités 1 et 2)

Un franchissement n'est exécuté que si la trajectoire traverse la `ZONE_MORTE` et s'éloigne dans la zone opposée :

##### Règle OUT (Sortie)
- **Trajectoire :** `ZONE_INTERIEURE` $\rightarrow$ `ZONE_MORTE` $\rightarrow$ `ZONE_EXTERIEURE`.
- **Condition :** `state != SORTIE_CONFIRMEE` ET `counted_in_occupancy == True`.
- **Action :**
  - `total_out += 1`
  - `counted_in_occupancy = False`
  - `is_counted_out = True`
  - `state = SORTIE_CONFIRMEE`
- **Gestion des réimpressions YOLO :** Si YOLO continue de détecter cette personne dans `ZONE_EXTERIEURE`, maintenir `SORTIE_CONFIRMEE`. L'effectif ne bouge plus.

##### Règle IN (Entrée / Retour)
- **Trajectoire :** `ZONE_EXTERIEURE` $\rightarrow$ `ZONE_MORTE` $\rightarrow$ `ZONE_INTERIEURE`.
- **Action :**
  - `total_in += 1`
  - `counted_in_occupancy = True`
  - `is_counted_out = False`
  - `state = PRESENTE` (ou `A_RETOURNE`)

#### Phase 4 : Gestion des Occultations / Pistes Perdues

- Si une piste BoT-SORT n'est plus mise à jour par YOLO :
  - **Ne pas supprimer** l'objet `LogicalTrack`.
  - Basculer son état vers `OCCULTEE`.
  - Conserver son dernier vecteur d'apparence et sa position théorique.
  - Purger la mémoire uniquement après expiration du `grace_period_frames` **sans décrémenter le compteur d'occupation**.

---

### 5. FORMULE GLOBALE ET TÉLÉMÉTRIE VISUELLE

#### Formule d'Occupation

Implémente le calcul d'occupation selon l'équation :

```text
occupancy_count = initial_occupancy + total_in + total_new_presences - total_out
```

#### Panneau de Contrôle Visuel

Inscrute un panneau de contrôle clair sur la sortie vidéo avec OpenCV :

```text
┌──────────────────────────────────────┐
│  OCCUPANCY : 5                       │
│  IN: 12  |  OUT: 8  |  NEW: 1       │
│  Visible: 4  |  Occluded: 1         │
│  Phase: RUNNING                      │
└──────────────────────────────────────┘
```

#### Couleurs des Bounding Boxes

- **`PRESENTE` / `NOUVELLE_PRESENCE` :** Vert `(0, 255, 0)`
- **`OCCULTEE` :** Orange/Gris `(0, 165, 255)` dessiné sur la dernière position connue
- **`EN_ZONE_LIGNE` :** Jaune `(0, 255, 255)`
- **`SORTIE_CONFIRMEE` :** Rouge `(0, 0, 255)`

---

### 6. INSTRUCTIONS D'EXÉCUTION DU REFACTORING

1. Isole la logique métier dans `OccupancyManager`.
2. Intercepte les prédictions provenant de YOLO11 + BoT-SORT pour alimenter le registre `LogicalTrack`.
3. Assure-toi que les paramètres d'hystérésis (`dead_zone_margin`), de warm-up (`init_duration_frames`) et de confirmation (`confirmation_threshold`) soient facilement modifiables via des arguments de configuration ou un fichier YAML.
4. Exécute la refactorisation et fournis un code Python complet, propre, modulaire et documenté.

---

## PROMPT 2 : SAS DE PORTE À DOUBLE LIGNE VIRTUELLE ($L_1$ ET $L_2$)

# MISSION DE REFACTORISATION : SAS DE CONTRÔLE À DOUBLE LIGNE VIRTUELLE & MACHINE À ÉTATS

Tu es un expert en Vision par Ordinateur (YOLO11, BoT-SORT, ReID OSNet) et Architecte Logiciel Python.
Ta mission est d'implémenter un **système de comptage d'occupation de salle basé sur un sas de porte à deux lignes virtuelles ($L_1$ et $L_2$)**, éliminant les erreurs d'hésitation, les oscillations de bounding box et les faux franchissements.

Le principe de base : **Un passage n'est validé que si une personne traverse séquentiellement les deux lignes dans un ordre précis.**

---

### 1. DÉFINITION DU SAS DU TRANSIT (ZONAGE GÉOMÉTRIQUE)

Le système doit permettre de définir deux lignes parallèles ou un polygone de sas :

```text
       [ INTÉRIEUR / SALLE ]
 ═══════════════════════════════════  <-- Ligne 1 (L1 - Limite Salle)
       [ ZONE DE PORTE / SAS ]       <-- Zone tampon (Dead Zone / Margin)
 ═══════════════════════════════════  <-- Ligne 2 (L2 - Limite Extérieur)
       [ EXTÉRIEUR ]
```

- **Ligne 1 ($L_1$) :** Frontière côté intérieur / salle.
- **Ligne 2 ($L_2$) :** Frontière côté extérieur.
- **Zone de Porte (Sas) :** Espace compris entre $L_1$ et $L_2$ (largeur paramétrable, ex: 30 à 60 pixels).

---

### 2. ÉTATS DE LA MACHINE À ÉTATS (`TrackState`)

Mets à jour le module `occupancy_types.py` avec les états suivants pour chaque tracker :

- **`DANS_LA_SALLE` :** Positionné du côté intérieur de $L_1$.
- **`EN_APPROCHE_L1` :** Entre dans le sas en venant de la salle (a franchi $L_1$).
- **`DANS_ZONE_PORTE` :** Présent dans le sas entre $L_1$ et $L_2$.
- **`EN_APPROCHE_L2` :** Entre dans le sas en venant de l'extérieur (a franchi $L_2$).
- **`SORTIE_CONFIRMEE` :** Séquence $L_1 \rightarrow L_2$ terminée. Personne côté extérieur.
- **`ENTREE_CONFIRMEE` :** Séquence $L_2 \rightarrow L_1$ terminée. Personne côté intérieur.
- **`OCCULTEE` :** Piste temporairement masquée/perdue (mémoire conservée).

---

### 3. LOGIQUE SÉQUENTIELLE DE FRANCHISSEMENT DE SAS

#### A. Validation d'une Sortie (OUT)

1. **Étape 1 :** La personne est `DANS_LA_SALLE`.
2. **Étape 2 :** Son point d'ancrage croise $L_1$ $\rightarrow$ L'état passe à `EN_APPROCHE_L1` / `DANS_ZONE_PORTE`.
3. **Étape 3 (Annulation) :** Si la personne fait demi-tour et repasse $L_1$ vers l'intérieur $\rightarrow$ L'état repasse à `DANS_LA_SALLE` (aucun changement d'effectif).
4. **Étape 4 (Validation) :** Si la personne poursuit sa trajectoire et croise $L_2$ vers l'extérieur :
   - L'état passe à `SORTIE_CONFIRMEE`.
   - `total_out += 1`.
   - `occupancy_count -= 1` (une seule fois).

#### B. Validation d'une Entrée (IN)

1. **Étape 1 :** La personne est à l'extérieur.
2. **Étape 2 :** Son point d'ancrage croise $L_2$ $\rightarrow$ L'état passe à `EN_APPROCHE_L2` / `DANS_ZONE_PORTE`.
3. **Étape 3 (Annulation) :** Si la personne fait demi-tour vers l'extérieur et repasse $L_2$ $\rightarrow$ Annulation de la transaction (aucun changement d'effectif).
4. **Étape 4 (Validation) :** Si la personne croise $L_1$ vers l'intérieur :
   - L'état passe à `ENTREE_CONFIRMEE` / `DANS_LA_SALLE`.
   - `total_in += 1`.
   - `occupancy_count += 1`.

---

### 4. CONTRÔLE DES OCCULTATIONS ET DISPARITIONS (RÈGLE STRICTE)

- **Règle de persistance :** Si une détection disparaît au milieu du sas (`DANS_ZONE_PORTE`) ou dans la salle, l'état bascule vers `OCCULTEE`.
- **Effet sur le compteur :** La disparition d'un tracker ne doit **JAMAIS** modifier l'effectif (`occupancy_count`).
- **Délai de grâce ReID :** Conserver la signature ReID et les coordonnées pendant $N$ frames (ex: 300 frames). Si la personne réapparaît, réassocier la piste sans incrémenter ni décrémenter.

---

### 5. HOMOGRAPHIE ET ANCRAGE GÉOMÉTRIQUE (OPTIONNEL / CONFIGURABLE)

- Ajoute une option `use_homography: False` dans le fichier de configuration.
- Si `use_homography: True`, applique une matrice de transformation $H$ $3 \times 3$ (`cv2.perspectiveTransform`) sur les coordonnées de la bounding box avant d'évaluer l'intersection avec $L_1$ et $L_2$ sur le plan rectifié.
- Supporte le choix du point d'ancrage :
  - **Perspective standard :** Bas de boîte `[x_center, y_max]`.
  - **Vue zénithale (`is_zenithal: True`) :** Centre de boîte `[x_center, y_center]`.

---

### 6. TÉLÉMÉTRIE ET ANNOTATION VISUELLE (OPENCV OVERLAY)

Rends le sas visuellement explicite sur le flux vidéo :

- **Ligne 1 ($L_1$) :** Trait Bleu/Cyan avec étiquette `"L1: SALLE"`.
- **Ligne 2 ($L_2$) :** Trait Magenta/Rouge avec étiquette `"L2: EXTÉRIEUR"`.
- **Zone de porte :** Hachurage ou rectangle semi-transparent entre $L_1$ et $L_2$.

#### Panneau de Contrôle d'Occupation

```text
==========================================
MODE : [ DOUBLE LIGNE / SAS ]
------------------------------------------
Présents (Salle)        : {occupancy_count}
Visibles (Instantané)   : {visible_count}
En transit (Dans Sas)   : {in_transit_count}
------------------------------------------
Entrées confirmées (IN) : {total_in}
Sorties confirmées (OUT): {total_out}
==========================================
```
