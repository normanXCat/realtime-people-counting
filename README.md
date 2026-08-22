# Real-Time People Counting

Ce projet implémente un pipeline de comptage de personnes en temps réel basé sur la détection d'objets (avec YOLO11) et le suivi multi-objets (Multi-Object Tracking). Développé dans le cadre du projet de fin d'études / mémoire de Master STIC en 2026.

## 🚀 Fonctionnalités

- **Détection de personnes en temps réel** à l'aide du modèle YOLO11 (`models/yolo11s.pt`).
- **Suivi d'objets (Tracking)** pour surveiller le mouvement des personnes et éviter le double comptage.
- **Comptage automatique** avec gestion de franchissement de ligne virtuelle (Entrées / Sorties).
- **Script de lancement automatisé** assurant l'activation de l'environnement virtuel Python.

---

## 📁 Structure du Projet

Le dépôt est structuré comme suit :

* 📁 **`docs/`** : Documents de recherche, étude bibliographique et planification du projet (Gantt).
  - `Etude_Bibliographique_Comptage_Personnes.pdf` : Rapport d'étude de l'existant.
  - `Gantt_Comptage_Personnes.xlsx` : Planification temporelle du projet.
* 📁 **`models/`** : Fichiers de configuration et poids pour les architectures de modèles supportées.
  - `coco.names` : Liste des classes COCO (dont la classe *person*).
  - `yolov4.cfg` / `yolov4.weights` : Configuration et poids pour YOLOv4.
  - `yolo11s.pt` : Poids pré-entraînés pour YOLO11.
* 📁 **`src/`** : Code source de l'application.
  - `detection.py` : Script Python principal effectuant la détection et affichant le flux vidéo.
* 📄 **`run.sh`** : Script Bash utilitaire pour lancer le projet dans l'environnement virtuel local.

---

## 🛠️ Installation et Configuration

### Prérequis
- Python 3.8 ou supérieur
- Système d'exploitation Linux/macOS (ou Windows avec Git Bash/WSL)

### 1. Cloner le dépôt
```bash
git clone https://github.com/normanxcat/realtime-people-counting.git
cd realtime-people-counting
```

### 2. Configurer l'environnement virtuel
Si l'environnement virtuel `.venv` n'existe pas encore, créez-le et installez les dépendances requises :
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python
```

---

## 💻 Utilisation

Pour exécuter le projet avec le suivi ByteTrack et le comptage de franchissement de ligne, utilisez le script de démarrage fourni :

```bash
chmod +x run.sh
./run.sh
```

Le script active automatiquement l'environnement virtuel `.venv` et lance l'application.

### Options de ligne de commande

Vous pouvez désormais passer des arguments pour configurer le comportement de l'application (qui seront transmis directement par `run.sh`) :

```bash
./run.sh [options]
```

#### Liste des options disponibles :
- `--source <valeur>` : Source vidéo à analyser. Peut être l'index de votre caméra locale (ex: `0` par défaut) ou le chemin vers un fichier vidéo (ex: `/chemin/vers/video.mp4`).
- `--model <chemin>` : Chemin vers le fichier de modèle YOLO (par défaut : `models/yolo11s.pt`).
- `--conf <valeur>` : Seuil de confiance minimal de détection de 0.0 à 1.0 (par défaut : `0.5`).
- `--line-pos <valeur>` : Position verticale de la ligne de franchissement virtuelle (fraction de la hauteur de l'image de 0.0 à 1.0, par défaut : `0.6` soit 60%).
- `--no-show` : Désactive l'affichage graphique de la fenêtre OpenCV (utile pour le traitement en arrière-plan ou sans interface graphique).

#### Exemples d'utilisation :

1. **Lancer avec la webcam par défaut (source 0) et configurer la ligne à 50% de la hauteur :**
   ```bash
   ./run.sh --source 0 --line-pos 0.5
   ```

2. **Lancer le traitement sur un fichier vidéo avec un seuil de confiance de 0.4 :**
   ```bash
   ./run.sh --source "chemin/ma_video.mp4" --conf 0.4
   ```

3. **Lancer le traitement en tâche de fond (sans fenêtre OpenCV) :**
   ```bash
   ./run.sh --source "chemin/ma_video.mp4" --no-show
   ```



## Réglages recommandés sur la branche `dev`

La branche `dev` conserve **ByteTrack comme chemin nominal** afin de limiter le coût d’inférence. Le module `src/hybrid_tracker.py` active une association de type DeepSORT uniquement lorsqu’une ambiguïté est détectée : recouvrement critique entre boîtes, ID ByteTrack dupliqué ou réapparition d’une piste dans la fenêtre de persistance. L’association secondaire combine similarité cosinus d’un embedding MobileNetV3, IoU avec une prédiction cinématique et l’algorithme hongrois.

| Paramètre | Valeur par défaut | Rôle |
|---|---:|---|
| `--conf` | `0.35` | Récupère davantage de personnes petites ou partiellement occultées. |
| `--iou` | `0.55` | Seuil IoU du NMS YOLO, compromis entre doublons et détections proches. |
| `--imgsz` | `960` | Améliore la détection des personnes éloignées au prix d’une latence supérieure. |
| `--occlusion-iou` | `0.35` | Déclenche le mode d’association apparence/mouvement. |
| `--max-age` | `150` frames | Conserve une identité récupérable pendant environ 5 secondes à 30 FPS. |
| `match_thresh` | `0.75` | Rend l’association ByteTrack moins permissive afin de réduire les fusions. |
| `track_low_thresh` | `0.08` | Seconde passe ByteTrack pour les détections affaiblies. |
| `track_buffer` | `150` frames | Évite la création d’un nouvel ID après une occultation courte. |

Le comptage utilise désormais le centre de la boîte, une distance signée lissée, une bande d’hystérésis et la vérification que le segment entre deux centres consécutifs coupe réellement la ligne. Le cooldown par ID empêche les doubles comptages lors d’une hésitation autour de la ligne.

Pour une caméra fixe et une vidéo à 30 FPS, un premier lancement peut être effectué avec :

```bash
./run.sh --source video.mp4 --conf 0.35 --iou 0.55 --imgsz 960 --no-show
```

Les tests unitaires de `tests/test_line_tracker.py` nécessitent `pytest`; la compilation statique et les assertions géométriques peuvent être vérifiées sans caméra avec `python3 -m py_compile src/*.py`.
