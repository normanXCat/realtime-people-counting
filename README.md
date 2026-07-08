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

Pour exécuter le script de détection en temps réel, vous pouvez utiliser le script de démarrage fourni :

```bash
chmod +x run.sh
./run.sh
```

Le script se chargera d'activer l'environnement virtuel `.venv` et lancera la détection en utilisant la webcam de votre machine par défaut (source `0`).

### Paramètres de détection (dans `src/detection.py`)
Vous pouvez modifier les paramètres de la fonction `model.predict()` dans `src/detection.py` :
- `source` : Index de la caméra (ex: `0` pour la webcam locale) ou chemin vers un fichier vidéo.
- `classes` : Filtré sur `[0]` pour détecter uniquement les personnes.
- `conf` : Seuil de confiance minimal (par défaut à `0.5`).
- `show` : Affichage de la fenêtre vidéo en temps réel (`True`).

