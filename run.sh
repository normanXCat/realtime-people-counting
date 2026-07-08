#!/bin/bash

# Déterminer le dossier du script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Activer l'environnement virtuel
if [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
else
    echo "Erreur : L'environnement virtuel .venv n'a pas été trouvé."
    exit 1
fi

# Exécuter la détection
python "$DIR/detection.py" "$@"
