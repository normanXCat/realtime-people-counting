#!/bin/bash
# Lanceur de l'unique point d'entrée officiel : src/main.py (spec règle 0.1).
#
# Aucun autre pipeline n'est lancé depuis ce script : les modules concurrents
# (hybrid_tracker, reid séparé, line tracker) ont été supprimés ou fusionnés dans
# le gestionnaire d'identités, et les composants expérimentaux (stabilisation,
# ReID externe) s'activent par configuration ou par drapeau explicite.
#
# Exemples :
#   ./run.sh                                        # configuration par défaut (caméra 0)
#   ./run.sh --source video.mp4 --write-video       # fichier vidéo + vidéo annotée
#   ./run.sh --source video.mp4 --inside-side positive
#   ./run.sh --no-long-term-reid                    # baseline 1 (ablation)
#
# À chaque lancement, la première image s'affiche et l'opérateur doit définir la
# ligne de comptage : deux clics, « C » pour valider, « G » pour recommencer,
# « I » pour inverser IN/OUT, « Échap » pour annuler. Aucune ligne par défaut
# n'existe et le comptage ne démarre pas sans cette validation.
#
# Les résultats vont dans results/<session_id>/ (jamais dans le répertoire courant).

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -f "$DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$DIR/.venv/bin/activate"
fi

exec python "$DIR/src/main.py" "$@"
