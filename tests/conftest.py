"""Fixtures partagées de la suite de tests.

Le dossier ``src/`` est ajouté au ``sys.path`` (le projet n'est pas packagé) et
les fixtures fournissent :

- ``config`` : la **vraie** configuration livrée (``config/pipeline.yaml``), ce
  qui valide au passage le fichier de référence à chaque exécution ;
- ``make_config`` : dérivée revalidée par le même chemin que les surcharges CLI ;
- ``sink`` : collecteur d'événements en mémoire, au contrat de ``EventLogger`` ;
- ``manager_factory`` : gestionnaire d'occupation prêt à l'emploi, budget fixé ;
- ``StubAppearance`` / ``person_box`` : utilitaires de trajectoires synthétiques.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: FPS simulé utilisé par toute la suite : les conversions secondes -> frames
#: deviennent exactes (0.5 s = 5 frames, 0.2 s = 2 frames, 1.0 s = 10 frames).
TEST_FPS = 10.0
TEST_FRAME = (600, 600)  # (hauteur, largeur)


@pytest.fixture(scope="session")
def config():
    from config import load_config

    return load_config()


@pytest.fixture
def make_config(config):
    from config import override_config

    def _make(overrides=None):
        return override_config(config, overrides or {})

    return _make


def make_test_line(p1=(0.0, 0.5), p2=(1.0, 0.5), inside_side="negative"):
    """Ligne de test **explicite**, équivalent des deux clics de l'opérateur.

    La configuration ne porte plus de points : chaque test qui construit un
    gestionnaire d'occupation doit fournir la ligne validée, exactement comme le
    fait ``main`` après la fenêtre de sélection.
    """
    from geometry import VirtualLine

    return VirtualLine(p1=p1, p2=p2, inside_side=inside_side)


@pytest.fixture
def test_config(make_config):
    """Configuration de test : durées courtes, orientation intérieure par défaut."""
    return make_config({
        "line": {"inside_side": "negative"},
        "timing": {
            "warmup_seconds": 0.5,      # 5 frames à 10 fps
            "confirmation_seconds": 0.2,  # 2 frames
            "grace_period_seconds": 1.0,  # 10 frames
            "fps_estimate_window": 4,
        },
        "occupancy": {"snapshot_interval_seconds": 0.1},
        "display": {"enabled": False},
    })


@pytest.fixture
def sink():
    from events import ListSink

    return ListSink("pytest")


@pytest.fixture
def manager_factory(test_config, sink):
    from occupancy_manager import OccupancyManager

    def _build(config=None, frame_budget=None, line=None, **kwargs):
        active = config or test_config
        manager = OccupancyManager(
            active,
            event_sink=sink,
            line=line or make_test_line(inside_side=active.line.inside_side),
            **kwargs,
        )
        manager.set_frame_budget(
            frame_budget or active.timing.frame_budget(TEST_FPS)
        )
        return manager

    return _build


@pytest.fixture
def manager(manager_factory):
    return manager_factory()


@pytest.fixture
def frame():
    return np.zeros((*TEST_FRAME, 3), dtype=np.uint8)


class StubAppearance:
    """Extracteur d'apparence déterministe, sans torch ni image réelle.

    Deux modes : un dictionnaire indexé par la bbox arrondie, ou une fonction
    ``describe_fn(frame, bbox)`` pour les scénarios multi-personnes où il faut
    des descripteurs distincts (ex. indexés par la position horizontale).
    """

    name = "stub"

    def __init__(self, vectors=None, default=None, describe_fn=None):
        self.vectors = {
            tuple(round(float(v), 3) for v in k): v for k, v in (vectors or {}).items()
        }
        self.default = default
        self.describe_fn = describe_fn
        self.calls = 0

    def describe(self, frame, bbox):
        self.calls += 1
        if self.describe_fn is not None:
            value = self.describe_fn(frame, bbox)
        else:
            key = tuple(round(float(v), 3) for v in bbox[:4])
            value = self.vectors.get(key, self.default)
        return None if value is None else np.asarray(value, dtype=np.float32)


def identity_vectors_by_x(vectors_by_x_center, tolerance=12.0):
    """Fabrique un ``describe_fn`` indexé par le centre horizontal de la bbox.

    Une tolérance de quelques pixels modélise le bruit de détection : une
    personne garde la même signature quand sa boîte se décale légèrement.
    """

    def _describe(_frame, bbox):
        x_center = float((bbox[0] + bbox[2]) / 2.0)
        nearest = min(vectors_by_x_center, key=lambda key: abs(key - x_center))
        if abs(nearest - x_center) > tolerance:
            return None
        return vectors_by_x_center[nearest]

    return _describe


def person_box(x_center: float, y_bottom: float, height: float = 200.0, width: float = 80.0):
    """Bounding box synthétique dont l'ancre pieds est ``(x_center, y_bottom)``."""
    return np.array(
        [
            x_center - width / 2.0,
            y_bottom - height,
            x_center + width / 2.0,
            y_bottom,
        ],
        dtype=float,
    )
