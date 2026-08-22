import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hybrid_tracker import HybridAssociation


def test_identity_survives_complete_occlusion():
    tracker = HybridAssociation(max_age=180, appearance_threshold=0.42)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    first_box = np.array([[10, 10, 30, 70]], dtype=np.float32)
    reappeared_box = np.array([[13, 12, 33, 72]], dtype=np.float32)
    # Évite le téléchargement d’un modèle et force une apparence identique.
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tracker._embedding = lambda _frame, _box: embedding
    ids, used_hybrid = tracker.resolve([4], first_box, frame, frame_index=1, confidences=[0.95])
    assert ids == [4]
    assert not used_hybrid
    for index in range(2, 92):
        tracker.resolve([], np.empty((0, 4), dtype=np.float32), frame, index)
    # Confiance faible au retour : la feature sert au matching mais ne pollue pas la galerie.
    ids, used_hybrid = tracker.resolve([19], reappeared_box, frame, frame_index=92, confidences=[0.35])
    assert used_hybrid
    assert ids == [4]
