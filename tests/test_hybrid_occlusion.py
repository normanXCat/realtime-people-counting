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

    # Évite le téléchargement d’un modèle dans ce test : les deux crops ont
    # volontairement la même signature d’apparence.
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tracker._embedding = lambda _frame, _box: embedding

    ids, used_hybrid = tracker.resolve([4], first_box, frame, frame_index=1)
    assert ids == [4]
    assert not used_hybrid

    # Aucune détection pendant 90 frames : la piste reste dans max_age.
    for index in range(2, 92):
        tracker.resolve([], np.empty((0, 4), dtype=np.float32), frame, index)

    ids, used_hybrid = tracker.resolve([19], reappeared_box, frame, frame_index=92)
    assert used_hybrid
    assert ids == [4]
