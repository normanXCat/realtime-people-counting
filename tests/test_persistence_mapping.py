import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hybrid_tracker import HybridAssociation


def test_lost_track_is_recovered_before_120_frames():
    tracker = HybridAssociation(max_age=120, appearance_threshold=0.40)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = np.array([[20, 10, 40, 80]], dtype=np.float32)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tracker._embedding = lambda _frame, _box: embedding

    stable_id, _ = tracker.resolve([45], box, frame, 1, confidences=[0.95])
    for index in range(2, 121):
        tracker.resolve([], np.empty((0, 4), dtype=np.float32), frame, index)
    recovered_id, hybrid_active = tracker.resolve([91], box, frame, 121, confidences=[0.35])

    assert hybrid_active
    assert recovered_id == stable_id


def test_display_mapping_does_not_increment_for_stable_id():
    id_mapping = {}
    next_display_id = 1

    display_ids = []
    for stable_id in [45, 45, 91]:
        if stable_id not in id_mapping:
            id_mapping[stable_id] = next_display_id
            next_display_id += 1
        display_ids.append(id_mapping[stable_id])

    assert display_ids == [1, 1, 2]
