import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tracker import LineCrossingTracker


def test_segment_intersection():
    assert LineCrossingTracker._segments_intersect((5, 0), (5, 10), (0, 5), (10, 5))
    assert not LineCrossingTracker._segments_intersect((0, 0), (2, 2), (5, 0), (5, 5))


def test_line_crossing_counts_once():
    class Boxes:
        id = None
        xyxy = None

    tracker = LineCrossingTracker(
        line_p1=(0.0, 0.5), line_p2=(1.0, 0.5),
        band_margin_ratio=0.01, smoothing_window=1,
        min_cooldown_frames=10,
    )

    class TensorLike:
        def __init__(self, value): self.value = value
        def cpu(self): return self
        def numpy(self): return self.value
        def int(self): return self
        def round(self): return self
        def tolist(self): return self.value

    class FrameBoxes:
        def __init__(self, y):
            self.id = TensorLike([7])
            self.xyxy = TensorLike(__import__("numpy").array([[40, y, 60, y + 20]], dtype=float))

    tracker.update(FrameBoxes(20), 100, 100)
    tracker.update(FrameBoxes(60), 100, 100)
    assert tracker.entries == 1
    tracker.update(FrameBoxes(40), 100, 100)
    assert tracker.entries == 1
