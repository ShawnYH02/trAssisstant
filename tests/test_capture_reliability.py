from pathlib import Path

import numpy as np

from capture_frame import CoherentFrame
from pose_guard import PoseGapGuard


class Capture:
    def __init__(self):
        self.calls = []

    def grab(self, region):
        self.calls.append(dict(region))
        return np.fromfunction(
            lambda y, x, channel: x + region['left'],
            (region['height'], region['width'], 3), dtype=int
        ).astype(np.uint8)


def test_one_capture_serves_multiple_rois():
    capture = Capture()
    board = {'left': 100, 'top': 200, 'width': 100, 'height': 300}
    queue = {'left': 220, 'top': 210, 'width': 50, 'height': 120}
    hold = {'left': 20, 'top': 210, 'width': 60, 'height': 100}
    frame = CoherentFrame(capture, [board, queue, hold])
    assert frame.coherent
    assert len(capture.calls) == 1
    assert frame.crop(board).shape == (300, 100, 3)
    assert frame.crop(queue).shape == (120, 50, 3)
    assert int(frame.crop(queue)[0, 0, 0]) == 220


def test_large_union_falls_back_to_individual_capture():
    capture = Capture()
    first = {'left': 0, 'top': 0, 'width': 10, 'height': 10}
    second = {'left': 1000, 'top': 1000, 'width': 100, 'height': 100}
    frame = CoherentFrame(capture, [first, second], max_pixels=10_000)
    assert not frame.coherent
    assert not capture.calls
    assert frame.crop(first).shape == (10, 10, 3)
    assert len(capture.calls) == 1


def test_pose_gap_keeps_short_same_state_dropout():
    guard = PoseGapGuard(90)
    labels = np.zeros((24, 10), dtype=np.uint8)
    guard.observe(labels, 'T', ('L', 'I'), None,
                  ghost_verified=True, now=1.0)
    modified = labels.copy()
    modified[2, 4:8] = 1
    assert guard.keep(modified, 'T', ('L', 'I'), None, now=1.05)
    assert not guard.keep(modified, 'T', ('L', 'I'), None, now=1.15)
    assert not guard.keep(modified, 'T', ('I', 'L'), None, now=1.05)


def test_pose_gap_rejects_garbage_and_unverified_ghost():
    labels = np.zeros((20, 10), dtype=np.uint8)
    guard = PoseGapGuard()
    guard.observe(labels, 'I', ('T',), None,
                  ghost_verified=True, now=1.0)
    changed = labels.copy()
    changed[-1, :9] = 1
    assert not guard.keep(changed, 'I', ('T',), None, now=1.01)

    guard.observe(labels, 'T', ('I',), None,
                  ghost_verified=False, now=2.0)
    assert not guard.keep(labels, 'T', ('I',), None, now=2.01)


def test_overlay_is_coherent_and_fail_closed():
    source = (Path(__file__).resolve().parents[1] /
              'assistant_overlay.py').read_text(encoding='utf-8')
    assert 'CoherentFrame(' in source
    assert 'pose_guard.keep(' in source
    assert 'COLD CLEAR UNAVAILABLE' in source
    assert 'legacy fallback is disabled' in source
