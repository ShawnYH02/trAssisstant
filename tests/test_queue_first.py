import numpy as np
import cv2

from queue_first import QueueTracker, read_next_queue, read_hold_piece, find_active_matching
from tetris_core import ROTATIONS

# OpenCV HSV hue in [0, 179]; one value per TETR.IO default color family.
HUES = {'Z': 0, 'L': 15, 'O': 30, 'S': 60, 'I': 90, 'J': 115, 'T': 150}


def make_color(name):
    hsv = np.uint8([[[HUES[name], 235, 220]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def make_queue(pieces, slot_height=45):
    img = np.zeros((slot_height * len(pieces), 60, 3), dtype=np.uint8)
    for index, p in enumerate(pieces):
        cv2.rectangle(img, (14, index * slot_height + 11),
                      (45, index * slot_height + 33), make_color(p), -1)
    return img


def observe_n(tracker, queue, n=3, hold=None, with_hold=False):
    for _ in range(n):
        if with_hold:
            tracker.observe(queue, hold)
        else:
            tracker.observe(queue)


def test_read_colored_queue():
    assert read_next_queue(make_queue('IOTSL')) == tuple('IOTSL')


def test_slot_not_readable_does_not_invent_identity():
    img = make_queue('IOTSL')
    img[45:90] = 0
    assert read_next_queue(img) is None


def test_hold_empty_or_visible():
    assert read_hold_piece(np.zeros((60, 60, 3), dtype=np.uint8)) is None
    assert read_hold_piece(make_queue('Z')) == 'Z'


def test_initial_queue_does_not_identify_current():
    tr = QueueTracker()
    observe_n(tr, tuple('IOTSL'))
    assert tr.current is None
    assert tr.queue == tuple('IOTSL')


def test_single_shift_debounced():
    tr = QueueTracker()
    observe_n(tr, tuple('IOTSL'))
    tr.observe(tuple('OTSLJ'))
    assert tr.current is None
    tr.observe(tuple('OTSLJ'))
    assert tr.current is None
    tr.observe(tuple('OTSLJ'))
    assert tr.current == 'I'
    assert tr.queue == tuple('OTSLJ')


def test_skipped_two_shifts():
    tr = QueueTracker()
    observe_n(tr, tuple('IOTSL'))
    observe_n(tr, tuple('TSLJZ'))
    assert tr.current == 'O'


def test_bad_queue_does_not_advance():
    tr = QueueTracker()
    observe_n(tr, tuple('IOTSL'))
    tr.observe(None)
    assert tr.current is None
    observe_n(tr, tuple('OTSLJ'))
    assert tr.current == 'I'


def test_hold_swap_without_queue_change():
    tr = QueueTracker()
    observe_n(tr, tuple('IOTSL'), hold='Z', with_hold=True)
    observe_n(tr, tuple('OTSLJ'), hold='Z', with_hold=True)
    assert tr.current == 'I'
    observe_n(tr, tuple('OTSLJ'), hold='I', with_hold=True)
    assert tr.current == 'Z'
    assert tr.hold == 'I'


def test_empty_hold_waits_for_queue_shift():
    tr = QueueTracker()
    observe_n(tr, tuple('IOTSL'), hold=None, with_hold=True)
    observe_n(tr, tuple('IOTSL'), hold='T', with_hold=True)
    assert tr.current is None
    observe_n(tr, tuple('OTSLJ'), hold='T', with_hold=True)
    assert tr.current == 'I'


def test_unexpected_queue_transition_suspends():
    tr = QueueTracker()
    observe_n(tr, tuple('IOTSL'))
    observe_n(tr, tuple('OTSLJ'))
    assert tr.current == 'I'
    observe_n(tr, tuple('ZJSTO'))
    assert tr.current is None


def test_find_active_by_expected_shape_not_color():
    labels = np.full((24, 10), '.', dtype='<U1')
    cells = [(2+dx, 2+dy) for dx, dy in ROTATIONS['T'][0]]
    for x, y in cells:
        labels[y, x] = 'Z'  # Wrong color classification but correct T geometry.
    active = find_active_matching(labels, 'T')
    assert active is not None
    assert active.name == 'T'
    assert set(active.cells) == set(cells)
    assert find_active_matching(labels, 'I') is None


def test_stale_frames_do_not_change_current():
    tr = QueueTracker(stable_frames=2)
    observe_n(tr, tuple('IOTSL'), n=2)
    observe_n(tr, tuple('OTSLJ'), n=2)
    assert tr.current == 'I'
    tr.observe(tuple('TLJOZ'))  # one unstable glitch
    tr.observe(tuple('OTSLJ'))
    assert tr.current == 'I'
