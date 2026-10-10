from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver_cc2_fast import Observed, Selected, applied_lock
from transition_debug import board_masks, record_mismatch, unmask
from transition_report import classify


def example_record(original, later, selected_cells, piece='O'):
    return {
        'reason': 'board_differs_from_predicted_lock',
        'old': {'board': board_masks(original), 'current': piece,
                'queue': ['I', 'T', 'L'], 'hold': None, 'can_hold': True},
        'new': {'board': board_masks(later), 'current': 'I',
                'queue': ['T', 'L', 'J'], 'hold': None, 'can_hold': True},
        'selected': {'type': piece, 'hold_used': False,
                     'cells': [list(cell) for cell in selected_cells]},
        'predicted': board_masks(applied_lock(original, selected_cells)),
    }


def test_encoding_lossless():
    generator = np.random.default_rng(5)
    board = generator.random((20, 10)) > .7
    assert np.array_equal(unmask(board_masks(board)), board)


def test_correct_board_detected():
    old = np.zeros((20, 10), bool)
    chosen = [(0, 18), (1, 18), (0, 19), (1, 19)]
    record = example_record(old, applied_lock(old, chosen), chosen)
    assert classify(record)[0] == 'board_actually_matches'


def test_different_placement_detected():
    old = np.zeros((20, 10), bool)
    chosen = [(0, 18), (1, 18), (0, 19), (1, 19)]
    other = [(3, 18), (4, 18), (3, 19), (4, 19)]
    record = example_record(old, applied_lock(old, other), chosen)
    assert classify(record)[0] == 'different_location_or_pose_for_selected_piece'


def test_unchanged_board_detected():
    old = np.zeros((20, 10), bool)
    chosen = [(0, 18), (1, 18), (0, 19), (1, 19)]
    record = example_record(old, old.copy(), chosen)
    assert classify(record)[0] == 'board_unchanged_or_premature_transition'


def test_no_clear_explains_noise():
    old = np.zeros((20, 10), bool)
    chosen = [(0, 18), (1, 18), (0, 19), (1, 19)]
    later = old.copy()
    later[14, 5] = True
    record = example_record(old, later, chosen)
    assert classify(record)[0] == 'no_four_cell_lock_explains_scan'


def test_tracing_snapshots(monkeypatch, tmp_path):
    import transition_debug as diagnostics

    old = Observed(np.zeros((20, 10), bool), 'O', ('I', 'T'), None, True)
    chosen = ((0, 18), (1, 18), (0, 19), (1, 19))
    selected = Selected({'location': {'type': 'O'}}, chosen, False)
    new = Observed(applied_lock(old.board, chosen), 'I',
                   ('T', 'Z'), None, True)
    trace = tmp_path / 'trace.jsonl'
    monkeypatch.setenv('TRASSIST_CC2_TRACE', str(trace))
    diagnostics._written = 0
    record_mismatch(old, selected, new, 'test',
                    applied_lock(old.board, chosen))
    assert trace.exists()
