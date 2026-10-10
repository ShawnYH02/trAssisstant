"""Local-only compact diagnostics of CC2 tree reset decisions.

No screenshots, pixels, usernames, or input history are recorded. Occupancy
rows (10-bit masks), piece names, queue names, and selection cells only.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

_lock = threading.Lock()
_written = 0


def board_masks(board):
    array = np.asarray(board, dtype=bool)
    if array.shape != (20, 10):
        raise ValueError('Expected 20x10 board')
    weights = 1 << np.arange(10, dtype=np.int64)
    return (array.astype(np.int64) @ weights).astype(int).tolist()


def record_mismatch(old, selected, new, reason, predicted_board):
    """Record one mismatch without ever raising into the live overlay."""
    global _written
    logfile = os.environ.get('TRASSIST_CC2_TRACE',
                             'cc2_transitions.jsonl')
    if logfile.lower() in ('off', '0', 'false'):
        return
    with _lock:
        if _written >= 250:
            return
        _written += 1
        try:
            report = {
                'time': round(time.time(), 3),
                'reason': reason,
                'old': {
                    'board': board_masks(old.board) if old is not None else None,
                    'current': old.current if old is not None else None,
                    'queue': list(old.queue) if old is not None else [],
                    'hold': old.hold if old is not None else None,
                    'can_hold': old.can_hold if old is not None else None,
                },
                'new': {
                    'board': board_masks(new.board),
                    'current': new.current,
                    'queue': list(new.queue),
                    'hold': new.hold,
                    'can_hold': new.can_hold,
                },
                'selected': None if selected is None else {
                    'type': selected.raw.get('location', {}).get('type'),
                    'hold_used': selected.hold_used,
                    'cells': [list(cell) for cell in selected.cells],
                },
                'predicted': (board_masks(predicted_board)
                              if predicted_board is not None else None),
            }
            with open(logfile, 'a', encoding='utf-8') as output:
                output.write(json.dumps(report, separators=(',', ':')) + '\n')
        except (OSError, ValueError, TypeError):
            pass


def unmask(rows):
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError('Expected 20 row masks')
    return np.array([
        [bool(int(row) & (1 << column)) for column in range(10)]
        for row in rows
    ], dtype=bool)
