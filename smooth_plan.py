"""Stable, validated placement preview cache for trAssisstant.

Tracks placement recommendations across small pose changes and confirmed locks.
Never assumes that a future placement was executed without verifying the board,
queue and hold state. No game inputs are generated.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any
import numpy as np


def state_key(board: np.ndarray, active_name: str, queue, hold, can_hold, settings):
    """A lock-state key. Excludes live piece x/y/rotation; includes all strategy inputs."""
    if board.shape != (20, 10):
        raise ValueError('Expected 20x10 settled board')
    return (board.astype(np.bool_, copy=False).tobytes(), active_name,
            tuple(queue or ()), hold, can_hold, settings)


def after_lock(board: np.ndarray, cells) -> np.ndarray | None:
    if board.shape != (20, 10) or len(cells) != 4:
        return None
    out = np.array(board, dtype=bool, copy=True)
    if len(set(tuple(c) for c in cells)) != 4:
        return None
    for x, y in cells:
        if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
            return None
        if not (0 <= x < 10 and 0 <= y < 20) or out[y, x]:
            return None
        out[y, x] = True
    clear = np.all(out, axis=1)
    n = int(clear.sum())
    if n:
        out = np.concatenate((np.zeros((n, 10), dtype=bool), out[~clear]), axis=0)
    return out


@dataclass
class CachedView:
    best: Any
    predicted: bool = False  # A future preview, not a freshly routed current move.


class PlanMemory:
    """Keep plan stable during movement; promote a verified future preview at lock.

    store(...) only after an async solver result is accepted for the EXACT state
    requested. show(...) returns a preview when safe or None otherwise.
    """
    def __init__(self):
        self.key = None
        self.best = None
        self.board = None
        self.predicted = False

    def store(self, key, board, best):
        if best is None:
            self.clear()
            return
        self.key = key
        self.board = np.asarray(board, dtype=bool).copy()
        self.best = best
        self.predicted = False

    def clear(self):
        self.key = self.best = self.board = None
        self.predicted = False

    def show(self, key, board) -> CachedView | None:
        if self.best is None:
            return None
        if self.key == key:
            return CachedView(self.best, self.predicted)
        if self._accept_hold(key, board):
            return CachedView(self.best, True)
        if self._promote(key, board):
            return CachedView(self.best, True)
        return None

    def _accept_hold(self, next_key, board):
        """A planned HOLD action happened but the current piece has not locked."""
        prev = self.key
        best = self.best
        if prev is None or best is None or self.board is None or not best.hold_used:
            return False
        _, old_current, old_queue, old_hold, old_can_hold, old_settings = prev
        _, new_current, new_queue, new_hold, new_can_hold, new_settings = next_key
        if old_can_hold is not True or new_can_hold is not False:
            return False
        if old_settings != new_settings or new_hold != old_current:
            return False
        if not np.array_equal(np.asarray(board, dtype=bool), self.board):
            return False
        if old_hold is None:
            if not old_queue or new_current != old_queue[0]:
                return False
            expected_queue_tail = old_queue[1:]
        else:
            if new_current != old_hold:
                return False
            expected_queue_tail = old_queue
        if tuple(new_queue[:len(expected_queue_tail)]) != tuple(expected_queue_tail):
            return False
        if best.name != new_current:
            return False
        try:
            next_actions = tuple(a for a in best.actions if a != 'HOLD')
            promoted = replace(best, actions=next_actions, hold_used=False)
        except (TypeError, ValueError):
            return False
        self.key = next_key
        self.best = promoted
        self.board = np.asarray(board, dtype=bool).copy()
        self.predicted = True
        return True

    def _promote(self, next_key, board):
        prev = self.key
        best = self.best
        if prev is None or best is None or self.board is None:
            return False
        _, old_current, old_queue, old_hold, _, old_settings = prev
        _, new_current, new_queue, new_hold, new_can_hold, new_settings = next_key
        name = getattr(best, 'next_name', None)
        cells = getattr(best, 'next_cells', None)
        if not name or not cells or name != new_current or old_settings != new_settings:
            return False
        expected_board = after_lock(self.board, best.cells)
        if expected_board is None or not np.array_equal(expected_board, np.asarray(board, dtype=bool)):
            return False
        used_hold = bool(getattr(best, 'hold_used', False))
        if used_hold:
            expected_hold = old_current
            consumed = 2 if old_hold is None else 1
            # A future hold-swap branch can be taken, but the preview protocol
            # does not specify which; require an unambiguous next piece.
        else:
            expected_hold = old_hold
            consumed = 1
        if new_hold != expected_hold or new_can_hold is False:
            return False
        expected_queue_tail = old_queue[consumed:]
        if tuple(new_queue[:len(expected_queue_tail)]) != tuple(expected_queue_tail):
            return False
        # Only accept if predicted NEXT is the piece the queue would spawn.
        # A future HOLD continuation requires an extended protocol to validate.
        if not old_queue or consumed < 1 or (consumed - 1) >= len(old_queue):
            return False
        if name != old_queue[consumed - 1]:
            return False
        # replace() works with the actual RecommendationV5 dataclass, and
        # preserves score/debug fields. Route and clear count are intentionally
        # removed because those refer to the previous piece.
        try:
            promoted = replace(
                best, name=name, cells=tuple(map(tuple, cells)),
                next_name=getattr(best, 'third_name', None),
                next_cells=getattr(best, 'third_cells', None),
                third_name=None, third_cells=None,
                actions=(), hold_used=False, cleared=0,
            )
        except (TypeError, ValueError):
            return False
        self.key = next_key
        self.board = np.asarray(board, dtype=bool).copy()
        self.best = promoted
        self.predicted = True
        return True
