"""Bound transient board/NEXT/HOLD disagreement before resetting CC2.

The screen can expose a new queue or current identity before line-clear and
lock animation pixels settle. A short, state-stability gate prevents those
intermediate frames from destroying a valid prefetched search tree.
"""
from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np

from solver_cc2_fast import Observed, possible_advance
from solver_cc2_prefetch import PrefetchTBPProcess


class TransitionPending(RuntimeError):
    """The observed transition has not remained coherent long enough yet."""


class ResyncTBPProcess(PrefetchTBPProcess):
    def __init__(self, executable: Path, command=None, settle_ms=130):
        super().__init__(executable, command=command)
        self.settle_ms = max(0.0, min(500.0, float(settle_ms)))
        self._pending_key = None
        self._pending_since = 0.0
        self._released_key = None

    @staticmethod
    def _transition_key(board, current, queue, hold, can_hold):
        array = np.asarray(board, dtype=bool)
        return (array.shape, array.tobytes(), current, tuple(queue), hold,
                can_hold)

    def _gate(self, board, current, queue, hold, can_hold):
        """Return `(pending, can_hold, key)` for a candidate observation."""
        queue = tuple(queue)
        candidate = Observed(np.asarray(board, dtype=bool).copy(), current,
                             queue, hold, can_hold)
        key = self._transition_key(board, current, queue, hold, can_hold)
        if key == self._released_key:
            return False, can_hold, key

        # Startup, repeated observations, and already-proven locks are safe to
        # process immediately. Only an unexpected transition following a
        # validated placement receives the short grace period.
        proven = (possible_advance(self.observed, self.selected, candidate)
                  if self.observed is not None and self.selected is not None
                  else None)
        immediate = (self.observed is None or
                     self._same(self.observed, candidate) or
                     proven is not None or self.selected is None or
                     self.settle_ms <= 0)
        if immediate:
            self._pending_key = None
            self._pending_since = 0.0
            self._released_key = None
            return False, can_hold, key

        now = time.monotonic()
        if key != self._pending_key:
            self._pending_key = key
            self._pending_since = now
            return True, can_hold, key
        pending = (now - self._pending_since) * 1000 < self.settle_ms
        if not pending:
            self._pending_key = None
            self._pending_since = 0.0
            self._released_key = key
        return pending, can_hold, key

    def preflight(self, board, current, queue, hold, can_hold):
        with self.lock:
            pending, _, _ = self._gate(
                board, current, tuple(queue), hold, can_hold)
            return pending

    def close(self):
        super().close()
        self._pending_key = None
        self._pending_since = 0.0
        self._released_key = None


_clients: dict[str, ResyncTBPProcess] = {}
_clients_lock = threading.Lock()


def get_resync_client(executable: Path) -> ResyncTBPProcess:
    key = str(Path(executable).resolve())
    with _clients_lock:
        if key not in _clients:
            _clients[key] = ResyncTBPProcess(executable)
        return _clients[key]


def close_resync_clients():
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()
