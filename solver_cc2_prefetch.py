"""V14 speculative Cold Clear search for the existing V13 adapter.

Uses the SAME Cold Clear process; never sends keyboard/game inputs.
After selecting a legal suggestion, asks the search engine to advance to that
*predicted* position while the human is still moving the current piece.
Only when the real board, HOLD and queue prove the lock do we append newly
revealed pieces and accept the prefetched tree. Otherwise restart safely.
"""
from __future__ import annotations

import atexit
from pathlib import Path
import threading
import time

import numpy as np

import solver_cc2 as base
from solver_cc2_fast import (FastTBPProcess, Observed, Selected, applied_lock,
                             possible_advance)


class PrefetchTBPProcess(FastTBPProcess):
    """Single engine, speculative `play`, verified handoff on actual lock."""
    def __init__(self, executable: Path, command=None):
        super().__init__(executable, command=command)
        self._prefetched = False
        self._cached_reply = None
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self._reasons: dict[str, int] = {}
        self.last_restart_reason = None

    def _reason(self, why: str):
        self._reasons[why] = self._reasons.get(why, 0) + 1

    def _record_mismatch(self, new: Observed, reason: str):
        """Write local diagnostics without affecting engine decisions."""
        try:
            from transition_debug import record_mismatch
            predicted = (
                applied_lock(self.observed.board, self.selected.cells)
                if self.observed is not None and self.selected is not None
                else None)
            record_mismatch(self.observed, self.selected, new, reason,
                            predicted)
        except Exception:
            pass

    def _explain_mismatch(self, old: Observed | None, selection: Selected | None,
                          new: Observed) -> str:
        if old is None:
            return 'first_observation'
        if selection is None:
            return 'no_validated_previous_placement'
        predicted_board = applied_lock(old.board, selection.cells)
        if predicted_board is None:
            return 'invalid_or_hidden_predicted_lock'
        if not np.array_equal(predicted_board, new.board):
            return 'board_differs_from_predicted_lock'
        consume = 2 if selection.hold_used and old.hold is None else 1
        expected_hold = old.current if selection.hold_used else old.hold
        if new.can_hold is False:
            return 'hold_cooldown_not_reset'
        if new.hold != expected_hold:
            return 'hold_identity_mismatch'
        if len(old.queue) < consume or new.current != old.queue[consume - 1]:
            return 'current_piece_queue_mismatch'
        tail = old.queue[consume:]
        if new.queue[:len(tail)] != tail:
            return 'next_queue_shift_mismatch'
        if len(new.queue) - len(tail) != consume:
            return 'next_queue_incomplete'
        return 'other_transition_mismatch'

    def _reset_for(self, observed: Observed, combo, b2b):
        self._send({'type': 'stop'})
        # `stop` has no acknowledgement. This ordered no-op barrier prevents a
        # late suggestion from the prior speculative board being consumed as
        # the first answer for this observed board.
        self._send({'type': 'rules'})
        self._await('ready', time.monotonic() + 1.0)
        self._drain()
        self._send({'type': 'start', 'board': base.tbp_board(observed.board),
                    'queue': [observed.current, *observed.queue],
                    'hold': observed.hold, 'combo': max(0, int(combo)),
                    'back_to_back': bool(b2b)})
        self.mode = 'start'
        self.starts += 1
        self._prefetched = False
        self._cached_reply = None

    def query(self, board, current, queue_pieces, hold, *, combo=0, b2b=False,
              budget_ms=180, can_hold=None):
        new = Observed(np.asarray(board, dtype=bool).copy(), current,
                       tuple(queue_pieces), hold, can_hold)
        with self.lock:
            self.open()
            if self.proc is not self._bound_process:
                self._bound_process = self.proc
                self.observed = None
                self.selected = None
                self._cached_reply = None
                self._prefetched = False

            if self.observed is not None and self._same(self.observed, new):
                # The engine may already be searching the *next* position.
                # Never ask it for the old piece again in that situation.
                if self._cached_reply is not None:
                    self.mode = 'cached_same_piece'
                    self.refreshes += 1
                    return self._cached_reply
                self.mode = 'refresh'
                self.refreshes += 1
            else:
                advance = (possible_advance(self.observed, self.selected, new)
                           if self.observed is not None and self.selected is not None
                           else None)
                if advance is not None:
                    self.last_restart_reason = None
                    self._send({'type': 'rules'})
                    self._await('ready', time.monotonic() + 1.0)
                    self._drain()
                    if self._prefetched:
                        # The 'play' was already sent when we selected the move.
                        self.prefetch_hits += 1
                        self.mode = 'prefetch_hit'
                    else:
                        # Safe fallback if prefetch send failed or was unavailable.
                        self._send({'type': 'play', 'move': self.selected.raw})
                        self.mode = 'advance'
                    for piece in advance:
                        self._send({'type': 'new_piece', 'piece': piece})
                    self.advances += 1
                    self._prefetched = False
                    self._cached_reply = None
                else:
                    self.prefetch_misses += int(self._prefetched)
                    reason = self._explain_mismatch(self.observed, self.selected, new)
                    self.last_restart_reason = reason
                    self._reason(reason)
                    self._record_mismatch(new, reason)
                    self._reset_for(new, combo, b2b)
                self.observed = new
                self.selected = None

            # A reset/start still gets the full configured budget; accepting
            # its first nonempty result can select a zero-node root. A
            # prefetched/advanced tree has already searched during the human's
            # move, so its first nonempty result is intentionally returned.
            deadline = time.monotonic() + max(0.025, float(budget_ms) / 1000.0)
            best = None
            while time.monotonic() < deadline:
                self._send({'type': 'suggest'})
                try:
                    msg = self._await('suggestion', min(deadline, time.monotonic() + .025))
                    if msg.get('moves'):
                        self._cached_reply = msg
                        best = msg
                        if self.mode != 'start':
                            return msg
                except TimeoutError:
                    pass
                time.sleep(.003)
            if best is not None:
                return best
            raise TimeoutError('Cold Clear 2 has no nonempty suggestion yet')

    def remember(self, raw: dict, cells, hold_used: bool):
        """Commit a VALIDATED move to search immediately, not after the lock.

        Never advance without the legal root validation the V13 bridge performs.
        If the player deviates, query() checks the real board and resets.
        """
        with self.lock:
            if (self.observed is None or self.proc is None or
                    self.proc.poll() is not None or self._prefetched):
                return
            self.selected = Selected(dict(raw), tuple(map(tuple, cells)), bool(hold_used))
            try:
                self._send({'type': 'rules'})
                self._await('ready', time.monotonic() + 1.0)
                self._drain()
                self._send({'type': 'play', 'move': self.selected.raw})
                self._prefetched = True
            except (OSError, RuntimeError, ValueError):
                # The selected move is retained; on confirmed lock the normal
                # 'play' handoff can be retried or the process will be reopened.
                self._prefetched = False

    def forget(self):
        # The engine might already be searching a predicted next board. We
        # cannot undo a 'play'; reset will be required for a future request.
        with self.lock:
            self.selected = None
            self._cached_reply = None

    def close(self):
        with self.lock:
            super().close()
            self._prefetched = False
            self._cached_reply = None


_instances: dict[str, PrefetchTBPProcess] = {}
_instances_lock = threading.Lock()


def get_prefetch_client(executable: Path) -> PrefetchTBPProcess:
    key = str(Path(executable).resolve())
    with _instances_lock:
        if key not in _instances:
            _instances[key] = PrefetchTBPProcess(executable)
        return _instances[key]


def close_prefetch_clients():
    with _instances_lock:
        clients = list(_instances.values())
        _instances.clear()
    for client in clients:
        client.close()


atexit.register(close_prefetch_clients)
