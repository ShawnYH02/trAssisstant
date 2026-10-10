"""First-available Cold Clear 2 replies with search-readiness timing.

Replaces V16's TBP client only. Never assumes a move exists until Cold Clear
returns a nonempty `suggestion` and V13's legal-root checks accept it.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

import solver_cc2 as base
from solver_cc2_fast import Observed, possible_advance
from solver_cc2_resync import ResyncTBPProcess, TransitionPending


class FirstTBPProcess(ResyncTBPProcess):
    """Fast polling + diagnostics, preserving V14 prefetch and V16 transition rules."""

    def __init__(self, executable: Path, command=None, settle_ms=130,
                 poll_ms: float = 8.0, min_send_interval_ms: float = 5.0):
        super().__init__(executable, command=command, settle_ms=settle_ms)
        self.poll_ms = max(2.0, min(50.0, float(poll_ms)))
        self.min_send_interval_ms = max(1.0, min(30.0, float(min_send_interval_ms)))
        self.last_first_metrics: dict = {}

    def _poll_first(self, budget_ms):
        """Return the first *nonempty* suggestion, without a fixed think-time.

        Cold Clear 2 drops suggest requests when no result is available. A
        single request isn't guaranteed to produce an eventual response, so
        retry with a bounded interval. No stale data is accepted as a move by
        the caller: `legal_root_for_suggestion` still validates its cells.
        """
        started = time.perf_counter()
        deadline = time.monotonic() + max(0.020, float(budget_ms) / 1000.0)
        attempts = 0
        empty = 0
        timeouts = 0
        first_reply_ms = None
        next_send = time.monotonic()
        result = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sleep_for = next_send - time.monotonic()
            if sleep_for > 0:
                time.sleep(min(sleep_for, remaining))
            if time.monotonic() >= deadline:
                break
            attempts += 1
            self._send({'type': 'suggest'})
            sent_at = time.monotonic()
            next_send = sent_at + self.min_send_interval_ms / 1000.0
            try:
                reply = self._await('suggestion', min(deadline, sent_at + self.poll_ms / 1000.0))
                if first_reply_ms is None:
                    first_reply_ms = round((time.perf_counter() - started) * 1000, 3)
                if reply.get('moves'):
                    result = reply
                    break
                empty += 1
            except TimeoutError:
                timeouts += 1
        metrics = {
            'event': 'cc2_first_suggestion',
            'mode': self.mode,
            'first_reply_ms': first_reply_ms,
            'first_nonempty_ms': (round((time.perf_counter() - started) * 1000, 3)
                                  if result is not None else None),
            'polls': attempts,
            'empty_replies': empty,
            'unanswered_polls': timeouts,
            'budget_ms': round(float(budget_ms), 3),
            'found': result is not None,
        }
        self.last_first_metrics = metrics
        _record(metrics)
        if result is None:
            raise TimeoutError('No nonempty Cold Clear suggestion before the search deadline')
        return result

    def query(self, board, current, queue_pieces, hold, *, combo=0, b2b=False,
              budget_ms=350, can_hold=None):
        with self.lock:
            pending, normalized_can_hold, _ = self._gate(
                board, current, tuple(queue_pieces), hold, can_hold)
            if pending:
                raise TransitionPending('Waiting for coherent board / NEXT / HOLD after lock')
            new = Observed(np.asarray(board, dtype=bool).copy(), current,
                           tuple(queue_pieces), hold, normalized_can_hold)
            self.open()
            if self.proc is not self._bound_process:
                self._bound_process = self.proc
                self.observed = None
                self.selected = None
                self._cached_reply = None
                self._prefetched = False

            if self.observed is not None and self._same(self.observed, new):
                if self._cached_reply is not None:
                    self.mode = 'cached_same_piece'
                    self.refreshes += 1
                    self.last_first_metrics = {
                        'event': 'cc2_first_suggestion', 'mode': self.mode,
                        'first_reply_ms': 0.0, 'first_nonempty_ms': 0.0,
                        'polls': 0, 'empty_replies': 0, 'unanswered_polls': 0,
                        'budget_ms': budget_ms, 'found': True,
                    }
                    _record(self.last_first_metrics)
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
                        self.prefetch_hits += 1
                        self.mode = 'prefetch_hit'
                    else:
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

            msg = self._poll_first(budget_ms)
            self._cached_reply = msg
            return msg


_clients: dict[str, FirstTBPProcess] = {}
_clients_lock = threading.Lock()


def get_first_client(executable: Path) -> FirstTBPProcess:
    key = str(Path(executable).resolve())
    with _clients_lock:
        if key not in _clients:
            def number(name, default):
                try:
                    return float(os.getenv(name, str(default)))
                except ValueError:
                    return float(default)
            _clients[key] = FirstTBPProcess(
                executable,
                settle_ms=int(max(0, min(500, number('TRASSIST_CC2_SETTLE_MS', 130)))),
                poll_ms=number('TRASSIST_CC2_FIRST_POLL_MS', 8),
                min_send_interval_ms=number('TRASSIST_CC2_FIRST_INTERVAL_MS', 5),
            )
        return _clients[key]


def close_first_clients():
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for c in clients:
        c.close()


def _record(data):
    filepath = os.environ.get('TRASSIST_CC2_METRICS', 'cc2_latency.jsonl')
    if filepath.lower() in ('0', 'off', 'false'):
        return
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, separators=(',', ':')) + '\n')
    except OSError:
        pass


atexit.register(close_first_clients)
