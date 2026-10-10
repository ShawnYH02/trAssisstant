"""Recovery client for persistent Cold Clear 2 'No placement' failures.

Use V16's tested, lower-frequency polling rather than V17's high-rate polling.
Do not accept empty or geometrically invalid suggestions.  Logs explicit
engine-timeout vs root-rejection diagnostics for the existing overlay.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

from solver_cc2_first import FirstTBPProcess
from solver_cc2_resync import TransitionPending


def _log(**data):
    output = os.environ.get('TRASSIST_CC2_RECOVERY_LOG', 'cc2_recovery.jsonl')
    if output.lower() in ('off', '0', 'false'):
        return
    try:
        with open(output, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'event': 'cc2_recovery', **data}, separators=(',', ':')) + '\n')
    except OSError:
        pass


class ReliableTBPProcess(FirstTBPProcess):
    """Conservative fallback to V16's request schedule, with diagnostics."""

    def __init__(self, executable: Path, command=None, settle_ms=130, minimum_budget_ms=700):
        # Conservative polling matches the earlier reliable cadence while
        # retaining first-available return semantics.
        super().__init__(executable, command=command, settle_ms=settle_ms,
                         poll_ms=25, min_send_interval_ms=30)
        # This is a *temporary availability fallback*, not a speed claim.
        # We stop as soon as the first nonempty suggestion is available.
        self.minimum_budget_ms = max(100, min(3000, int(minimum_budget_ms)))
        self.last_recovery_reason = None

    def query(self, board, current, queue_pieces, hold, *, combo=0, b2b=False,
              budget_ms=350, can_hold=None):
        start = time.perf_counter()
        budget = max(float(budget_ms), self.minimum_budget_ms)
        try:
            reply = super().query(board, current, queue_pieces, hold,
                                  combo=combo, b2b=b2b, budget_ms=budget,
                                  can_hold=can_hold)
        except TransitionPending:
            # Caller already handles incomplete lock transitions.
            self.last_recovery_reason = 'transition_pending'
            raise
        except TimeoutError:
            self.last_recovery_reason = 'engine_timeout'
            _log(status='engine_timeout', piece=current,
                 elapsed_ms=round((time.perf_counter() - start)*1000, 2),
                 budget_ms=budget, mode=self.mode)
            raise
        self.last_recovery_reason = None
        _log(status='engine_reply', piece=current,
             elapsed_ms=round((time.perf_counter() - start)*1000, 2),
             moves=len(reply.get('moves') or []), mode=self.mode)
        return reply

    def retry_for_valid(self, valid: Callable[[dict], bool], budget_ms=250):
        """Keep searching *same* position if the earliest reply has no legal route.

        Returns a reply only after at least one suggested candidate passes the
        existing route matcher. Never fabricates a route or relaxes spin rules.
        """
        start = time.perf_counter()
        with self.lock:
            self._cached_reply = None  # do not reuse known-invalid first reply
            deadline = time.monotonic() + max(.01, float(budget_ms)/1000)
            attempts = 0
            while time.monotonic() < deadline:
                attempts += 1
                self._send({'type':'suggest'})
                try:
                    reply = self._await('suggestion', min(deadline, time.monotonic()+.025))
                except TimeoutError:
                    time.sleep(.003)
                    continue
                moves = reply.get('moves') or []
                if any(valid(move) for move in moves):
                    self._cached_reply = reply
                    self.last_recovery_reason = 'recovered_valid_route'
                    _log(status='recovered_valid_route', attempts=attempts,
                         elapsed_ms=round((time.perf_counter()-start)*1000, 2))
                    return reply
                time.sleep(.01)  # avoid spamming requests without new search work
            self.last_recovery_reason = 'no_legal_route'
            _log(status='no_legal_route', attempts=attempts,
                 elapsed_ms=round((time.perf_counter()-start)*1000, 2))
            return None


def report_unmatched(piece, message, *, states, strict_spin):
    locations = []
    for move in (message.get('moves') or [])[:3]:
        loc = move.get('location') or {}
        locations.append({'type':loc.get('type'), 'x':loc.get('x'),
                          'y':loc.get('y'), 'spin':move.get('spin')})
    _log(status='unmatched_root', piece=piece, roots=states,
         strict_spin=bool(strict_spin), candidates=locations)


_clients = {}
_clients_lock = threading.Lock()


def get_reliable_client(executable: Path):
    key = str(Path(executable).resolve())
    with _clients_lock:
        if key not in _clients:
            def integer(name, default):
                try:
                    return int(os.environ.get(name, str(default)))
                except ValueError:
                    return default
            _clients[key] = ReliableTBPProcess(
                executable,
                settle_ms=integer('TRASSIST_CC2_SETTLE_MS', 130),
                minimum_budget_ms=integer('TRASSIST_CC2_RECOVERY_BUDGET_MS', 700))
        return _clients[key]


def close_reliable_clients():
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()


atexit.register(close_reliable_clients)
