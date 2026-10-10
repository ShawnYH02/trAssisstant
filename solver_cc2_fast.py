"""Latency-focused adapter for the Cold Clear 2 bridge.

Reuses Cold Clear's TBP search tree *only when the next observed settled
board / NEXT queue / HOLD state proves the selected move was executed*.
Does not send any game inputs. Falls back to stop/start on any discrepancy.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json
import os
import threading
import time
from typing import Optional

import numpy as np

import solver_cc2 as base
import solver_v5 as v5


@dataclass(frozen=True)
class Observed:
    board: np.ndarray
    current: str
    queue: tuple[str, ...]
    hold: str | None
    can_hold: bool | None = None


@dataclass(frozen=True)
class Selected:
    raw: dict
    cells: tuple[tuple[int, int], ...]
    hold_used: bool


def applied_lock(board: np.ndarray, cells) -> Optional[np.ndarray]:
    """Place 4 minos, clear full lines; None if illegal or out of viewport."""
    if np.shape(board) != (20, 10) or len(cells) != 4:
        return None
    result = np.asarray(board, dtype=bool).copy()
    if len(set(tuple(p) for p in cells)) != 4:
        return None
    for x, y in cells:
        if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))):
            return None
        if not (0 <= x < 10 and 0 <= y < 20) or result[y, x]:
            return None
        result[y, x] = True
    full = np.all(result, axis=1)
    if full.any():
        result = np.concatenate((np.zeros((int(full.sum()), 10), dtype=bool), result[~full]))
    return result


def possible_advance(old: Observed, selected: Selected, new: Observed):
    """Return only newly revealed queue pieces, or None if transition unproven.

    In particular, HOLD without a lock, unexpected placements, garbage, or
    missing NEXT pixels are never forwarded as a `play` command.
    """
    expected_board = applied_lock(old.board, selected.cells)
    if expected_board is None or not np.array_equal(expected_board, new.board):
        return None
    if selected.hold_used:
        if selected.raw.get('location', {}).get('type') != (old.hold or (old.queue[0] if old.queue else None)):
            return None
        consumed = 2 if old.hold is None else 1
        expected_hold = old.current
    else:
        if selected.raw.get('location', {}).get('type') != old.current:
            return None
        consumed = 1
        expected_hold = old.hold
    if new.can_hold is False or len(old.queue) < consumed or new.hold != expected_hold:
        return None
    if new.current != old.queue[consumed - 1]:
        return None
    prior_tail = old.queue[consumed:]
    if len(new.queue) < len(prior_tail) or new.queue[:len(prior_tail)] != prior_tail:
        return None
    added = new.queue[len(prior_tail):]
    # The number of visible NEXT slots should stay the same after locking.
    # Allow an incomplete preview to force restart, not invent unknown pieces.
    if len(added) != consumed or any(p not in base.PIECES for p in added):
        return None
    return added


class FastTBPProcess(base.TBPProcess):
    """Persistent process + verified search-tree handoff across piece locks."""

    def __init__(self, executable: Path, command=None):
        super().__init__(executable, command=command)
        self.observed: Observed | None = None
        self.selected: Selected | None = None
        self._bound_process = None
        self.mode = 'start'
        self.starts = 0
        self.advances = 0
        self.refreshes = 0

    def _same(self, old: Observed, new: Observed) -> bool:
        return (old.current == new.current and old.queue == new.queue and
                old.hold == new.hold and old.can_hold == new.can_hold and
                np.array_equal(old.board, new.board))

    def remember(self, raw: dict, cells, hold_used: bool):
        with self.lock:
            if self.observed is not None and self.proc is not None and self.proc.poll() is None:
                self.selected = Selected(dict(raw), tuple(map(tuple, cells)), bool(hold_used))

    def forget(self):
        """Prevent advancing an old suggestion after current-route validation fails."""
        with self.lock:
            self.selected = None

    def query(self, board, current, queue_pieces, hold, *, combo=0, b2b=False,
              budget_ms=180, can_hold=None):
        # This is called by the existing single-worker pool, NOT the Qt thread.
        new = Observed(np.asarray(board, dtype=bool).copy(), current,
                       tuple(queue_pieces), hold, can_hold)
        with self.lock:
            self.open()
            if self.proc is not self._bound_process:
                self._bound_process = self.proc
                self.observed = None
                self.selected = None
            if self.observed is not None and self._same(self.observed, new):
                self.mode = 'refresh'
                self.refreshes += 1
            else:
                added = (possible_advance(self.observed, self.selected, new)
                         if self.observed is not None and self.selected is not None else None)
                if added is not None:
                    # This is exactly the move CC2 suggested and the observed
                    # board proves it was played. Preserve the existing DAG.
                    # rules/ready also flushes any late reply to the final
                    # suggest poll without modifying the active bot state.
                    self._send({'type': 'rules'})
                    self._await('ready', time.monotonic() + 1.0)
                    self._drain()
                    self._send({'type': 'play', 'move': self.selected.raw})
                    for p in added:
                        self._send({'type': 'new_piece', 'piece': p})
                    self.mode = 'advance'
                    self.advances += 1
                else:
                    self._send({'type': 'stop'})
                    # `stop` has no acknowledgement. Use rules/ready as an
                    # ordered barrier before draining, otherwise a suggestion
                    # from the previous position can arrive after the new
                    # start and be mistaken for the new board's answer.
                    self._send({'type': 'rules'})
                    self._await('ready', time.monotonic() + 1.0)
                    self._drain()
                    self._send({'type': 'start', 'board': base.tbp_board(new.board),
                                'queue': [current, *queue_pieces], 'hold': hold,
                                'combo': max(0, int(combo)), 'back_to_back': bool(b2b)})
                    self.mode = 'start'
                    self.starts += 1
                self.observed, self.selected = new, None

            # A fresh root needs the full configured search budget: CC2 can
            # answer immediately with a zero-node result. An advanced tree has
            # already been searched under the previous root, so its first
            # nonempty answer is the latency benefit tree reuse provides.
            deadline = time.monotonic() + max(0.04, float(budget_ms) / 1000)
            best = None
            while time.monotonic() < deadline:
                self._send({'type': 'suggest'})
                try:
                    result = self._await('suggestion', min(deadline, time.monotonic() + 0.025))
                    if result.get('moves'):
                        best = result
                        if self.mode != 'start':
                            return best
                except TimeoutError:
                    pass
                time.sleep(0.005)
            if best is not None:
                return best
            raise TimeoutError('Cold Clear 2 did not return a nonempty move list in time')

    def close(self):
        super().close()
        self.observed = None
        self.selected = None
        self._bound_process = None


_engine_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cc2-tbp')
_fast_clients: dict[str, FastTBPProcess] = {}
_fast_lock = threading.Lock()


def get_fast_client(executable: Path) -> FastTBPProcess:
    key = str(Path(executable).resolve())
    with _fast_lock:
        if key not in _fast_clients:
            _fast_clients[key] = FastTBPProcess(executable)
        return _fast_clients[key]


def close_fast_engines():
    with _fast_lock:
        clients = list(_fast_clients.values())
        _fast_clients.clear()
    for client in clients:
        client.close()


atexit.register(close_fast_engines)
atexit.register(lambda: _engine_pool.shutdown(wait=False, cancel_futures=True))


def warm_up_cc2(executable: Path):
    """Optional eager handshake, should be invoked from a background thread."""
    try:
        from solver_cc2_first import get_first_client
        get_first_client(executable).open()
    except (OSError, RuntimeError, TimeoutError) as exc:
        print('Cold Clear warmup warning:', exc)


def _emit_metric(settings, data):
    location = os.environ.get('TRASSIST_CC2_METRICS', 'cc2_latency.jsonl')
    if location.lower() in ('off', '0', 'false'):
        return
    try:
        with open(location, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, separators=(',', ':')) + '\n')
    except OSError:
        pass  # Diagnostic logging must never block the placement overlay.


def find_best_cc2_fast(board: np.ndarray, active, next_queue=(), hold=None,
                       can_hold=False, settings: base.SearchSettingsCC2 | None = None,
                       *, initial_b2b=0, initial_combo=-1) -> Optional[base.CC2Recommendation]:
    """Overlay solve function with verified TBP prefetch and route checks."""
    settings = settings or base.SearchSettingsCC2()
    if active.name not in base.PIECES:
        return None
    started = time.perf_counter()
    queue_pieces = tuple(p for p in (next_queue or ()) if p in base.PIECES)
    from solver_cc2_first import get_first_client
    client = get_first_client(base.cc2_path(settings))
    client.set_route_limit(settings.max_current_states)
    # Engine search runs independently of Python's collision-path enumeration.
    # On Windows this overlaps native CC2 computation with Python geometry work.
    query_started = time.perf_counter()
    pending = _engine_pool.submit(
        client.query, board, active.name, queue_pieces, hold,
        combo=max(0, initial_combo), b2b=bool(initial_b2b),
        budget_ms=settings.time_budget_ms, can_hold=can_hold)
    predicted = client.get_predicted_routes(
        board, active, queue_pieces, hold, can_hold,
        settings.max_current_states)
    if predicted is not None:
        paired, states = predicted
    else:
        roots, states = v5._current_locks(
            board, active, settings.max_current_states)
        paired = [(m, False) for m in roots]
        if settings.allow_hold and can_hold is True:
            held = hold or (queue_pieces[0] if queue_pieces else None)
            if held:
                spawn = v5._make_spawn(held)
                if spawn is not None:
                    hold_roots, extra = v5._current_locks(
                        board, spawn, settings.max_current_states)
                    states += extra
                    paired.extend((m, True) for m in hold_roots)
    pathfinding_ms = (time.perf_counter() - started) * 1000
    # Collect engine reply even if no legal routes: its subprocess state must
    # not stay in-flight when the next observed board is submitted.
    msg = pending.result()
    engine_ms = (time.perf_counter() - query_started) * 1000
    if not paired:
        client.forget()
        return None
    selected = None
    for choice in msg.get('moves', []):
        match = base.legal_root_for_suggestion(choice, paired, active.name, hold,
                                               queue_pieces, can_hold is True and settings.allow_hold,
                                               settings.strict_spin)
        if match is not None:
            selected = (choice, match)
            break
    # The search tree is reused only if a root was physically validated.
    if selected is None:
        client.forget()
        return None
    raw, (move, hold_used) = selected
    client.remember(raw, move.cells, hold_used)
    elapsed = (time.perf_counter() - started) * 1000
    _emit_metric(settings, dict(event='solver', mode=client.mode,
                                elapsed_ms=round(elapsed, 3),
                                pathfinding_ms=round(pathfinding_ms, 3),
                                engine_ms=round(engine_ms, 3),
                                states=states,
                                reset_reason=client.last_restart_reason,
                                prefetch_hits=client.prefetch_hits,
                                prefetch_misses=client.prefetch_misses,
                                route_cache_hits=client.route_cache_hits))
    actions = (('HOLD',) if hold_used else ()) + move.actions
    return base.CC2Recommendation(
        name=move.name, rotation=move.r, x=move.x, y=move.y,
        cells=move.cells, score=0.0, cleared=move.lines, actions=actions,
        visited_states=states, spin=move.spin, depth_used=0,
        nodes_expanded=msg.get('move_info', {}).get('nodes', 0),
        elapsed_ms=elapsed, hold_used=hold_used)
