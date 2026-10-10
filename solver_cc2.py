"""Cold Clear 2 / Tetris Bot Protocol adapter for trAssisstant.

The Cold Clear process is started once and remains alive across state changes.
Observed boards are authoritative; we reset the TBP game state for each new
locked-board state rather than falsely telling the engine a suggested move was
actually played. This version does not yet reuse the engine search tree.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import solver_v5 as v5
from solver_v5 import describe_action_v5

PIECES = frozenset("IOTSZJL")
ORIENTATIONS = {"north": 0, "east": 1, "south": 2, "west": 3}
# Exact mino offsets from cold-clear-2/src/data.rs. Coordinates: +y upward.
CC2_CELLS = {
    "I": ((-1, 0), (0, 0), (1, 0), (2, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((-1, 0), (0, 0), (1, 0), (0, 1)),
    "L": ((-1, 0), (0, 0), (1, 0), (1, 1)),
    "J": ((-1, 0), (0, 0), (1, 0), (-1, 1)),
    "S": ((-1, 0), (0, 0), (0, 1), (1, 1)),
    "Z": ((-1, 1), (0, 1), (0, 0), (1, 0)),
}


@dataclass(frozen=True)
class SearchSettingsCC2:
    executable: str | None = None
    time_budget_ms: float = 350.0
    max_current_states: int = 5000
    allow_hold: bool = True
    strict_spin: bool = True


@dataclass(frozen=True)
class CC2Recommendation:
    name: str
    rotation: int
    x: int
    y: int
    cells: tuple[tuple[int, int], ...]
    score: float
    cleared: int
    actions: tuple[str, ...]
    visited_states: int
    spin: str = "none"
    depth_used: int = 0
    nodes_expanded: int = 0
    elapsed_ms: float = 0.0
    hold_used: bool = False
    b2b_after: int = 0
    next_name: str | None = None
    next_cells: tuple[tuple[int, int], ...] | None = None
    third_name: str | None = None
    third_cells: tuple[tuple[int, int], ...] | None = None


def cc2_path(settings: SearchSettingsCC2 | None = None) -> Path:
    custom = (settings.executable if settings else None) or os.environ.get("TRASSIST_CC2_EXE")
    if custom:
        return Path(custom).expanduser().resolve()
    exe = "cold-clear-2.exe" if os.name == "nt" else "cold-clear-2"
    return Path(__file__).resolve().parent / "cold_clear_2" / "target" / "release" / exe


def tbp_board(board: np.ndarray) -> list[list[str | None]]:
    """20 visible top-down rows -> 40 TBP bottom-up rows (row zero is bottom)."""
    if board.shape != (20, 10):
        raise ValueError("Expected a 20x10 settled-cell board")
    visible_bottom_up = [
        ["G" if bool(board[y, x]) else None for x in range(10)]
        for y in range(19, -1, -1)
    ]
    return visible_bottom_up + [[None] * 10 for _ in range(20)]


def tbp_cells(location: dict) -> tuple[tuple[int, int], ...]:
    """Convert a CC2 SRS-centered, bottom-up placement into visible coordinates."""
    name = location["type"]
    orientation = location["orientation"]
    if name not in CC2_CELLS or orientation not in ORIENTATIONS:
        raise ValueError("Unknown CC2 piece or orientation")
    x, y = int(location["x"]), int(location["y"])
    rotated = CC2_CELLS[name]
    for _ in range(ORIENTATIONS[orientation]):
        rotated = tuple((cy, -cx) for cx, cy in rotated)
    return tuple(sorted((x + dx, 19 - (y + dy)) for dx, dy in rotated))


def legal_root_for_suggestion(suggestion: dict, roots: list, current: str,
                              hold: str | None, next_queue: tuple[str, ...],
                              can_hold: bool, strict_spin=True):
    """Only accept a Cold Clear placement with a verified V5 current-piece route.

    roots are (v5._Move, uses_hold). This is deliberate: a T-spin cell layout
    alone does not prove an executable final rotation path.
    """
    loc = suggestion.get("location") or {}
    name = loc.get("type")
    if name not in PIECES:
        return None
    hold_used = name != current
    if hold_used and (not can_hold or name != (hold or (next_queue[0] if next_queue else None))):
        return None
    try:
        cells = tbp_cells(loc)
    except (KeyError, TypeError, ValueError):
        return None
    if len(set(cells)) != 4 or any(not (0 <= x < 10 and 0 <= y < 20) for x, y in cells):
        return None
    expected_spin = suggestion.get("spin", "none")
    candidates = [(move, used) for move, used in roots
                  if used == hold_used and move.name == name and move.cells == cells]
    if strict_spin and expected_spin in {"mini", "full"}:
        candidates = [(move, used) for move, used in candidates if move.spin == expected_spin]
    if not candidates:
        return None
    # For non-spin recommendations, avoid choosing a gratuitously longer path.
    return min(candidates, key=lambda entry: len(entry[0].actions))


class TBPProcess:
    """Synchronized, persistent newline-delimited JSON process.

    When requesting a fresh suggestion, polls until the engine has finished
    enough search to respond (Cold Clear 2 can ignore early suggest messages).
    Never shares a pipe read with the Qt UI thread.
    """
    def __init__(self, executable: Path, command: list[str] | None = None):
        self.executable = Path(executable)
        self.command = command
        self.proc = None
        self.messages = queue.Queue()
        self.lock = threading.RLock()
        self._reader = None

    def _read(self, proc):
        try:
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                    if isinstance(msg, dict):
                        self.messages.put(msg)
                except json.JSONDecodeError:
                    continue
        except (OSError, ValueError):
            pass

    def _send(self, msg):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Cold Clear 2 process exited unexpectedly")
        try:
            self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Cannot communicate with Cold Clear 2") from exc

    def _await(self, kind, deadline):
        while time.monotonic() < deadline:
            try:
                msg = self.messages.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if msg.get("type") == "error":
                raise RuntimeError("Cold Clear 2 error: " + str(msg))
            if msg.get("type") == kind:
                return msg
        raise TimeoutError(f"Cold Clear 2 did not send {kind} within the timeout")

    def _drain(self):
        while True:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                return

    def open(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self.close()
            if not self.executable.is_file():
                raise FileNotFoundError(f"Cold Clear 2 executable not found: {self.executable}")
            self.proc = subprocess.Popen(
                self.command or [str(self.executable)], cwd=str(self.executable.parent),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, encoding="utf-8",
            )
            self._reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
            self._reader.start()
            try:
                self._await("info", time.monotonic() + 4.0)
                self._send({"type": "rules"})
                self._await("ready", time.monotonic() + 4.0)
            except Exception:
                self.close()
                raise

    def query(self, board, current, queue_pieces, hold, *, combo=0, b2b=False,
              budget_ms=350):
        with self.lock:
            self.open()
            # Observed game state supersedes a previous speculative position.
            # Keep the PROCESS alive, but do not lie to CC2 by sending 'play'
            # when the human has not confirmed executing that suggested move.
            self._send({"type": "stop"})
            # TBP stop has no dedicated acknowledgement. Rules/ready is an
            # ordered protocol barrier: once ready arrives, stop and every old
            # suggest request were processed, so stale suggestions can be
            # discarded before the new board starts.
            self._send({"type": "rules"})
            self._await("ready", time.monotonic() + 1.0)
            self._drain()
            self._send({"type": "start", "board": tbp_board(board),
                        "queue": [current, *queue_pieces], "hold": hold,
                        "combo": max(0, int(combo)), "back_to_back": bool(b2b)})
            deadline = time.monotonic() + max(0.05, budget_ms / 1000)
            best = None
            next_poll = time.monotonic()
            # Keep the newest result through the requested budget. Accepting
            # the first ready root can return nodes=0 before CC2 has searched.
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now < next_poll:
                    time.sleep(min(next_poll - now, deadline - now))
                if time.monotonic() >= deadline:
                    break
                self._send({"type": "suggest"})
                poll_end = min(deadline, time.monotonic() + 0.03)
                try:
                    best = self._await("suggestion", poll_end)
                except TimeoutError:
                    pass
                next_poll = time.monotonic() + 0.04
            if best is None:
                raise TimeoutError("Cold Clear 2 did not produce a suggestion within budget")
            return best

    def close(self):
        with self.lock:
            proc, self.proc = self.proc, None
            if proc is None:
                return
            try:
                if proc.poll() is None:
                    proc.stdin.write('{"type":"quit"}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=0.5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                if proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                for stream in (proc.stdin, proc.stdout):
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
                self._drain()


_clients: dict[str, TBPProcess] = {}
_clients_lock = threading.Lock()


def get_client(executable: Path) -> TBPProcess:
    key = str(executable.resolve())
    with _clients_lock:
        if key not in _clients:
            _clients[key] = TBPProcess(executable)
        return _clients[key]


def close_engines():
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()


atexit.register(close_engines)


def find_best_cc2(board: np.ndarray, active, next_queue=(), hold=None,
                  can_hold=False, settings: SearchSettingsCC2 | None = None,
                  *, initial_b2b=0, initial_combo=-1) -> Optional[CC2Recommendation]:
    settings = settings or SearchSettingsCC2()
    if active.name not in PIECES:
        return None
    queue_pieces = tuple(p for p in (next_queue or ()) if p in PIECES)
    roots, states = v5._current_locks(board, active, settings.max_current_states)
    paired = [(move, False) for move in roots]
    if settings.allow_hold and can_hold is True:
        held = hold or (queue_pieces[0] if queue_pieces else None)
        if held:
            spawn = v5._make_spawn(held)
            if spawn:
                held_moves, more = v5._current_locks(board, spawn, settings.max_current_states)
                states += more
                paired.extend((m, True) for m in held_moves)
    if not paired:
        return None
    t0 = time.perf_counter()
    message = get_client(cc2_path(settings)).query(
        board, active.name, queue_pieces, hold,
        combo=max(0, initial_combo), b2b=bool(initial_b2b),
        budget_ms=settings.time_budget_ms,
    )
    for choice in message.get("moves") or ():
        root = legal_root_for_suggestion(choice, paired, active.name, hold,
                                         queue_pieces, can_hold is True and settings.allow_hold,
                                         settings.strict_spin)
        if root is None:
            continue
        move, hold_used = root
        actions = (("HOLD",) if hold_used else ()) + move.actions
        move_info = message.get("move_info") or {}
        return CC2Recommendation(
            name=move.name, rotation=move.r, x=move.x, y=move.y,
            cells=move.cells, score=0.0, cleared=move.lines, actions=actions,
            visited_states=states, spin=move.spin, depth_used=0,
            nodes_expanded=move_info.get("nodes", 0),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            hold_used=hold_used,
        )
    # Better no placement than rendering a potentially unreachable fake T-spin.
    return None


def describe_action_cc2(active, best):
    return "CC2: " + describe_action_v5(active, best)
