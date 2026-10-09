"""TETR.IO Season-1-inspired lookahead solver for trAssisstant.

Standalone additive upgrade: uses V2's collision-checked paths for the
CURRENT piece and fast straight-drop positions for future pieces. This is an
advisor/analysis heuristic, not a frame-exact SRS+ engine or a rules emulator.

S1: only T-spins score as spins, Quads and T-spins preserve B2B, no Surge.
Depth 5 is the default with beam pruning, transposition merging and deadline.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import groupby
import math
import time
from typing import Optional

import numpy as np

from tetris_core import ActivePiece
from solver_v2 import (STATES, OFFSETS, KICKS_J, Pose, active_pose, pose_cells,
                       legal, rotate, drop_pose, transitions)

W, H, FULL = 10, 20, (1 << 10) - 1
PIECES = frozenset(STATES)


@dataclass(frozen=True)
class SearchSettings:
    depth: int = 5
    beam_width: int = 30
    max_current_states: int = 3000
    time_budget_ms: float = 190.0
    allow_hold: bool = True
    # Reduced only if time budget expires; intermediate candidates are retained.


@dataclass(frozen=True)
class RecommendationV5:
    name: str
    rotation: int
    x: int
    y: int
    cells: tuple[tuple[int, int], ...]
    score: float
    cleared: int
    actions: tuple[str, ...]
    visited_states: int
    spin: str = "none"  # none, mini, full
    depth_used: int = 1
    nodes_expanded: int = 0
    elapsed_ms: float = 0.0
    hold_used: bool = False
    b2b_after: int = 0


@dataclass(frozen=True)
class _Move:
    board: tuple[int, ...]
    name: str
    r: int
    x: int
    y: int
    cells: tuple[tuple[int, int], ...]
    lines: int
    spin: str
    actions: tuple[str, ...] = ()


@dataclass
class _Node:
    board: tuple[int, ...]
    index: int
    hold: str | None
    b2b: int
    combo: int
    reward: float
    first: _Move
    first_hold: bool
    estimate: float


def to_rows(board: np.ndarray) -> tuple[int, ...]:
    if board.shape != (H, W):
        raise ValueError(f"Expected {(H, W)} board, got {board.shape}")
    return tuple(sum((1 << x) for x in range(W) if board[y, x]) for y in range(H))


def from_rows(rows: tuple[int, ...]) -> np.ndarray:
    return np.array([[(r >> x) & 1 for x in range(W)] for r in rows], dtype=bool)


def occupied(rows: tuple[int, ...], x: int, y: int) -> bool:
    return x < 0 or x >= W or y >= H or (y >= 0 and bool(rows[y] & (1 << x)))


def corner_tspin(rows: tuple[int, ...], pose: Pose, *, last_rotation: bool, kick_upgrade: bool = False) -> str:
    """Conservative S1 corner recognition using board BEFORE locking a T.

    T pivot is center of its 3x3 rotation box. T-spins require a final rotation
    at the landing pose and three occupied diagonal corners. Mini if only one
    'front' corner is filled. 5th kick exceptions intentionally excluded until
    verified against exact TETR.IO SRS+ tables.
    """
    if not last_rotation:
        return "none"
    cx, cy = pose.bx + 1, pose.by + 1
    corners = {(-1, -1): occupied(rows, cx - 1, cy - 1),
               (1, -1): occupied(rows, cx + 1, cy - 1),
               (-1, 1): occupied(rows, cx - 1, cy + 1),
               (1, 1): occupied(rows, cx + 1, cy + 1)}
    if sum(corners.values()) < 3:
        return "none"
    front = {0: ((-1, -1), (1, -1)), 1: ((1, -1), (1, 1)),
             2: ((-1, 1), (1, 1)), 3: ((-1, -1), (-1, 1))}[pose.r]
    return "full" if all(corners[c] for c in front) or kick_upgrade else "mini"


def lock(rows: tuple[int, ...], cells: tuple[tuple[int, int], ...]):
    if any(not (0 <= x < W and 0 <= y < H) or occupied(rows, x, y) for x, y in cells):
        return None
    out = list(rows)
    for x, y in cells:
        out[y] |= 1 << x
    kept = [r for r in out if r != FULL]
    cleared = H - len(kept)
    return (0,) * cleared + tuple(kept), cleared


@lru_cache(maxsize=12000)
def eval_board(rows: tuple[int, ...]) -> float:
    heights = []
    holes = 0
    covered = 0
    for x in range(W):
        top = next((y for y in range(H) if rows[y] & (1 << x)), H)
        heights.append(H - top)
        n_above = 0
        for y in range(top, H):
            if rows[y] & (1 << x):
                n_above += 1
            else:
                holes += 1
                covered += n_above
    peak = max(heights)
    rough = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    row_changes = sum((((r << 1) | 1) ^ (r | (1 << W))).bit_count() for r in rows)
    # Allow an accessible side well without rewarding deeply buried cavities.
    side_well = max(0, heights[1] - heights[0], heights[-2] - heights[-1])
    side_bonus = 0.22 * min(5, side_well) if holes == 0 and peak < 15 else 0.0
    danger = max(0, peak - 12)
    return (-0.31 * sum(heights) - 8.0 * holes - 0.5 * covered
            - 0.47 * rough - 0.37 * peak - 1.0 * danger * danger
            - 0.06 * row_changes + side_bonus)


def s1_reward(lines: int, spin: str, b2b: int = 0, combo: int = -1):
    """Approximate S1 attack-oriented value, NOT an exact attack simulator.

    Returns (reward, updated_b2b, updated_combo). The B2B threshold tiers
    follow S1 chaining; S2 Surge is deliberately absent.
    """
    if not 0 <= lines <= 4:
        raise ValueError("Only 0..4 line clears are supported")
    if spin not in {"none", "mini", "full"}:
        raise ValueError("Unknown spin type")
    base = ({0: 0, 1: 0, 2: 1, 3: 2, 4: 4}[lines] if spin == "none"
            else ({0: 0, 1: 1, 2: 2}.get(lines, 0) if spin == "mini"
                  else {0: 0, 1: 2, 2: 4, 3: 6}.get(lines, 0)))
    combo_new = combo + 1 if lines else -1
    difficult = lines == 4 or (spin != "none" and lines > 0)
    b2b_new = (b2b + 1 if difficult else 0 if lines else b2b)
    # Based on the published B2B chaining tier ranges, but value is tunable.
    bonus = (0 if b2b_new < 2 else 1 if b2b_new < 4
             else 2 if b2b_new < 9 else 3 if b2b_new < 25 else 4)
    attack = int(base * (1 + .25 * max(0, combo_new))) + (bonus if difficult else 0)
    # B2B maintenance with zero clear does not break chain.
    reward = 2.5 * attack + .75 * lines
    if spin != "none" and lines:
        reward += 0.45 if spin == "mini" else 1.1
    if difficult:
        reward += .7  # favor preserving difficult-clear chain, not greedily
    if lines and not difficult and b2b > 0:
        reward -= min(3.0, .5 * b2b)
    return reward, b2b_new, combo_new


def _canonical_rotations(name: str):
    seen = set()
    for r in range(4):
        shape = STATES[name][r]
        ox = min(x for x, _ in shape)
        oy = min(y for _, y in shape)
        norm = tuple(sorted((x - ox, y - oy) for x, y in shape))
        if norm not in seen:
            seen.add(norm)
            yield r, norm


_CANON = {p: tuple(_canonical_rotations(p)) for p in PIECES}


def _can_place(rows: tuple[int, ...], cells, x: int, y: int):
    return all(not occupied(rows, x + dx, y + dy) and y + dy >= -4 for dx, dy in cells)


def _tops(rows):
    """Highest occupied row in each column (H for an empty column)."""
    seen = [H] * W
    for y, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            x = bit.bit_length() - 1
            if seen[x] == H:
                seen[x] = y
            bits -= bit
    return seen


def _landing_box_y(tops, bx, shape):
    """Fast vertical landing computed from the first blocking block/column."""
    low = {}
    for dx, dy in shape:
        if bx + dx < 0 or bx + dx >= W:
            return None
        low[dx] = max(dy, low.get(dx, -99))
    return min(tops[bx + dx] - dy - 1 for dx, dy in low.items())


@lru_cache(maxsize=2000)
def future_locks(rows: tuple[int, ...], name: str) -> tuple[_Move, ...]:
    """Fast future drops + conservative real final-rotation T-spin options."""
    moves = []
    seen = set()
    tops = _tops(rows)
    for r, norm in _CANON[name]:
        width = max(x for x, _ in norm) + 1
        height = max(y for _, y in norm) + 1
        for x in range(W - width + 1):
            y = _landing_box_y(tops, x, norm)
            if y is None or y < 0:
                continue
            cells = tuple(sorted((x + dx, y + dy) for dx, dy in norm))
            if cells in seen:
                continue
            seen.add(cells)
            result = lock(rows, cells)
            if result is None:
                continue
            after, cleared = result
            moves.append(_Move(after, name, r, x, y, cells, cleared, "none"))
    if name == "T":
        # A legal final CW/CCW rotation from a straight-drop-accessible pose.
        # Limit predecessors to the four rows above each hard drop: 3-corner
        # spins require neighboring stack cells, not rotations high in the air.
        for prev_r in range(4):
            shape = STATES["T"][prev_r]
            for bx in range(-2, W):
                y_end = _landing_box_y(tops, bx, shape)
                if y_end is None or y_end < -3:
                    continue
                for y in range(max(-3, y_end - 4), y_end + 1):
                    prev = Pose(prev_r, bx, y)
                    if not _pose_legal_rows(rows, "T", prev):
                        continue
                    for direction in (-1, 1):
                        dest = _rotate_rows(rows, prev, direction)
                        if dest is None or _pose_legal_rows(rows, "T", Pose(dest.r, dest.bx, dest.by + 1)):
                            continue
                        spin = corner_tspin(rows, dest, last_rotation=True)
                        if spin == "none":
                            continue
                        cells = pose_cells("T", dest)
                        if not all(0 <= cy < H for _, cy in cells):
                            continue
                        result = lock(rows, cells)
                        if result is None:
                            continue
                        after, cleared = result
                        key = (cells, spin)
                        if key in seen:
                            continue
                        seen.add(key)
                        ox, oy = OFFSETS["T"][dest.r]
                        moves.append(_Move(after, "T", dest.r, dest.bx + ox,
                                           dest.by + oy, cells, cleared, spin))
    return tuple(moves)


def _pose_legal_rows(rows, name, pose):
    return all(not occupied(rows, x, y) and y >= -4
               for x, y in pose_cells(name, pose))


def _rotate_rows(rows, pose, direction):
    r = (pose.r + direction) % 4
    for dx, dy in KICKS_J[(pose.r, r)]:
        new = Pose(r, pose.bx + dx, pose.by + dy)
        if _pose_legal_rows(rows, "T", new):
            return new
    return None


def transitions_v5(board: np.ndarray, name: str, pose: Pose):
    """V2 collision-aware moves plus conservative 180 (no kick)."""
    yield from transitions(board, name, pose)
    if name != "O":
        rotated = Pose((pose.r + 2) % 4, pose.bx, pose.by)
        if rotated != pose and legal(board, name, rotated):
            yield "180", rotated


def _current_locks(board: np.ndarray, active: ActivePiece, max_states: int):
    name = active.name
    if name not in PIECES:
        return [], 0
    start = active_pose(active)
    if (set(pose_cells(name, start)) != set(active.cells)
            or not legal(board, name, start)):
        return [], 0
    rows = to_rows(board)
    # Keep last-rotation provenance in the search key, otherwise BFS can
    # discard a T-spin route as a duplicate of a non-spin path.
    q = deque([(start, False)])
    routes = {(start, False): ()}
    outcomes = {}
    while q and len(routes) < max_states:
        pose, last_rot = q.popleft()
        route = routes[(pose, last_rot)]
        landing = drop_pose(board, name, pose)
        cells = pose_cells(name, landing)
        if all(0 <= y < H for _, y in cells):
            spin = (corner_tspin(rows, landing, last_rotation=last_rot and pose == landing)
                    if name == "T" else "none")
            key = (cells, spin)
            if key not in outcomes:
                result = lock(rows, cells)
                if result is not None:
                    after, cleared = result
                    ox, oy = OFFSETS[name][landing.r]
                    outcomes[key] = _Move(after, name, landing.r,
                                         landing.bx + ox, landing.by + oy,
                                         cells, cleared, spin, route + ("Hard drop",))
        for label, nxt in transitions_v5(board, name, pose):
            # "Hold left/right" in V2 means a DAS-style movement, NOT HOLD piece.
            key = (nxt, label in ("CW", "CCW", "180"))
            if key not in routes and len(routes) < max_states:
                routes[key] = route + (label,)
                q.append(key)
    return list(outcomes.values()), len(routes)


def _best_nodes(nodes: list[_Node], width: int) -> list[_Node]:
    best_by_key = {}
    for n in nodes:
        k = (n.board, n.index, n.hold, n.b2b, n.combo)
        prev = best_by_key.get(k)
        if prev is None or prev.estimate < n.estimate:
            best_by_key[k] = n
    return sorted(best_by_key.values(), key=lambda n: n.estimate, reverse=True)[:width]


def _consider(move: _Move, parent: _Node | None, first_hold: bool,
              new_hold: str | None, index: int, depth_index: int):
    b2b, combo, total = (parent.b2b, parent.combo, parent.reward) if parent else (0, -1, 0.0)
    reward, b2b_next, combo_next = s1_reward(move.lines, move.spin, b2b, combo)
    # Future gains are discounted lightly to prefer near-term realizable value.
    total += reward * (0.92 ** depth_index)
    first = parent.first if parent else move
    hold_was_used = parent.first_hold if parent else first_hold
    estimate = total + eval_board(move.board)
    return _Node(move.board, index, new_hold, b2b_next, combo_next,
                 total, first, hold_was_used, estimate)


def _recurse_options(node: _Node, next_pieces: tuple[str, ...], allow_hold: bool):
    if node.index >= len(next_pieces):
        return
    natural = next_pieces[node.index]
    yield natural, node.index + 1, node.hold
    if allow_hold:
        if node.hold is not None:
            yield node.hold, node.index + 1, natural
        elif node.index + 1 < len(next_pieces):
            yield next_pieces[node.index + 1], node.index + 2, natural


def find_best_v5(board: np.ndarray, active: ActivePiece,
                 next_queue=(), hold: str | None = None,
                 can_hold: bool | None = False,
                 settings: SearchSettings | None = None,
                 *, initial_b2b: int = 0, initial_combo: int = -1
                 ) -> Optional[RecommendationV5]:
    """Return a collision-checked current path with up-to-5-piece S1 planning.

    `next_queue` excludes current, in order of appearance. `can_hold=None` is
    considered False for safety. Planned HOLD directions are advisory only.
    """
    settings = settings or SearchSettings()
    if settings.depth < 1 or settings.beam_width < 1:
        raise ValueError("depth and beam_width must be positive")
    if active.name not in PIECES or hold is not None and hold not in PIECES:
        return None
    next_queue = tuple(p for p in next_queue if p in PIECES)
    start_time = time.perf_counter()
    deadline = start_time + max(10., settings.time_budget_ms) / 1000
    nodes = []
    expanded = 0
    total_states = 0
    current_moves, states = _current_locks(board, active, settings.max_current_states)
    total_states += states
    for move in current_moves:
        n = _consider(move, None, False, hold, 0, 0)
        # update from the observed combo/B2B state, if available
        if initial_b2b or initial_combo != -1:
            reward, b2b, combo = s1_reward(move.lines, move.spin, initial_b2b, initial_combo)
            n.b2b, n.combo, n.reward = b2b, combo, reward
            n.estimate = reward + eval_board(n.board)
        nodes.append(n)
        expanded += 1
    if can_hold is True and settings.allow_hold:
        swap_piece = hold or (next_queue[0] if next_queue else None)
        if swap_piece:
            spawn = _make_spawn(swap_piece)
            if spawn is not None:
                other, nstates = _current_locks(board, spawn, settings.max_current_states)
                total_states += nstates
                for move in other:
                    n = _consider(move, None, True, active.name,
                                  0 if hold else 1, 0)
                    if initial_b2b or initial_combo != -1:
                        reward, b2b, combo = s1_reward(move.lines, move.spin,
                                                       initial_b2b, initial_combo)
                        n.b2b, n.combo, n.reward = b2b, combo, reward
                        n.estimate = reward + eval_board(n.board)
                    nodes.append(n)
                    expanded += 1
    if not nodes:
        return None
    nodes = _best_nodes(nodes, settings.beam_width)
    best = nodes[0]
    depth_used = 1
    # The initial piece counts as depth 1; queue supplies future pieces.
    for d in range(1, min(settings.depth, len(next_queue) + 1)):
        if time.perf_counter() >= deadline:
            break
        candidates = []
        cut = False
        for node in nodes:
            if time.perf_counter() >= deadline:
                cut = True
                break
            for name, nxt_index, nxt_hold in _recurse_options(
                    node, next_queue, settings.allow_hold):
                for move in future_locks(node.board, name):
                    candidates.append(_consider(move, node, False, nxt_hold, nxt_index, d))
                    expanded += 1
        if cut or not candidates:
            break  # Don't compare incomplete expansions with complete depths.
        nodes = _best_nodes(candidates, settings.beam_width)
        best = nodes[0]
        depth_used = d + 1
    elapsed = (time.perf_counter() - start_time) * 1000
    m = best.first
    actions = (("HOLD",) if best.first_hold else ()) + m.actions
    return RecommendationV5(m.name, m.r, m.x, m.y, m.cells,
                             best.estimate, m.lines, actions, total_states,
                             m.spin, depth_used, expanded, elapsed,
                             best.first_hold, best.b2b)


def _make_spawn(name: str) -> Optional[ActivePiece]:
    # An approximate standard 10-wide spawn; used only for a *hypothetical*
    # held piece. V2 validates collision and all resulting action paths.
    r = 0
    shape = STATES[name][0]
    bx, by = 3, -2
    cells = pose_cells(name, Pose(r, bx, by))
    ox, oy = OFFSETS[name][r]
    return ActivePiece(name, r, bx + ox, by + oy, cells)


def describe_action_v5(active: ActivePiece, best: Optional[RecommendationV5]):
    if best is None:
        return "No reachable placement found"
    labels = []
    for act, group in groupby(best.actions):
        n = len(tuple(group))
        labels.append(f"{act} x{n}" if n > 1 else act)
    feature = (f" | T-spin {best.spin}" if best.spin != "none" else
               " | QUAD" if best.cleared == 4 else "")
    return f"{best.name}{feature}: " + " > ".join(labels)
