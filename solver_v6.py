"""V6 experimental attack-aware, root-diverse beam search for offline Tetris analysis.

Uses existing V5 reachability and S1-inspired attack simulator. Unlike V5,
this keeps multiple *first* choices alive and selects local continuations per
parent. It is not the Cold Clear/ZZZTOJ engine; strength needs held-out tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from heapq import nlargest
from collections import defaultdict
import time

import numpy as np
import solver_v5 as v5
from solver_v5 import RecommendationV5, to_rows, s1_reward, future_locks

FULL = 1023
W, H = 10, 20
# A row transition lookup avoids 20*10 Python branches per board evaluation.
_ROW_CHANGE = tuple((((r << 1) | 1) ^ (r | (1 << W))).bit_count()
                    for r in range(1 << W))


@dataclass(frozen=True)
class SearchSettingsV6:
    depth: int = 5
    beam_width: int = 24
    # Limit branching, not just width; avoids spending the entire deadline on
    # nodes with many placements or a reachable HOLD branch.
    continuations_per_parent: int = 5
    root_diversity: int = 10
    max_current_states: int = 3000
    time_budget_ms: float = 250.0
    allow_hold: bool = True


@lru_cache(maxsize=100_000)
def board_value(rows: tuple[int, ...]) -> float:
    """Nonlinear S1-oriented potential: height, structure, wells, holes, danger.

    Returns higher-is-better score. All scoring is *heuristic*. It does not
    assume a T-spin exists merely from a three-corner geometry.
    """
    heights = [0] * W
    covered_count = [0] * W
    seen = 0
    holes = 0
    hole_cover = 0
    row_trans = 0
    col_trans = 0
    # Top/bottom as occupied boundaries for vertical transition penalty.
    last = 0
    for y, row in enumerate(rows):
        row_trans += _ROW_CHANGE[row]
        col_trans += (last ^ row).bit_count()
        last = row
        introduced = row & ~seen
        while introduced:
            bit = introduced & -introduced
            x = bit.bit_length() - 1
            heights[x] = H - y
            introduced ^= bit
        seen |= row
        hole_bits = seen & ~row & FULL
        holes += hole_bits.bit_count()
        # Filled blocks *above* an empty cell: burial penalty, not simply
        # the same as the number of holes.
        b = hole_bits
        while b:
            bit = b & -b
            hole_cover += covered_count[bit.bit_length() - 1]
            b ^= bit
        b = row
        while b:
            bit = b & -b
            covered_count[bit.bit_length() - 1] += 1
            b ^= bit
    col_trans += (last ^ FULL).bit_count()
    total_height = sum(heights)
    peak = max(heights)
    rough = sum(abs(a-b) for a, b in zip(heights, heights[1:]))
    # Small internal pits may be fillable; deep narrow pits usually turn into
    # blockers. Side wells are rewarded only when the board is clean.
    internal_wells = 0
    for x in range(1, W-1):
        depth = max(0, min(heights[x-1], heights[x+1]) - heights[x])
        internal_wells += depth * (depth + 1) // 2
    side_depth = max(0, heights[1] - heights[0], heights[8] - heights[9])
    well_bonus = (2.8 * min(4, side_depth) - 0.45 * max(0, side_depth - 5) ** 2
                  if holes == 0 and peak <= 14 else 0.0)
    # A well that reaches beyond the safe height can become a death trap.
    danger = max(0, peak - 11)
    squared = sum(h * h for h in heights)
    return (well_bonus
            - 8.6 * holes
            - 0.78 * hole_cover
            - 0.17 * total_height
            - 0.017 * squared
            - 0.38 * rough
            - 0.25 * row_trans
            - 0.18 * col_trans
            - 0.32 * internal_wells
            - 1.40 * danger * danger)


def _root_id(n: v5._Node):
    return (n.first.name, n.first.cells, n.first.spin, n.first_hold)


def _prune(nodes: list[v5._Node], settings: SearchSettingsV6):
    """Transposition dedup + one survivor per promising first-action branch.

    Dedup happens before root diversification, so duplicate board states
    aren't given extra slots merely because they arrived via another route.
    """
    best_states = {}
    for node in nodes:
        key = (node.board, node.index, node.hold, node.b2b, node.combo)
        previous = best_states.get(key)
        if previous is None or node.estimate > previous.estimate:
            best_states[key] = node
    ranked = nlargest(settings.beam_width * 6, best_states.values(),
                      key=lambda node: node.estimate)
    chosen = []
    kept = set()
    # Enforce root coverage, including attacks that appear one ply behind.
    if settings.root_diversity > 0:
        seen_root = set()
        for node in ranked:
            root = _root_id(node)
            if root in seen_root:
                continue
            chosen.append(node)
            kept.add(id(node))
            seen_root.add(root)
            if len(chosen) >= min(settings.root_diversity, settings.beam_width):
                break
    for node in ranked:
        if len(chosen) >= settings.beam_width:
            break
        if id(node) not in kept:
            chosen.append(node)
            kept.add(id(node))
    return chosen


def _root_node(move, parent, first_hold, next_hold, next_index,
               depth_i, initial_b2b, initial_combo):
    # Extends V5's per-move outcome model, including the first action.
    if parent is None:
        prev_b2b, prev_combo, reward = initial_b2b, initial_combo, 0.0
    else:
        prev_b2b, prev_combo, reward = parent.b2b, parent.combo, parent.reward
    gain, new_b2b, new_combo = s1_reward(move.lines, move.spin,
                                         prev_b2b, prev_combo)
    if move.lines and not any(move.board):
        gain += 16.0  # Perfect clear, with no guessed combo or spin bonus.
    gain *= 0.94 ** depth_i
    reward += gain
    first = move if parent is None else parent.first
    held = first_hold if parent is None else parent.first_hold
    estimate = reward + board_value(move.board)
    return v5._Node(move.board, next_index, next_hold, new_b2b,
                    new_combo, reward, first, held, estimate)


def find_best_v6(board: np.ndarray, active, next_queue=(), hold=None,
                 can_hold=False, settings: SearchSettingsV6 | None = None,
                 *, initial_b2b: int = 0, initial_combo: int = -1
                 ) -> RecommendationV5 | None:
    """Five-piece anytime beam; first action uses collision-checked V5 BFS.

    Future pieces are still approximate. The recommended first action is
    always based on a current-piece legal route when the source pose is valid.
    """
    settings = settings or SearchSettingsV6()
    if (settings.depth < 1 or settings.beam_width < 1
            or settings.continuations_per_parent < 1):
        raise ValueError('Search depth, width and branch count must be positive')
    if active.name not in v5.PIECES or (hold is not None and hold not in v5.PIECES):
        return None
    next_queue = tuple(p for p in next_queue if p in v5.PIECES)
    start = time.perf_counter()
    deadline = start + max(10., settings.time_budget_ms) / 1000
    roots = []
    expanded = 0
    visited = 0
    current_moves, current_states = v5._current_locks(board, active,
                                                       settings.max_current_states)
    visited += current_states
    for move in current_moves:
        roots.append(_root_node(move, None, False, hold, 0, 0,
                                initial_b2b, initial_combo))
        expanded += 1
    if can_hold is True and settings.allow_hold:
        swap = hold or (next_queue[0] if next_queue else None)
        if swap:
            spawn = v5._make_spawn(swap)
            if spawn:
                other, count = v5._current_locks(board, spawn,
                                                  settings.max_current_states)
                visited += count
                for move in other:
                    roots.append(_root_node(move, None, True, active.name,
                                            0 if hold else 1, 0,
                                            initial_b2b, initial_combo))
                    expanded += 1
    if not roots:
        return None
    nodes = _prune(roots, settings)
    best = max(nodes, key=lambda n: n.estimate)
    used_depth = 1
    for d in range(1, min(settings.depth, len(next_queue) + 1)):
        if time.perf_counter() >= deadline:
            break
        candidates = []
        interrupted = False
        for parent in nodes:
            if time.perf_counter() >= deadline:
                interrupted = True
                break
            # Correct HOLD semantics: holding with empty slot consumes an
            # extra queue entry; holding a stored piece consumes only current.
            options = v5._recurse_options(parent, next_queue, settings.allow_hold)
            local = []
            for piece, new_index, new_hold in options:
                for move in future_locks(parent.board, piece):
                    local.append(_root_node(move, parent, False, new_hold,
                                             new_index, d, 0, -1))
                    expanded += 1
            # Only the strongest continuations PER parent are carried forward
            # before the global beam; this prevents one position with many
            # nearly equivalent futures from saturating the next frontier.
            candidates.extend(nlargest(settings.continuations_per_parent,
                                       local, key=lambda n: n.estimate))
        if interrupted or not candidates:
            break
        nodes = _prune(candidates, settings)
        if not nodes:
            break
        best = max(nodes, key=lambda n: n.estimate)
        used_depth = d + 1
    m = best.first
    actions = (('HOLD',) if best.first_hold else ()) + m.actions
    return RecommendationV5(m.name, m.r, m.x, m.y, m.cells,
                            best.estimate, m.lines, actions, visited,
                            m.spin, used_depth, expanded,
                            (time.perf_counter() - start) * 1000,
                            best.first_hold, best.b2b)


def describe_action_v6(active, best):
    return v5.describe_action_v5(active, best)
