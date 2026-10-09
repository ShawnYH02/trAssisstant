"""Collision-aware, single-piece Tetris move search for the trAssisstant overlay.

Drop-in addition: does NOT change image recognition, screen capture, or calibration.

Moves: left/right (tap or hold-to-obstacle), soft drop, CW/CCW with SRS
90-degree kicks, and hard drop. I uses the symmetric (Arika-style) I kick
variant as a close SRS+ approximation; custom 180 kicks and timing/gravity
are NOT modeled. No NEXT/HOLD search yet.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import groupby
from typing import Optional

import numpy as np

from tetris_core import ROWS, COLS, SPAWN, ActivePiece, apply_placement


# Rotations are held in the standard SRS 3x3 boxes (I: 4x4). Keeping the
# box origin across rotations is essential: rotating normalized bounding
# boxes around their top-left corner shifts pieces incorrectly.
BASE = {**SPAWN, "I": tuple((x, 1) for x in range(4))}
BOX_SIZE = {name: (4 if name == "I" else 2 if name == "O" else 3)
            for name in SPAWN}


def rotate_cw(shape, size):
    return tuple((size - 1 - y, x) for x, y in shape)


def piece_states(name):
    cells, size = BASE[name], BOX_SIZE[name]
    out = []
    for _ in range(4):
        out.append(tuple(sorted(cells)))
        cells = rotate_cw(cells, size)
    return tuple(out)


STATES = {name: piece_states(name) for name in BASE}
OFFSETS = {
    name: tuple((min(x for x, _ in cells), min(y for _, y in cells))
                for cells in states)
    for name, states in STATES.items()
}

# SRS guideline 90-degree kicks, with screen coordinates (positive y down).
# Source: https://tetris.wiki/Super_Rotation_System
# For I: use symmetric I kicks (Arika variant) close to TETR.IO's SRS+;
# unusual I kicks can differ from live-game behavior and need validation.
_J_UP = {
    (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
}
_I_UP = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (1, 2), (-2, -1)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    (2, 1): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -1)),
    (3, 2): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (3, 0): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (0, 3): ((0, 0), (2, 0), (-1, 0), (-1, 2), (2, -1)),
}
KICKS_J = {turn: tuple((dx, -dy) for dx, dy in kicks)
           for turn, kicks in _J_UP.items()}
KICKS_I = {turn: tuple((dx, -dy) for dx, dy in kicks)
           for turn, kicks in _I_UP.items()}


@dataclass(frozen=True)
class Pose:
    r: int
    bx: int
    by: int


@dataclass(frozen=True)
class RecommendationV2:
    name: str
    rotation: int
    x: int
    y: int
    cells: tuple[tuple[int, int], ...]
    score: float
    cleared: int
    actions: tuple[str, ...]
    visited_states: int


def pose_cells(name: str, pose: Pose):
    return tuple(sorted((pose.bx + x, pose.by + y)
                        for x, y in STATES[name][pose.r]))


def active_pose(active: ActivePiece) -> Pose:
    rotation = active.rotation % 4
    ox, oy = OFFSETS[active.name][rotation]
    return Pose(rotation, active.x - ox, active.y - oy)


def legal(board: np.ndarray, name: str, pose: Pose) -> bool:
    # Allow ordinary above-board spawns, but bound the otherwise unbounded
    # invisible ceiling. This is NOT a complete model of TETR.IO's hidden rows.
    cells = pose_cells(name, pose)
    for x, y in cells:
        if x < 0 or x >= COLS or y >= ROWS or y < -8:
            return False
        if y >= 0 and board[y, x]:
            return False
    return True


def rotate(board: np.ndarray, name: str, pose: Pose, direction: int) -> Optional[Pose]:
    if name == "O":
        return None  # O rotation has no useful visible action
    destination = (pose.r + direction) % 4
    table = KICKS_I if name == "I" else KICKS_J
    for dx, dy in table[(pose.r, destination)]:
        candidate = Pose(destination, pose.bx + dx, pose.by + dy)
        if legal(board, name, candidate):
            return candidate
    return None


def drop_pose(board: np.ndarray, name: str, pose: Pose) -> Pose:
    if not legal(board, name, pose):
        raise ValueError("Cannot drop an illegal pose")
    while True:
        nxt = Pose(pose.r, pose.bx, pose.by + 1)
        if not legal(board, name, nxt):
            return pose
        pose = nxt


def transitions(board: np.ndarray, name: str, p: Pose):
    """Generate real collision-checked paths, not hypothetical destinations."""
    for label, dx in (("Left", -1), ("Right", 1)):
        dest = Pose(p.r, p.bx + dx, p.by)
        if legal(board, name, dest):
            yield label, dest

    for label, dx in (("Hold left", -1), ("Hold right", 1)):
        dest = p
        while legal(board, name, Pose(dest.r, dest.bx + dx, dest.by)):
            dest = Pose(dest.r, dest.bx + dx, dest.by)
        if dest != p:
            yield label, dest

    for label, turn in (("CW", 1), ("CCW", -1)):
        dest = rotate(board, name, p, turn)
        if dest is not None and dest != p:
            yield label, dest

    dest = Pose(p.r, p.bx, p.by + 1)
    if legal(board, name, dest):
        yield "Soft drop", dest


def score_board_v2(board: np.ndarray, cleared: int) -> float:
    """Fast interpretable heuristic: cleaner / accessible stacks score higher.

    Weights are hand-tuned; evaluate against your own recorded boards rather
    than treating these as professionally optimized competitive weights.
    """
    heights = []
    holes = 0
    covered_depth = 0
    for col in range(COLS):
        filled = np.flatnonzero(board[:, col])
        if len(filled) == 0:
            heights.append(0)
            continue
        top = int(filled[0])
        heights.append(ROWS - top)
        below = board[top:, col]
        missing = np.flatnonzero(~below)
        holes += len(missing)
        # Count occupied cells above holes in this column (burial severity).
        for relative_row in missing:
            covered_depth += int(np.count_nonzero(below[:relative_row]))

    bumpiness = sum(abs(a-b) for a, b in zip(heights, heights[1:]))
    peak = max(heights)
    danger = max(0, peak - 12)
    wells = 0
    for i, h in enumerate(heights):
        left = heights[i-1] if i else ROWS
        right = heights[i+1] if i < COLS-1 else ROWS
        depth = max(0, min(left, right) - h)
        wells += depth * (depth + 1) // 2

    # Row transitions: occupied walls on either side count as filled; changing
    # between occupied/empty along a row means a surface that needs attention.
    row_transitions = 0
    for row in board:
        prev = True
        for occupied in row:
            cur = bool(occupied)
            row_transitions += cur != prev
            prev = cur
        row_transitions += not prev

    return (
        9.0 * cleared + 1.6 * cleared * cleared
        - 0.27 * sum(heights)
        - 7.0 * holes
        - 0.75 * covered_depth
        - 0.43 * bumpiness
        - 0.22 * wells
        - 0.10 * row_transitions
        - 0.32 * peak
        - 0.75 * danger * danger
    )


def find_best_v2(board: np.ndarray, active: ActivePiece,
                 max_states: int = 2500) -> Optional[RecommendationV2]:
    """Find reachable landing + shortest legal key path for one active piece.

    BFS models actions without gravity/timing; returned paths are collision-legal
    but may require playing fast enough for the current lock-delay setting.
    Only a 90-degree CW/CCW rotation table is used. No 180, NEXT, HOLD or spins.
    """
    if board.shape != (ROWS, COLS):
        raise ValueError(f"Expected {(ROWS, COLS)} boolean board, got {board.shape}")
    if active.name not in STATES:
        return None
    start = active_pose(active)
    if set(pose_cells(active.name, start)) != set(active.cells):
        # If vision produced an inconsistent piece orientation, do not guess.
        return None
    if not legal(board, active.name, start):
        return None

    queue = deque([start])
    routes = {start: ()}
    scored_cells = set()
    best = None

    while queue and len(routes) <= max_states:
        p = queue.popleft()
        path = routes[p]
        landing = drop_pose(board, active.name, p)
        cells = pose_cells(active.name, landing)
        if all(0 <= y < ROWS for _, y in cells) and cells not in scored_cells:
            scored_cells.add(cells)
            final, cleared = apply_placement(board, cells)
            # Tiny input-cost tie-breaker avoids needlessly complex routes.
            score = score_board_v2(final, cleared) - 0.025 * len(path)
            ox, oy = OFFSETS[active.name][landing.r]
            rec = RecommendationV2(active.name, landing.r,
                                   landing.bx + ox, landing.by + oy,
                                   cells, score, cleared,
                                   path + ("Hard drop",), len(routes))
            if best is None or rec.score > best.score:
                best = rec

        for label, nxt in transitions(board, active.name, p):
            if nxt not in routes and len(routes) < max_states:
                routes[nxt] = path + (label,)
                queue.append(nxt)

    if best is None:
        return None
    return RecommendationV2(best.name, best.rotation, best.x, best.y,
                            best.cells, best.score, best.cleared,
                            best.actions, len(routes))


def describe_action_v2(active: ActivePiece, best: Optional[RecommendationV2]) -> str:
    if best is None:
        return "No reachable landing found"
    labels = []
    for action, group in groupby(best.actions):
        n = sum(1 for _ in group)
        labels.append(f"{action} x{n}" if n > 1 else action)
    return f"{active.name} | " + " > ".join(labels)
