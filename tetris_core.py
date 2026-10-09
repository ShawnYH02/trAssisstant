"""Minimal computer-vision parser and single-piece hard-drop Tetris heuristic.

Assumptions: visible 10x20 board; default vividly colored tetromino skin;
no hold, next queue, SRS+ kicks, hidden rows, spins or move-path search.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

ROWS, COLS = 20, 10

# TETR.IO uses the familiar seven colored tetrominoes in its default skin.
# Coordinates here describe bounding-box-normalized *shapes*, not SRS pivots.
SPAWN = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((1, 0), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "J": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 0), (0, 1), (1, 1), (2, 1)),
}


def normalize(cells):
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def orientations(piece):
    """4 rotation *states*. Symmetric pieces have duplicate shapes."""
    cells = SPAWN[piece]
    result = []
    for _ in range(4):
        result.append(normalize(cells))
        cells = tuple((-y, x) for x, y in cells)  # clockwise in screen coords
    return result


ROTATIONS = {piece: orientations(piece) for piece in SPAWN}


def hue_to_piece(hue: float) -> str | None:
    """OpenCV HSV hue is in [0,179]; thresholds fit the default bright skin.
    Different palettes should be calibrated rather than forced into this model.
    """
    if hue < 9 or hue >= 174:
        return "Z"
    if hue < 22:
        return "L"
    if hue < 40:
        return "O"
    if hue < 79:
        return "S"
    if hue < 101:
        return "I"
    if hue < 136:
        return "J"
    if hue < 174:
        return "T"
    return None


def read_cells(bgra, saturation_min=72, value_min=70, gray_value_min=108,
               rows=ROWS):
    """Sample centers, not edges, to avoid gridlines and the outline overlay.

    Returns a rows-by-10 array of labels: '.', 'G', or I/O/T/S/Z/J/L.  Passing
    more than 20 rows lets callers include the hidden spawn area above the
    playfield.  This is intentionally a basic image heuristic, not general OCR.
    """
    h, w = bgra.shape[:2]
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR) if bgra.shape[2] == 4 else bgra
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    board = np.full((rows, COLS), ".", dtype="<U1")
    for row in range(rows):
        for col in range(COLS):
            # central 30% of each cell; skip thin borders and effects
            x0 = int((col + 0.35) * w / COLS)
            x1 = max(x0 + 1, int((col + 0.65) * w / COLS))
            y0 = int((row + 0.35) * h / rows)
            y1 = max(y0 + 1, int((row + 0.65) * h / rows))
            patch = hsv[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            hue, sat, val = np.median(patch.reshape(-1, 3), axis=0)
            if sat >= saturation_min and val >= value_min:
                piece = hue_to_piece(hue)
                if piece:
                    board[row, col] = piece
            elif val >= gray_value_min and sat < saturation_min:
                board[row, col] = "G"
    return board


def same_color_components(labels):
    """Return orthogonally-connected non-gray colored components."""
    rows, cols = labels.shape
    visited = set()
    for row in range(rows):
        for col in range(cols):
            color = labels[row, col]
            if color in (".", "G") or (col, row) in visited:
                continue
            component = []
            q = deque([(col, row)])
            visited.add((col, row))
            while q:
                x, y = q.popleft()
                component.append((x, y))
                for nx, ny in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                    if (0 <= nx < cols and 0 <= ny < rows
                        and (nx, ny) not in visited and labels[ny, nx] == color):
                        visited.add((nx, ny))
                        q.append((nx, ny))
            yield color, component


@dataclass(frozen=True)
class ActivePiece:
    name: str
    rotation: int  # 0,1,2,3; visually symmetric states may be ambiguous
    x: int         # normalized bounding-box leftmost column
    y: int         # normalized bounding-box top row
    cells: tuple[tuple[int, int], ...]


def find_active(labels):
    """Heuristic: highest intact same-colored tetromino component is falling.

    Returns None instead of guessing if the falling piece touches same-color
    locked blocks or partly lies above the visible board.
    """
    matches = []
    for name, cells in same_color_components(labels):
        if len(cells) != 4:
            continue
        shape = normalize(cells)
        try:
            rotation = ROTATIONS[name].index(shape)
        except ValueError:
            continue
        x, y = min(p[0] for p in cells), min(p[1] for p in cells)
        matches.append(ActivePiece(name, rotation, x, y, tuple(sorted(cells))))
    if not matches:
        return None
    # An intact current tetromino normally appears above the locked stack.
    # Equal-height ambiguous components are rejected instead of guessing.
    matches.sort(key=lambda candidate: (candidate.y, candidate.x))
    if len(matches) > 1 and matches[0].y == matches[1].y:
        return None
    return matches[0]


def occupied_without_active(labels, active):
    board = labels != "."
    for x, y in active.cells:
        if 0 <= y < board.shape[0] and 0 <= x < board.shape[1]:
            board[y, x] = False
    return board


def shift_active(active, dy):
    """Translate an active piece vertically between capture and board grids."""
    return ActivePiece(active.name, active.rotation, active.x, active.y + dy,
                       tuple((x, y + dy) for x, y in active.cells))


def can_place(board, shape, x, y):
    for dx, dy in shape:
        col, row = x + dx, y + dy
        if col < 0 or col >= COLS or row >= ROWS:
            return False
        if row >= 0 and board[row, col]:
            return False
    return True


def hard_drop(board, shape, x):
    """Find a hard-drop landing. None if it tops out or has no spawn space."""
    shape_h = max(y for _, y in shape) + 1
    y = -shape_h
    if not can_place(board, shape, x, y):
        return None
    while can_place(board, shape, x, y + 1):
        y += 1
    if y < 0:
        return None
    return y


def apply_placement(board, cells):
    result = board.copy()
    for x, y in cells:
        result[y, x] = True
    full = np.all(result, axis=1)
    cleared = int(np.sum(full))
    if cleared:
        remaining = result[~full]
        result = np.vstack((np.zeros((cleared, COLS), dtype=bool), remaining))
    return result, cleared


def score_board(board, cleared):
    """Basic single-move heuristic: smaller/cleaner surface + line rewards."""
    heights = []
    holes = 0
    for col in range(COLS):
        ys = np.flatnonzero(board[:, col])
        if len(ys) == 0:
            heights.append(0)
        else:
            first = int(ys[0])
            heights.append(ROWS - first)
            holes += int(np.count_nonzero(~board[first:, col]))
    bumpiness = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    return (
        9.0 * cleared + 2.0 * cleared * cleared
        - 0.54 * sum(heights)
        - 3.8 * holes
        - 0.35 * bumpiness
        - 0.4 * max(heights)
    )


@dataclass(frozen=True)
class Recommendation:
    name: str
    rotation: int
    x: int
    y: int
    cells: tuple[tuple[int, int], ...]
    score: float
    cleared: int


def find_best(board, name):
    """Search straight hard-drop landings in all 4 visual orientations."""
    best = None
    unique = set()
    for rotation, shape in enumerate(ROTATIONS[name]):
        if shape in unique:  # avoid duplicate shapes of symmetric pieces
            continue
        unique.add(shape)
        shape_w = max(x for x, _ in shape) + 1
        for x in range(COLS - shape_w + 1):
            y = hard_drop(board, shape, x)
            if y is None:
                continue
            cells = tuple(sorted((x + dx, y + dy) for dx, dy in shape))
            after, cleared = apply_placement(board, cells)
            score = score_board(after, cleared)
            candidate = Recommendation(name, rotation, x, y, cells, score, cleared)
            if best is None or candidate.score > best.score:
                best = candidate
    return best


def describe_action(active, best):
    """Compact finesse-style controls; SRS+ kicks are not modeled."""
    if best is None:
        return "No valid hard-drop placement"
    actions = finesse_actions(active, best)
    return f"{active.name}  |  " + " > ".join(actions)


def finesse_actions(active, best):
    """Find a minimum-key route in the prototype's bounding-box model.

    A tap, rotation, or DAS-to-wall each costs one key press. This provides
    useful finesse guidance but intentionally does not claim exact SRS+ kicks,
    collision-aware movement paths, or frame-perfect DAS timing.
    """
    if best is None:
        return ()

    start = (active.rotation % 4, active.x)
    goal = (best.rotation % 4, best.x)
    queue = deque([start])
    parent = {start: None}
    action_to = {}

    while queue:
        state = queue.popleft()
        if state == goal:
            break
        rotation, x = state
        transitions = []

        for label, turn in (("CW", 1), ("CCW", -1), ("180", 2)):
            new_rotation = (rotation + turn) % 4
            width = max(dx for dx, _ in ROTATIONS[active.name][new_rotation]) + 1
            if x + width <= COLS:
                transitions.append((label, (new_rotation, x)))

        width = max(dx for dx, _ in ROTATIONS[active.name][rotation]) + 1
        if x > 0:
            transitions.append(("Tap L", (rotation, x - 1)))
        if x + width < COLS:
            transitions.append(("Tap R", (rotation, x + 1)))
        if x != 0:
            transitions.append(("DAS L", (rotation, 0)))
        right_wall = COLS - width
        if x != right_wall:
            transitions.append(("DAS R", (rotation, right_wall)))

        for label, next_state in transitions:
            if next_state in parent:
                continue
            parent[next_state] = state
            action_to[next_state] = label
            queue.append(next_state)

    if goal not in parent:
        return ("Route unavailable", "Hard drop")

    route = []
    state = goal
    while parent[state] is not None:
        route.append(action_to[state])
        state = parent[state]
    route.reverse()
    route.append("Hard drop")
    return tuple(route)
