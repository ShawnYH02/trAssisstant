import numpy as np
import pytest

from tetris_core import ROWS, COLS, ActivePiece, ROTATIONS, apply_placement
from solver_v2 import (
    STATES, OFFSETS, Pose, active_pose, pose_cells, legal, drop_pose,
    transitions, find_best_v2, score_board_v2, describe_action_v2,
)


def empty():
    return np.zeros((ROWS, COLS), dtype=bool)


def active(name='T', r=0, x=3, y=0):
    # Manufacture a vision-like detected piece from its known bounding box.
    ox, oy = OFFSETS[name][r]
    cells = pose_cells(name, Pose(r, x-ox, y-oy))
    return ActivePiece(name, r, x, y, cells)


@pytest.mark.parametrize('name', list(STATES))
def test_solver_geometry_agrees_with_existing_detection(name):
    for r in range(4):
        cells = STATES[name][r]
        offset = OFFSETS[name][r]
        normalized = tuple(sorted((x-offset[0], y-offset[1]) for x, y in cells))
        assert normalized == ROTATIONS[name][r]


def test_clear_line_and_replay_moves():
    board = empty()
    board[-1, :6] = True
    a = active('I', 0, 3, 2)
    best = find_best_v2(board, a)
    assert best is not None
    assert best.cleared == 1
    assert set(best.cells) == {(6,19), (7,19), (8,19), (9,19)}
    assert best.actions[-1] == 'Hard drop'
    replay(board, a, best)


def replay(board, a, rec):
    pose = active_pose(a)
    for label in rec.actions[:-1]:
        possibilities = dict(transitions(board, a.name, pose))
        assert label in possibilities, f'{label} cannot be played at {pose}'
        pose = possibilities[label]
        assert legal(board, a.name, pose)
    assert rec.actions[-1] == 'Hard drop'
    landed = drop_pose(board, a.name, pose)
    assert pose_cells(a.name, landed) == rec.cells


@pytest.mark.parametrize('name', list(STATES))
def test_replay_from_mixed_stack(name):
    board = empty()
    board[-1, 0:3] = True
    board[-2, 2] = True
    a = active(name, 0, 3, 1)
    best = find_best_v2(board, a)
    assert best is not None
    replay(board, a, best)


def test_blocked_side_unreachable_from_current_pose():
    board = empty()
    board[8:, 4] = True  # full-height vertical barrier from below row 8
    a = active('O', 0, 0, 16)  # trapped left of the barrier
    best = find_best_v2(board, a)
    assert best is not None
    assert all(x < 4 for x, _ in best.cells)
    replay(board, a, best)


def test_holes_worse_than_clean():
    clean = empty()
    clean[17:20, 3] = True
    hole = clean.copy()
    hole[18, 3] = False
    assert score_board_v2(clean, 0) > score_board_v2(hole, 0)


def test_occupied_active_pose_rejected():
    board = empty()
    board[2, 3] = True
    assert find_best_v2(board, active('O', 0, 3, 2)) is None


def test_instructions_have_ending():
    board = empty()
    a = active('T')
    best = find_best_v2(board, a)
    assert 'Hard drop' in describe_action_v2(a, best)


def test_t_wall_kick_near_left_wall():
    from solver_v2 import rotate
    board = empty()
    p = Pose(1, -1, 4)   # T in R orientation, bounding box reaches column zero
    assert legal(board, 'T', p)
    kicked = rotate(board, 'T', p, -1)  # R -> 0: requires shifting right
    assert kicked is not None
    assert kicked.r == 0
    assert kicked.bx == 0
    assert legal(board, 'T', kicked)


def test_non_empty_path_with_soft_drop_tuck():
    # A roof and an opening: every suggested step must remain collision legal.
    board = empty()
    board[16, 2:8] = True
    board[16, 4:6] = False
    a = active('T', 0, 3, 4)
    best = find_best_v2(board, a)
    assert best is not None
    replay(board, a, best)
