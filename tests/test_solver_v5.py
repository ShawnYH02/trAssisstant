import math
import numpy as np
import pytest
from tetris_core import ActivePiece
from solver_v2 import Pose, OFFSETS, pose_cells, active_pose, transitions, drop_pose
from solver_v5 import (SearchSettings, to_rows, from_rows, occupied, corner_tspin,
                       s1_reward, future_locks, find_best_v5, eval_board, transitions_v5)


def make_active(name='T', r=0, x=3, y=-2):
    cells = pose_cells(name, Pose(r, x, y))
    return ActivePiece(name, r, min(xx for xx, yy in cells),
                       min(yy for xx, yy in cells), cells)


def zero():
    return np.zeros((20,10), dtype=bool)


def replay(b, a, rec):
    if rec.hold_used:
        # New held piece starts from a fresh spawn; replay this when testing
        # a move WITHOUT HOLD only.
        return
    pose = active_pose(a)
    for label in rec.actions[:-1]:
        legal_moves = dict(transitions_v5(b, a.name, pose))
        assert label in legal_moves
        pose = legal_moves[label]
    landing = drop_pose(b, a.name, pose)
    assert set(rec.cells) == set(pose_cells(a.name, landing))


def test_rows_roundtrip():
    b = zero(); b[5, 7] = True; b[19, 0] = True
    assert np.array_equal(from_rows(to_rows(b)), b)


def test_tspin_corners_require_rotation():
    b = zero()
    # T pivot at (4,17), facing up, both front corners + one back corner.
    b[16,3] = b[16,5] = b[18,3] = True
    rows = to_rows(b)
    assert corner_tspin(rows, Pose(0,3,16), last_rotation=True) == 'full'
    assert corner_tspin(rows, Pose(0,3,16), last_rotation=False) == 'none'
    b[16,5] = False
    assert corner_tspin(to_rows(b), Pose(0,3,16), last_rotation=True) == 'none'


def test_tspin_mini_front_corners():
    b = zero(); b[16,3] = b[18,3] = b[18,5] = True
    assert corner_tspin(to_rows(b), Pose(0,3,16), last_rotation=True) == 'mini'


def test_s1_b2b_chain_no_surge():
    _, chain, combo = s1_reward(4,'none',0,-1)
    assert chain == 1 and combo == 0
    _, chain, combo = s1_reward(4,'none',chain,combo)
    assert chain == 2 and combo == 1
    _, same_chain, _ = s1_reward(0,'none',chain,combo)
    assert same_chain == chain
    _, reset_chain, _ = s1_reward(2,'none',chain,combo)
    assert reset_chain == 0
    full, _, _ = s1_reward(2,'full',0,-1)
    basic, _, _ = s1_reward(2,'none',0,-1)
    assert full > basic


def test_future_t_spin_not_awarded_by_shape_only():
    b = zero()
    assert all(m.spin == 'none' for m in future_locks(to_rows(b), 'T'))


def test_straight_quad_gets_rewarded():
    b = zero(); b[-4:, 1:] = True
    a = make_active('I')
    rec = find_best_v5(b,a,(), settings=SearchSettings(depth=1,beam_width=30))
    assert rec is not None
    assert rec.cleared == 4
    assert rec.cells == ((0,16),(0,17),(0,18),(0,19))
    replay(b,a,rec)


def test_five_piece_search_and_path():
    b = zero()
    a = make_active('T')
    rec = find_best_v5(b,a,('I','O','L','J','S'),
                settings=SearchSettings(depth=5,beam_width=30,time_budget_ms=1500))
    assert rec is not None
    assert rec.depth_used == 5
    assert rec.nodes_expanded > 500
    assert not rec.hold_used
    replay(b,a,rec)


def test_legal_hold_use_when_permitted():
    b = zero(); b[-4:,1:] = True
    rec = find_best_v5(b,make_active('T'),('I','L','S','Z','O'),
                hold=None,can_hold=True,
                settings=SearchSettings(depth=1,time_budget_ms=1500))
    assert rec is not None and rec.hold_used
    assert rec.name == 'I'
    assert rec.actions[0] == 'HOLD'
    assert rec.cleared == 4


def test_no_hold_when_unavailable():
    rec = find_best_v5(zero(),make_active('T'),('I','O'),
        hold='I',can_hold=False,settings=SearchSettings(depth=1))
    assert rec and not rec.hold_used


def test_evaluation_penalizes_holes():
    b = zero(); b[-3:,3] = True
    clean = eval_board(to_rows(b))
    b[-2,3] = False
    assert eval_board(to_rows(b)) < clean


def test_invalid_pose_rejected():
    b = zero(); b[0,4] = True
    rec = find_best_v5(b,make_active('T',y=0),(),
             settings=SearchSettings(depth=1))
    assert rec is None


def test_180_only_when_collision_legal():
    p = Pose(0, 3, 1)
    b = zero()
    assert '180' in dict(transitions_v5(b, 'T', p))
    # Block only a destination cell that isn't covered by the original pose.
    b[3, 4] = True
    assert '180' not in dict(transitions_v5(b, 'T', p))


def test_tspin_is_available_to_current_piece_when_rotated_last():
    from solver_v5 import _current_locks
    b = zero()
    b[16, 3] = b[16, 5] = b[18, 3] = True
    moves, _ = _current_locks(b, make_active('T'), 3000)
    assert any(m.spin == 'full' for m in moves)
    assert all(m.actions[-1] == 'Hard drop' for m in moves)


def test_fast_future_drop_matches_naive():
    from tetris_core import hard_drop
    from solver_v5 import _CANON
    rng = np.random.default_rng(123)
    for _ in range(12):
        b = zero()
        for col in range(10):
            h = int(rng.integers(0, 8))
            if h:
                b[-h:, col] = True
        rows = to_rows(b)
        for name in ('I','O','T','J','L','S','Z'):
            expected = set()
            for r, shape in _CANON[name]:
                width = max(x for x,y in shape) + 1
                for x in range(10-width+1):
                    y = hard_drop(b,shape,x)
                    if y is not None:
                        expected.add(tuple(sorted((x+dx,y+dy) for dx,dy in shape)))
            actual = {m.cells for m in future_locks(rows,name) if m.spin == 'none'}
            assert actual == expected, name
