import numpy as np
from tetris_core import ActivePiece
from solver_v2 import Pose, pose_cells, active_pose, drop_pose
from solver_v5 import to_rows, s1_reward
from solver_v6 import (SearchSettingsV6, board_value, find_best_v6,
                       _prune, _root_id)
import solver_v5 as v5


def make_active(name='T'):
    pose = Pose(0, 3, -2)
    cells = pose_cells(name, pose)
    return ActivePiece(name, 0, min(x for x,y in cells),
                       min(y for x,y in cells), cells)


def test_board_value_discourages_buried_holes():
    b = np.zeros((20, 10), dtype=bool)
    b[17:, 4] = True
    clean = board_value(to_rows(b))
    b[18, 4] = False
    assert board_value(to_rows(b)) < clean


def test_board_value_recognizes_clean_side_well():
    b = np.zeros((20, 10), dtype=bool)
    b[-4:, :] = True
    b[-4:, 0] = False
    assert board_value(to_rows(b)) > board_value(to_rows(np.ones((20,10),bool)))


def test_quad_and_replay():
    b = np.zeros((20, 10), bool)
    b[-4:, 1:] = True
    active = make_active('I')
    answer = find_best_v6(b, active, ('T','J','L','S','O'),
                settings=SearchSettingsV6(depth=1,time_budget_ms=1000))
    assert answer is not None
    assert answer.cleared == 4
    p = active_pose(active)
    for action in answer.actions[:-1]:
        p = dict(v5.transitions_v5(b, active.name, p))[action]
    landed = drop_pose(b,active.name,p)
    assert set(answer.cells) == set(pose_cells(active.name, landed))


def test_depth_and_hold():
    b = np.zeros((20,10),bool)
    active = make_active('T')
    answer = find_best_v6(b,active,('O','I','L','J','S'),hold='Z',can_hold=True,
               settings=SearchSettingsV6(depth=5,beam_width=15,
                   time_budget_ms=4000))
    assert answer
    assert answer.depth_used == 5
    assert answer.name in ('T','Z')
    assert answer.actions[-1] == 'Hard drop'
    assert answer.elapsed_ms > 0
    assert answer.next_name in {'I','L','J','S','O','T','Z'}
    assert answer.next_cells is not None
    after_first = v5.lock(to_rows(b), answer.cells)
    assert after_first is not None
    assert v5.lock(after_first[0], answer.next_cells) is not None
    after_second = v5.lock(after_first[0], answer.next_cells)
    assert after_second is not None
    assert answer.third_name in {'I','L','J','S','O','T','Z'}
    assert answer.third_cells is not None
    assert v5.lock(after_second[0], answer.third_cells) is not None


def test_no_hold_when_cooldown():
    answer = find_best_v6(np.zeros((20,10),bool), make_active('L'),
        ('T', 'I'), hold='I', can_hold=False,
        settings=SearchSettingsV6(depth=1))
    assert answer and not answer.hold_used


def test_settings_validate():
    try:
        find_best_v6(np.zeros((20,10),bool), make_active(),
                settings=SearchSettingsV6(beam_width=0))
    except ValueError:
        pass
    else:
        raise AssertionError('should reject invalid beam')


def test_root_diversity_keeps_alternative():
    b = np.zeros((20,10),bool)
    active = make_active('T')
    moves, _ = v5._current_locks(b,active,3000)
    nodes = [v5._consider(m,None,False,None,0,0) for m in moves]
    roots = _prune(nodes, SearchSettingsV6(beam_width=15,root_diversity=12))
    assert len(roots) == 15
    assert len(set(map(_root_id,roots))) >= 12
