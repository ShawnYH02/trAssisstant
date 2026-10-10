from dataclasses import dataclass
import numpy as np
from smooth_plan import PlanMemory, after_lock, state_key

@dataclass(frozen=True)
class Result:
    name: str = 'T'
    cells: tuple = ((4,18), (3,19), (4,19), (5,19))
    next_name: str | None = 'I'
    next_cells: tuple | None = ((0,16),(0,17),(0,18),(0,19))
    third_name: str | None = 'O'
    third_cells: tuple | None = ((8,18),(9,18),(8,19),(9,19))
    hold_used: bool = False
    actions: tuple = ('CCW','Hard drop')
    cleared: int = 0

SETTINGS = 'fast'
B = np.zeros((20,10), dtype=bool)

def key(b,name,queue=('I','O','L','S','Z'),hold=None,can_hold=True):
    return state_key(b,name,queue,hold,can_hold,SETTINGS)

def test_key_ignores_pose_and_board_copy():
    assert key(B,'T') == key(B.copy(),'T')

def test_keep_same_move_through_pose_changes():
    p=PlanMemory()
    p.store(key(B,'T'),B,Result())
    x=p.show(key(B,'T'),B)
    assert x is not None and not x.predicted
    assert x.best.actions == ('CCW','Hard drop')

def test_successor_reuses_next_and_shifts_preview():
    p=PlanMemory(); original=Result()
    p.store(key(B,'T'),B,original)
    follow=after_lock(B,original.cells)
    assert follow is not None
    state=key(follow,'I',('O','L','S','Z','J'),None,True)
    preview=p.show(state,follow)
    assert preview and preview.predicted
    assert preview.best.cells == original.next_cells
    assert preview.best.next_cells == original.third_cells
    assert preview.best.actions == ()
    assert p.show(state,follow).predicted

def test_wrong_move_forces_replan():
    p=PlanMemory(); p.store(key(B,'T'),B,Result())
    altered=B.copy();altered[19,2]=True
    assert p.show(key(altered,'I',('O','L','S','Z','J')),altered) is None

def test_misread_queue_does_not_promote():
    p=PlanMemory(); best=Result(); p.store(key(B,'T'),B,best)
    after=after_lock(B,best.cells)
    assert p.show(key(after,'I',('Z','L','S','O','J')),after) is None

def test_occupied_hold_preserved_on_swap():
    p=PlanMemory()
    result=Result(name='T', next_name='I', hold_used=True)
    p.store(key(B,'T',hold='L'),B,result)
    after=after_lock(B,result.cells)
    assert p.show(key(after,'I',('O','L','S','Z','J'),'T',True),after).predicted

def test_empty_hold_consumes_two_queue_entries():
    p=PlanMemory()
    result=Result(name='T', next_name='O', hold_used=True)
    p.store(key(B,'T'),B,result)
    after=after_lock(B,result.cells)
    assert p.show(key(after,'O',('L','S','Z','J','T'),'T',True),after).predicted

def test_cooldown_or_no_hold_change_rejected():
    p=PlanMemory(); result=Result(hold_used=True)
    p.store(key(B,'T',hold='L'),B,result)
    after=after_lock(B,result.cells)
    assert p.show(key(after,'I',('O','L','S','Z','J'),'L',True),after) is None

def test_line_clear_matches_expected():
    b=np.zeros((20,10),dtype=bool)
    b[19,:]=True
    b[19,3:7]=False
    result=after_lock(b,((3,19),(4,19),(5,19),(6,19)))
    assert result is not None
    assert not result.any()

def test_collision_rejected():
    b=B.copy();b[19,3]=True
    assert after_lock(b,Result().cells) is None

def test_prediction_only_first_proper_transition():
    p=PlanMemory();best=Result()
    p.store(key(B,'T'),B,best)
    after=after_lock(B,best.cells)
    wrong=key(after,'O',('L','S','Z','J','I'),None,True)
    assert p.show(wrong,after) is None


def test_planned_hold_swap_has_no_blank_period():
    p=PlanMemory()
    best=Result(name='L', hold_used=True, actions=('HOLD','Right','Hard drop'))
    p.store(key(B,'T',hold='L'),B,best)
    view=p.show(key(B,'L',('I','O','L','S','Z'),'T',False),B)
    assert view and view.predicted and view.best.cells==best.cells
    assert view.best.actions==('Right','Hard drop')
    assert not view.best.hold_used


def test_planned_empty_hold_consumes_queue_piece_without_lock():
    p=PlanMemory()
    best=Result(name='I', hold_used=True, actions=('HOLD','Hard drop'))
    p.store(key(B,'T'),B,best)
    view=p.show(key(B,'I',('O','L','S','Z','J'),'T',False),B)
    assert view and view.predicted


def test_unplanned_hold_disallows_reuse():
    p=PlanMemory();p.store(key(B,'T',hold='L'),B,Result())
    assert p.show(key(B,'L',('I','O','L','S','Z'),'T',False),B) is None
