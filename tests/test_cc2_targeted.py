from __future__ import annotations
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
# Test file is intended to live in a working trAssisstant directory.
sys.path.insert(0,str(ROOT))
from solver_cc2_targeted import (find_best_cc2_targeted, match_suggestions,
                                 targeted_route)
import solver_cc2 as cc2
import solver_v5 as v5


def test_empty_board_i_instant_route():
    board=np.zeros((20,10),dtype=bool)
    piece=v5._make_spawn('I')
    move,count=targeted_route(board,piece,[(3,19),(4,19),(5,19),(6,19)],budget_ms=200)
    assert move is not None and move.actions==('Hard drop',)
    assert count < 10
    assert move.lines == 0


def test_lateral_route_and_invalid_lock():
    board=np.zeros((20,10),dtype=bool)
    piece=v5._make_spawn('I')
    move,count=targeted_route(board,piece,[(0,19),(1,19),(2,19),(3,19)],budget_ms=200)
    assert move is not None and move.actions[-1]=='Hard drop'
    assert len(move.actions) > 1
    board[19,0] = True
    wrong,count=targeted_route(board,piece,[(0,19),(1,19),(2,19),(3,19)],budget_ms=200)
    assert wrong is None


def test_reject_non_lock_and_bad_geometry():
    board=np.zeros((20,10),dtype=bool)
    piece=v5._make_spawn('I')
    invalid=[(3,17),(4,17),(5,17),(6,17)]
    move,n=targeted_route(board,piece,invalid,budget_ms=100)
    assert move is None
    invalid=[(3,19),(4,19),(4,19),(6,19)]
    move,n=targeted_route(board,piece,invalid,budget_ms=100)
    assert move is None


def test_CC2_choice_matching_no_hold():
    board=np.zeros((20,10),dtype=bool)
    piece=v5._make_spawn('I')
    setting=cc2.SearchSettingsCC2()
    cc2_move={'location':{'type':'I','orientation':'north','x':4,'y':0},'spin':'none'}
    result,n=match_suggestions(board,piece,{'moves':[cc2_move]}, ('T','J'),None,True,setting)
    assert result is not None
    choice, move, hold_used=result
    assert not hold_used and move.cells == ((3,19),(4,19),(5,19),(6,19))


def test_invalid_piece_not_rewritten_as_a_different_move():
    board=np.zeros((20,10),dtype=bool)
    piece=v5._make_spawn('I')
    setting=cc2.SearchSettingsCC2()
    cc2_move={'location':{'type':'T','orientation':'north','x':4,'y':0},'spin':'none'}
    result,n=match_suggestions(board,piece,{'moves':[cc2_move]}, ('S','Z'),None,False,setting)
    assert result is None


def test_selected_recommendation_committed_once(monkeypatch):
    board=np.zeros((20,10),dtype=bool)
    piece=v5._make_spawn('I')
    loc={'location':{'type':'I','orientation':'north','x':4,'y':0},'spin':'none'}
    class FakeClient:
        mode='start';last_restart_reason=None
        def __init__(self): self.calls=0;self.remembers=[]
        def preflight(self,*args): return False
        def query(self,*args,**kwargs):
            self.calls += 1
            return {'moves':[loc],'move_info':{'nodes':7}}
        def remember(self,*args): self.remembers.append(args)
        def forget(self): raise AssertionError('should not forget valid root')
    client=FakeClient()
    mod=importlib.import_module('solver_cc2_targeted')
    monkeypatch.setattr(mod.reliable, 'get_reliable_client', lambda _: client)
    monkeypatch.setattr(mod.cc2,'cc2_path', lambda _: Path('fake-engine'))
    monkeypatch.setenv('TRASSIST_CC2_METRICS','off')
    best=mod.find_best_cc2_targeted(board,piece,next_queue=('T','J'),settings=cc2.SearchSettingsCC2())
    assert best is not None and best.name=='I' and best.actions==('Hard drop',)
    assert client.calls==1 and len(client.remembers)==1


def test_reaches_sampled_legacy_routes():
    """Regression: the early-exit matcher must not invent better movement rules."""
    import random
    rng = random.Random(33)
    board = np.zeros((20,10), dtype=bool)
    for name in 'IOTJLSZ':
        active=v5._make_spawn(name)
        moves, states=v5._current_locks(board,active,5000)
        for original in rng.sample(moves, min(6, len(moves))):
            route, searched=targeted_route(board,active,original.cells,original.spin,
                                          budget_ms=200,max_states=5000)
            assert route is not None, (name, original.cells, original.spin)
            assert route.cells == original.cells
            assert route.actions[-1] == 'Hard drop'


def test_transition_pending_is_propagated(monkeypatch):
    from solver_cc2_resync import TransitionPending

    class PendingClient:
        def preflight(self, *args):
            return True

    module = importlib.import_module('solver_cc2_targeted')
    monkeypatch.setattr(module.reliable, 'get_reliable_client',
                        lambda _: PendingClient())
    monkeypatch.setattr(module.cc2, 'cc2_path', lambda _: Path('fake-engine'))
    with pytest.raises(TransitionPending):
        module.find_best_cc2_targeted(
            np.zeros((20, 10), dtype=bool), v5._make_spawn('I'),
            next_queue=('T', 'J'), settings=cc2.SearchSettingsCC2())
