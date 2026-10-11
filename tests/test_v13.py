from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
V12=ROOT if (ROOT/'solver_cc2.py').is_file() else Path(__file__).resolve().parents[2]/'cc2_adapter_work'
sys.path.insert(0,str(V12))
sys.path.insert(0,str(ROOT))

import solver_cc2 as base
from solver_cc2_fast import (Observed,Selected, possible_advance, applied_lock,
                             FastTBPProcess,find_best_cc2_fast)
from perf_metrics import UILatencyTracker


def obs(board=None, cur='O', queue=('I','T','S'),hold=None):
    return Observed(np.zeros((20,10),bool) if board is None else board,cur,queue,hold)


def selection(cells=((0,18),(1,18),(0,19),(1,19)), name='O', used=False):
    return Selected({'location':{'type':name,'orientation':'north','x':0,'y':0},'spin':'none'},cells,used)


def test_lock_simulation_and_clear():
    b=np.zeros((20,10),bool)
    assert applied_lock(b,((0,18),(1,18),(0,19),(1,19))).sum()==4
    assert applied_lock(b,((0,18),(0,18),(0,19),(1,19))) is None
    assert applied_lock(b,((-1,0),(1,18),(0,19),(1,19))) is None
    b[19,2:]=True
    after=applied_lock(b,((0,18),(1,18),(0,19),(1,19)))
    assert after is not None and after.sum()==2
    assert all(after[19,:2])


def test_confirmed_normal_lock_and_new_piece():
    old=obs()
    new_board=applied_lock(old.board,selection().cells)
    new=obs(new_board,'I',('T','S','Z'))
    assert possible_advance(old,selection(),new)==('Z',)
    bad=obs(new_board,'T',('S','Z','J'))
    assert possible_advance(old,selection(),bad) is None
    bad_queue=obs(new_board,'I',('T','J','Z'))
    assert possible_advance(old,selection(),bad_queue) is None
    external=new_board.copy();external[10,5]=True
    assert possible_advance(old,selection(),obs(external,'I',('T','S','Z'))) is None


def test_confirmed_hold_lock_with_occupied_slot():
    old=obs(cur='L',queue=('I','T','S'),hold='O')
    select=selection(used=True)
    b=applied_lock(old.board,select.cells)
    assert possible_advance(old,select,obs(b,'I',('T','S','Z'),hold='L'))==('Z',)
    assert possible_advance(old,select,obs(b,'I',('T','S','Z'),hold='O')) is None


def test_confirmed_empty_hold_consumes_two_pieces():
    old=obs(cur='J',queue=('O','I','T','S'),hold=None)
    select=selection(used=True)
    b=applied_lock(old.board,select.cells)
    assert possible_advance(old,select,obs(b,'I',('T','S','Z','L'),hold='J')) == ('Z','L')
    assert possible_advance(old,select,obs(b,'O',('I','T','S','Z'),hold='J')) is None


def test_protocol_tree_reuse_and_reset(tmp_path):
    log=tmp_path/'protocol.jsonl'
    script=ROOT/'tests'/'fake_cc2_fast.py'
    client=FastTBPProcess(Path(sys.executable),command=[sys.executable,str(script),str(log)])
    try:
        old=obs()
        first=client.query(old.board,old.current,old.queue,old.hold,budget_ms=300)
        assert first['moves']; assert client.starts==1
        pid=client.proc.pid
        client.remember(first['moves'][0],selection().cells,False)
        next_board=applied_lock(old.board,selection().cells)
        second=client.query(next_board,'I',('T','S','Z'),None,budget_ms=300)
        assert second['moves'];assert client.mode=='advance'
        assert client.advances==1 and client.starts==1 and client.proc.pid==pid
        # Same state: no reset and no repeat play.
        client.query(next_board,'I',('T','S','Z'),None,budget_ms=300)
        assert client.mode=='refresh';assert client.refreshes==1
        corrupted=next_board.copy();corrupted[7,5]=True
        client.query(corrupted,'I',('T','S','Z'),None,budget_ms=300)
        assert client.mode=='start';assert client.starts==2
    finally:
        client.close()
    messages=[json.loads(x) for x in log.read_text().splitlines()]
    types=[x['type'] for x in messages]
    assert types.count('start')==2 and types.count('play')==1
    assert types.count('new_piece')==1
    assert next(x['piece'] for x in messages if x['type']=='new_piece')=='Z'


def test_real_v12_solver_bridge_with_fake(monkeypatch,tmp_path):
    import solver_cc2_fast as fast
    import solver_cc2_reliable as reliable
    import solver_cc2_prefetch as prefetch
    import solver_v5 as v5
    log=tmp_path/'audit.jsonl'
    client=prefetch.PrefetchTBPProcess(
        Path(sys.executable),
        command=[sys.executable, str(ROOT/'tests'/'fake_cc2_fast.py'), str(log)])
    monkeypatch.setattr(reliable,'get_reliable_client',lambda _:client)
    monkeypatch.setenv('TRASSIST_CC2_METRICS','off')
    try:
        board=np.zeros((20,10),bool)
        active=v5._make_spawn('O')
        result=find_best_cc2_fast(board,active,('I','T','S'),None,False,
                                   base.SearchSettingsCC2(time_budget_ms=350))
        assert result is not None and result.name=='O'
        assert result.cells==((0,18),(0,19),(1,18),(1,19)) or set(result.cells)=={(0,18),(0,19),(1,18),(1,19)}
        assert client.selected is not None
    finally:client.close()


def test_ui_latency_one_record_per_key(tmp_path,monkeypatch):
    path=tmp_path/'perf.jsonl'
    monkeypatch.setenv('TRASSIST_CC2_METRICS',str(path))
    t=UILatencyTracker()
    t.observe('a'); t.visible('a');t.visible('a')
    t.observe('b'); t.visible('wrong');t.visible('b',predicted=True)
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows)==2 and rows[1]['predicted']


def test_hold_cooldown_change_does_not_reuse_stale_tree(tmp_path):
    log=tmp_path/'protocol.jsonl'
    client=FastTBPProcess(Path(sys.executable),command=[sys.executable,str(ROOT/'tests'/'fake_cc2_fast.py'),str(log)])
    try:
        b=np.zeros((20,10),bool)
        client.query(b,'T',('O','L'), 'J', can_hold=True,budget_ms=300)
        assert client.starts==1
        client.query(b,'T',('O','L'), 'J', can_hold=False,budget_ms=300)
        assert client.mode=='start' and client.starts==2
    finally:client.close()

def test_targeted_bridge_does_not_enumerate_all_roots(monkeypatch):
    import solver_cc2_fast as fast
    import solver_cc2_reliable as reliable
    import solver_v5 as v5
    import time
    class Client:
        mode='refresh'
        last_restart_reason=None
        prefetch_hits=0
        prefetch_misses=0
        def preflight(self, *args):return False
        def query(self, *a, **k):
            time.sleep(.085)
            return {'moves':[{'location':{'type':'O','orientation':'north','x':0,'y':0},'spin':'none'}], 'move_info':{'nodes':1}}
        def remember(self,*args):pass
        def forget(self):pass
    fake=Client()
    monkeypatch.setattr(reliable,'get_reliable_client',lambda path:fake)
    def reject_whole_board_search(*args):
        raise AssertionError('whole-board search must not run')
    monkeypatch.setattr(v5, '_current_locks', reject_whole_board_search)
    monkeypatch.setenv('TRASSIST_CC2_METRICS','off')
    t0=time.perf_counter()
    result=fast.find_best_cc2_fast(np.zeros((20,10),bool),v5._make_spawn('O'),('I','T'),None,False,
                                   base.SearchSettingsCC2(time_budget_ms=200))
    duration=time.perf_counter()-t0
    assert result is not None
    # Engine readiness is still awaited; only whole-board Python search is gone.
    assert duration >= .08
