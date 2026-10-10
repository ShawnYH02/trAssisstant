from __future__ import annotations
import json
import sys
from pathlib import Path
import time

import numpy as np

DIR=Path(__file__).resolve().parents[1]
V13=DIR if (DIR/'solver_cc2_fast.py').exists() else DIR.parent/'trAssisstant_v13_perf'
V12=DIR if (DIR/'solver_cc2.py').exists() else DIR.parent/'cc2_adapter_work'
sys.path.insert(0,str(V12))
sys.path.insert(0,str(V13))
sys.path.insert(0,str(DIR))
from solver_cc2_prefetch import PrefetchTBPProcess
from solver_cc2_fast import applied_lock

CELLS=((0,18),(1,18),(0,19),(1,19))
MOVE={'location':{'type':'O','orientation':'north','x':0,'y':0},'spin':'none'}


def make_client(tmp_path,slow=False):
    log=tmp_path/'protocol.jsonl'
    script=DIR/'tests'/('fake_slow_cc2.py' if slow else 'fake_cc2_fast.py')
    c=PrefetchTBPProcess(Path(sys.executable),[sys.executable,str(script),str(log)])
    return c,log


def events(log):
    return [json.loads(x) for x in log.read_text(encoding='utf-8').splitlines()]


def test_early_play_single_command_and_confirmed_next(tmp_path):
    c,log=make_client(tmp_path)
    b=np.zeros((20,10),bool)
    nxt=applied_lock(b,CELLS)
    try:
        c.query(b,'O',('I','T','S'),None,budget_ms=200)
        c.remember(MOVE,CELLS,False)
        assert c._prefetched
        # The speculative command is sent BEFORE the player locks anything.
        time.sleep(.025)  # OS pipe reader may not have processed the write yet
        assert [e['type'] for e in events(log)].count('play')==1
        # An identical old-state query must return cached recommendation, not
        # a suggestion for the next piece being searched right now.
        c.query(b,'O',('I','T','S'),None,budget_ms=200)
        assert c.mode == 'cached_same_piece'
        c.query(nxt,'I',('T','S','Z'),None,budget_ms=200)
        assert c.mode == 'prefetch_hit'
        assert c.prefetch_hits==1
        assert c.starts==1
    finally:
        c.close()
    types=[e['type'] for e in events(log)]
    assert types.count('start')==1
    assert types.count('play')==1
    assert types.count('new_piece')==1


def test_wrong_lock_triggers_full_reset(tmp_path):
    c,log=make_client(tmp_path)
    b=np.zeros((20,10),bool)
    wrong=applied_lock(b,((8,18),(9,18),(8,19),(9,19)))
    try:
        c.query(b,'O',('I','T','S'),None,budget_ms=200)
        c.remember(MOVE,CELLS,False)
        c.query(wrong,'I',('T','S','Z'),None,budget_ms=200)
        assert c.mode == 'start'
        assert c.starts == 2
        assert c.prefetch_misses==1
    finally:c.close()
    types=[e['type'] for e in events(log)]
    assert types.count('start')==2 and types.count('play')==1


def test_occupied_hold_confirmation(tmp_path):
    c,log=make_client(tmp_path)
    b=np.zeros((20,10),bool)
    nxt=applied_lock(b,CELLS)
    try:
        c.query(b,'L',('I','T','S'), 'O', can_hold=True,budget_ms=200)
        c.remember(MOVE,CELLS,True)
        c.query(nxt,'I',('T','S','Z'),'L',can_hold=True,budget_ms=200)
        assert c.mode=='prefetch_hit'
    finally:c.close()


def test_first_hold_consumes_two_queue_entries(tmp_path):
    c,log=make_client(tmp_path)
    b=np.zeros((20,10),bool)
    nxt=applied_lock(b,CELLS)
    try:
        c.query(b,'L',('O','I','T','S'),None,can_hold=True,budget_ms=200)
        c.remember(MOVE,CELLS,True)
        c.query(nxt,'I',('T','S','Z','J'),'L',can_hold=True,budget_ms=200)
        assert c.mode=='prefetch_hit'
    finally:c.close()
    add=[e['piece'] for e in events(log) if e['type']=='new_piece']
    assert add==['Z','J']


def test_slow_engine_uses_human_time_as_think_time(tmp_path):
    c,log=make_client(tmp_path,slow=True)
    b=np.zeros((20,10),bool)
    nxt=applied_lock(b,CELLS)
    try:
        c.query(b,'O',('I','T','S'),None,budget_ms=380)
        c.remember(MOVE,CELLS,False)
        time.sleep(.19)  # time human spends moving current piece
        t=time.perf_counter()
        c.query(nxt,'I',('T','S','Z'),None,budget_ms=380)
        millis=(time.perf_counter()-t)*1000
        assert c.mode=='prefetch_hit'
        assert millis < 110, millis
    finally:c.close()


def test_routes_prefetched_only_for_matching_spawn(tmp_path, monkeypatch):
    import solver_v5 as v5
    monkeypatch.setenv('TRASSIST_PREFETCH_ROUTES', '1')
    c,log=make_client(tmp_path)
    b=np.zeros((20,10),bool)
    nxt=applied_lock(b,CELLS)
    try:
        c.query(b,'O',('I','T','S'),None,budget_ms=200)
        c.remember(MOVE,CELLS,False)
        routes_future=c._route_future
        assert routes_future is not None
        routes_future.result(timeout=12)
        active=v5._make_spawn('I')
        assert active is not None
        cached=c.get_predicted_routes(nxt, active, ('T','S','Z'), None, True, 5000)
        assert cached is not None
        paired, states=cached
        assert any(not use_hold for _,use_hold in paired)
        assert any(use_hold and move.name=='T' for move,use_hold in paired)
        assert states>0
        assert c.get_predicted_routes(nxt, active, ('S','T','Z'), None, True, 5000) is None
        wrong=nxt.copy();wrong[10,4]=True
        assert c.get_predicted_routes(wrong, active, ('T','S','Z'), None, True, 5000) is None
        assert c.get_predicted_routes(nxt, active, ('T','S','Z'), None, False, 5000) is not None
        no_hold,_=c.get_predicted_routes(nxt, active, ('T','S','Z'), None, False, 5000)
        assert not any(used for _,used in no_hold)
    finally:c.close()


def test_early_prefetch_works_in_integrated_bridge(tmp_path, monkeypatch):
    import solver_cc2_fast as fast
    import solver_cc2_first as first
    import solver_cc2_prefetch as prefetch
    import solver_v5 as v5
    client,log=make_client(tmp_path)
    monkeypatch.setattr(first,'get_first_client',lambda exe:client)
    monkeypatch.setenv('TRASSIST_CC2_METRICS','off')
    try:
        result=fast.find_best_cc2_fast(np.zeros((20,10),bool),
            v5._make_spawn('O'),('I','T','S'),None,False,
            fast.base.SearchSettingsCC2(time_budget_ms=200))
        assert result is not None
        assert client._prefetched
        assert client.selected is not None
    finally:client.close()
