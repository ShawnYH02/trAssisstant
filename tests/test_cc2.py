import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solver_cc2 import (TBPProcess, legal_root_for_suggestion, tbp_board,
                        tbp_cells)


def test_board_orientation():
    b=np.zeros((20,10),dtype=bool); b[19,0]=True; b[0,9]=True
    out=tbp_board(b)
    assert len(out)==40 and all(len(r)==10 for r in out)
    assert out[0][0]=='G' and out[19][9]=='G'
    assert out[20]==[None]*10


def test_o_piece_center():
    assert tbp_cells({'type':'O','orientation':'north','x':0,'y':0})==((0,18),(0,19),(1,18),(1,19))


def test_t_spin_rotation_coords():
    # CC2 north T has minos left, center, right, top, relative to pivot.
    assert tbp_cells({'type':'T','orientation':'north','x':4,'y':1}) == ((3,18),(4,17),(4,18),(5,18))
    assert tbp_cells({'type':'T','orientation':'east','x':4,'y':1}) == ((4,17),(4,18),(4,19),(5,18))


def test_legal_root_matching():
    from types import SimpleNamespace
    cells=tbp_cells({'type':'O','orientation':'north','x':0,'y':0})
    root=SimpleNamespace(name='O', cells=cells, spin='none', actions=('Hard drop',))
    move={'location':{'type':'O','orientation':'north','x':0,'y':0},'spin':'none'}
    assert legal_root_for_suggestion(move,[(root,False)],'O',None,(),True)==(root,False)
    assert legal_root_for_suggestion(move,[(root,False)],'T','O',(),False) is None
    assert legal_root_for_suggestion(move,[(root,True)],'T','O',(),True)==(root,True)
    assert legal_root_for_suggestion({'location':move['location'],'spin':'full'},[(root,False)],'O',None,(),True) is None


def test_process_persistent_and_start_again():
    script = Path(__file__).resolve().parent / 'fake_cc2.py'
    p=TBPProcess(Path(sys.executable), command=[sys.executable, str(script)])
    try:
        b=np.zeros((20,10),dtype=bool)
        started=time.perf_counter()
        first=p.query(b,'O',('T','I'),None,budget_ms=140)
        assert time.perf_counter()-started >= .10
        assert first['move_info']['nodes'] >= 222
        old_pid=p.proc.pid
        second=p.query(b,'O',('I','S'),None,budget_ms=140)
        assert p.proc.pid==old_pid
        assert first['moves'][0]['location']['type']=='O'
        assert second['moves'][0]['location']['type']=='O'
    finally:p.close()


def test_full_solver_bridge_with_mock(monkeypatch):
    import solver_cc2
    import solver_v5 as v5
    script = Path(__file__).resolve().parent / 'fake_cc2.py'
    client = TBPProcess(Path(sys.executable), command=[sys.executable, str(script)])
    monkeypatch.setattr(solver_cc2, 'get_client', lambda _: client)
    try:
        b = np.zeros((20, 10), dtype=bool)
        active = v5._make_spawn('O')
        answer = solver_cc2.find_best_cc2(b, active, ('T', 'I'), None, True,
                                          solver_cc2.SearchSettingsCC2(time_budget_ms=450))
        assert answer is not None
        assert answer.name == 'O'
        assert set(answer.cells) == {(0, 18), (1, 18), (0, 19), (1, 19)}
        assert answer.actions[-1] == 'Hard drop'
        assert answer.next_cells is None
    finally:
        client.close()
