import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import solver_v7 as v7
import solver_v5 as v5


def fixture_roots():
    board=np.zeros((20,10),dtype=bool)
    active=v5._make_spawn('T')
    assert active is not None
    roots, states=v7._legal_roots(board,active,('I','S','Z','O'),None,True,v7.SearchSettingsV7())
    return board,active,roots,states


def test_roots_current_and_hold_are_reachable():
    board,_,roots,states=fixture_roots()
    assert states>0
    assert any(not r[1] for r in roots)
    assert any(r[1] for r in roots)
    for move,use_hold,held,index in roots:
        assert len(move.cells)==4
        assert v5.lock(v5.to_rows(board),move.cells) is not None
        if use_hold:
            assert move.name=='I'
            assert held=='T'
            assert index==1


def test_wire_format_root_ids_and_rows():
    _,_,roots,_=fixture_roots()
    wire=v7.make_request(roots,['I','S','Z'],v7.SearchSettingsV7())
    parts=wire.split()
    assert parts[:9]==['V7','ISZ','5','36','250','0','-1','1',str(len(roots))]
    assert (len(parts)-9)==6*len(roots)
    assert all(len(parts[10+6*i].split(','))==20 for i in range(len(roots)))


def test_fake_native_reply_selects_exact_root(monkeypatch,tmp_path):
    board,active,roots,_=fixture_roots()
    intended=len(roots)-1
    fake=tmp_path/'trassist-v7.exe';fake.write_bytes(b'fake')
    monkeypatch.setattr(v7,'native_path',lambda settings:fake)
    def fake_run(argv,**kw):
        assert argv==[str(fake)]
        assert kw['input'].startswith('V7 ')
        return SimpleNamespace(stdout=f'OK {intended} 121.0 5 900 57\n',stderr='')
    monkeypatch.setattr(v7.subprocess,'run',fake_run)
    best=v7.find_best_v7(board,active,next_queue=['I','S','Z'],can_hold=True)
    assert best.name==roots[intended][0].name
    assert best.cells==roots[intended][0].cells
    assert best.actions[0]=='HOLD'
    assert best.depth_used==5
    assert best.nodes_expanded==900


def test_missing_executable_is_clear_error(monkeypatch,tmp_path):
    board,active,_,_=fixture_roots()
    monkeypatch.setattr(v7,'native_path',lambda settings:tmp_path/'no-engine.exe')
    with pytest.raises(FileNotFoundError,match='cargo build'):
        v7.find_best_v7(board,active,[])
