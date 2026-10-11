from __future__ import annotations
import concurrent.futures
from pathlib import Path
import sys
import types

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from result_delivery import CompletionPulse, PaintDeduper


def test_overlay_has_completion_wakeup_without_faster_capture_loop():
    source = (Path(__file__).resolve().parents[1] /
              'assistant_overlay.py').read_text(encoding='utf-8')
    assert 'completion_pulse.watch(solver_future, pending_key)' in source
    assert 'completion_pulse.processed(completed)' in source
    assert 'completion_pulse.assigned(state_key)' in source
    assert 'completion_interval_ms", 12' in source
    assert source.count('paint_deduper.changed(overlay)') == 1
    assert 'paint_deduper.invalidate()' in source
    assert 'completion_timer.stop()' in source


def test_result_wakeup_and_metrics(tmp_path):
    f=concurrent.futures.Future()
    now=[1.0]
    pulse=CompletionPulse(clock=lambda:now[0], metrics_file=str(tmp_path/'log.jsonl'))
    pulse.watch(f,('state',1))
    assert pulse.ready(f) is False
    now[0]=1.025
    f.set_result('move')
    now[0]=1.032
    assert pulse.ready(f) is True
    assert pulse.ready(f) is False
    now[0]=1.04
    pulse.processed(f)
    now[0]=1.045
    pulse.assigned(('wrong',1))
    assert not (tmp_path/'log.jsonl').exists()
    pulse.assigned(('state',1))
    import json
    record=json.loads((tmp_path/'log.jsonl').read_text().strip())
    assert record['solve_ms']==25
    assert record['completion_to_processing_ms']==15
    assert record['wakeup_used'] is True
    assert record['event'] == 'result_delivery'
    pulse.assigned(('state',1))
    assert len((tmp_path/'log.jsonl').read_text().splitlines())==1


def test_normal_capture_tick_preempts_pulse():
    f=concurrent.futures.Future()
    pulse=CompletionPulse(metrics_file='off')
    pulse.watch(f, 'abc')
    f.set_result(None)
    pulse.processed(f)
    assert not pulse.ready(f)


def test_deduper():
    dummy=types.SimpleNamespace(text='hello',subtitle='',target=[(1,1)],target_name='T',
                next_target=None,third_target=None,next_target_name=None,third_target_name=None)
    deduper=PaintDeduper()
    assert deduper.changed(dummy)
    assert not deduper.changed(dummy)
    dummy.target = [(2,1)]
    assert deduper.changed(dummy)
    dummy.subtitle='next'
    assert deduper.changed(dummy)
    deduper.invalidate()
    assert deduper.changed(dummy)
