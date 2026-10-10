from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import solver_cc2 as base
import solver_cc2_fast as fast
import solver_cc2_reliable as reliable
import solver_v5 as v5
from solver_cc2_reliable import ReliableTBPProcess


def make_client(scenario='slow'):
    return ReliableTBPProcess(
        Path(sys.executable),
        command=[sys.executable,
                 str(ROOT / 'tests' / 'fake_cc2_recovery.py'), scenario],
        settle_ms=40, minimum_budget_ms=650)


def test_recovery_from_late_engine_response(monkeypatch, tmp_path):
    log = tmp_path / 'recovery.jsonl'
    monkeypatch.setenv('TRASSIST_CC2_RECOVERY_LOG', str(log))
    client = make_client()
    try:
        client.open()
        started = time.perf_counter()
        message = client.query(np.zeros((20, 10), bool), 'O',
                               ('I', 'T', 'S'), None,
                               can_hold=True, budget_ms=120)
        elapsed = (time.perf_counter() - started) * 1000
        assert message['moves'][0]['location']['type'] == 'O'
        assert 300 < elapsed < 650
        assert 'engine_reply' in log.read_text()
    finally:
        client.close()


def test_retry_rejects_invalid_until_valid(monkeypatch, tmp_path):
    monkeypatch.setenv('TRASSIST_CC2_RECOVERY_LOG',
                       str(tmp_path / 'recovery.jsonl'))
    client = make_client('late_legal')
    try:
        first = client.query(np.zeros((20, 10), bool), 'O',
                             ('I', 'T', 'S'), None,
                             can_hold=True, budget_ms=80)
        assert first['moves'][0]['location']['type'] == 'I'
        later = client.retry_for_valid(
            lambda move: move['location']['type'] == 'O', budget_ms=350)
        assert later['moves'][0]['location']['type'] == 'O'
        assert client.last_recovery_reason == 'recovered_valid_route'
    finally:
        client.close()


def test_no_unsafe_candidate_in_retry():
    client = make_client('late_legal')
    try:
        client.query(np.zeros((20, 10), bool), 'O', ('I', 'T', 'S'),
                     None, can_hold=True, budget_ms=80)
        assert client.retry_for_valid(lambda move: False,
                                      budget_ms=75) is None
        assert client.last_recovery_reason == 'no_legal_route'
    finally:
        client.close()


def test_integrated_bridge_recovers_legal_route(monkeypatch):
    client = make_client('late_legal')
    monkeypatch.setattr(reliable, 'get_reliable_client', lambda _: client)
    try:
        result = fast.find_best_cc2_fast(
            np.zeros((20, 10), bool), v5._make_spawn('O'),
            ('I', 'T', 'S'), None, False,
            base.SearchSettingsCC2(time_budget_ms=80))
        assert result is not None
        assert result.name == 'O'
        assert client.last_recovery_reason == 'recovered_valid_route'
    finally:
        client.close()
