from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solver_cc2_fast import applied_lock
from solver_cc2_first import FirstTBPProcess


def fake(tmp_path, delay=0.0):
    return FirstTBPProcess(
        Path(sys.executable),
        [sys.executable, str(ROOT / 'tests' / 'fake_cc2_polling.py'),
         str(delay), str(tmp_path / 'tbp.jsonl')],
        poll_ms=7, min_send_interval_ms=4, settle_ms=30)


def test_immediate_nonempty_returns_without_filling_budget(tmp_path,
                                                            monkeypatch):
    metrics = tmp_path / 'metrics.jsonl'
    monkeypatch.setenv('TRASSIST_CC2_METRICS', str(metrics))
    client = fake(tmp_path)
    try:
        client.open()
        started = time.perf_counter()
        message = client.query(np.zeros((20, 10), bool), 'O',
                               ('I', 'T', 'S'), None,
                               can_hold=True, budget_ms=500)
        elapsed = (time.perf_counter() - started) * 1000
        assert message['moves'][0]['location']['type'] == 'O'
        assert client.last_first_metrics['found'] is True
        assert client.last_first_metrics['first_nonempty_ms'] < 250
        assert elapsed < 250
        assert any(json.loads(line)['event'] == 'cc2_first_suggestion'
                   for line in metrics.read_text().splitlines())
    finally:
        client.close()


def test_delayed_nonempty_is_measured(tmp_path):
    client = fake(tmp_path, 0.075)
    try:
        client.query(np.zeros((20, 10), bool), 'O', ('I', 'T', 'S'),
                     None, can_hold=True, budget_ms=500)
        metrics = client.last_first_metrics
        assert metrics['first_reply_ms'] is not None
        assert 45 <= metrics['first_nonempty_ms'] < 350
        assert metrics['polls'] >= 2
        assert (metrics['empty_replies'] +
                metrics['unanswered_polls']) > 0
    finally:
        client.close()


def test_no_reply_does_not_create_a_move(tmp_path):
    client = fake(tmp_path, 1.0)
    try:
        with pytest.raises(TimeoutError):
            client.query(np.zeros((20, 10), bool), 'O', ('I', 'T', 'S'),
                         None, can_hold=True, budget_ms=55)
        assert client.last_first_metrics['found'] is False
        assert client.last_first_metrics['first_nonempty_ms'] is None
    finally:
        client.close()


def test_cached_reply_uses_zero_polls(tmp_path):
    client = fake(tmp_path)
    try:
        board = np.zeros((20, 10), bool)
        client.query(board, 'O', ('I', 'T', 'S'), None,
                     can_hold=True, budget_ms=150)
        client.query(board, 'O', ('I', 'T', 'S'), None,
                     can_hold=True, budget_ms=150)
        assert client.mode == 'cached_same_piece'
        assert client.last_first_metrics['polls'] == 0
    finally:
        client.close()


def test_wrong_placement_waits_then_resets(tmp_path):
    client = fake(tmp_path)
    try:
        board = np.zeros((20, 10), bool)
        client.query(board, 'O', ('I', 'T', 'S'), None,
                     can_hold=True, budget_ms=150)
        client.remember(
            {'location': {'type': 'O', 'orientation': 'north',
                          'x': 0, 'y': 0}},
            ((0, 18), (1, 18), (0, 19), (1, 19)), False)
        wrong = applied_lock(
            board, ((8, 18), (9, 18), (8, 19), (9, 19)))
        assert client.preflight(wrong, 'I', ('T', 'S', 'Z'), None, True)
        time.sleep(.04)
        assert not client.preflight(wrong, 'I', ('T', 'S', 'Z'), None, True)
        client.query(wrong, 'I', ('T', 'S', 'Z'), None,
                     can_hold=True, budget_ms=150)
        assert client.starts == 2
    finally:
        client.close()
