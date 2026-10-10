"""Record screen-state-to-first-visible-ghost time once per observed state."""
from __future__ import annotations

import json
import os
import time


class UILatencyTracker:
    def __init__(self):
        self.key = None
        self.started = 0.0
        self.logged = False

    def observe(self, state_key):
        if self.key != state_key:
            self.key = state_key
            self.started = time.perf_counter()
            self.logged = False

    def visible(self, state_key, predicted=False):
        if self.logged or self.key != state_key:
            return
        self.logged = True
        path = os.environ.get('TRASSIST_CC2_METRICS', 'cc2_latency.jsonl')
        if path.lower() in ('off', '0', 'false'):
            return
        record = {
            'event': 'visible',
            'latency_ms': round((time.perf_counter() - self.started) * 1000, 3),
            'predicted': bool(predicted),
        }
        try:
            with open(path, 'a', encoding='utf-8') as output:
                output.write(json.dumps(record, separators=(',', ':')) + '\n')
        except OSError:
            pass
