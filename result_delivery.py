"""Qt-independent helper for low-overhead completion wakeups and UI timing.

Never produces a move, alters a result, or accesses Qt from a worker thread.
"""
from __future__ import annotations

from collections import deque
import json
import os
import threading
import time


class CompletionPulse:
    def __init__(self, *, clock=None, metrics_file=None):
        self._clock = clock or time.perf_counter
        self._file = metrics_file if metrics_file is not None else os.environ.get(
            'TRASSIST_DELIVERY_LOG', 'cc2_delivery.jsonl')
        self._lock = threading.Lock()
        self._current = None
        self._seq = 0

    def watch(self, future, key):
        """Register a new background solve. Safe even if already finished."""
        with self._lock:
            self._seq += 1
            record = {
                'future': future, 'key': key,
                'seq': self._seq, 'submitted': self._clock(),
                'finished': None, 'wake': None,
                'processed': None, 'assigned': None,
                'woken': False,
            }
            self._current = record

        def on_done(done):
            completed_at = self._clock()
            with self._lock:
                if self._current is record and record['finished'] is None:
                    record['finished'] = completed_at

        future.add_done_callback(on_done)

    def ready(self, future):
        """True only once per newly finished future, to request a Qt tick."""
        if future is None or not future.done():
            return False
        with self._lock:
            rec = self._current
            if rec is None or rec['future'] is not future or rec['woken'] or rec['processed'] is not None:
                return False
            rec['woken'] = True
            rec['wake'] = self._clock()
            return True

    def processed(self, future):
        """Called on UI thread when the completed Future is collected."""
        with self._lock:
            rec = self._current
            if rec is None or rec['future'] is not future or rec['processed'] is not None:
                return
            rec['processed'] = self._clock()
            if rec['finished'] is None:  # a callback can run just after done() becomes true
                rec['finished'] = rec['processed']

    def assigned(self, key):
        """Called after all overlay fields for the correct result are assigned."""
        with self._lock:
            rec = self._current
            if (rec is None or rec['key'] != key or
                    rec['processed'] is None or rec['assigned'] is not None):
                return
            rec['assigned'] = self._clock()
            snapshot = dict(rec)
        self._record(snapshot)

    def _record(self, rec):
        if str(self._file).lower() in ('0', 'false', 'off', ''):
            return
        try:
            data = {
                'event': 'result_delivery',
                'sequence': rec['seq'],
                'solve_ms': round(1000 * (rec['finished'] - rec['submitted']), 3),
                'completion_to_processing_ms': round(1000 * (rec['processed'] - rec['finished']), 3),
                'processing_to_ui_assignment_ms': round(1000 * (rec['assigned'] - rec['processed']), 3),
                'submit_to_ui_assignment_ms': round(1000 * (rec['assigned'] - rec['submitted']), 3),
                'wakeup_used': rec['wake'] is not None,
            }
            with open(self._file, 'a', encoding='utf-8') as log:
                log.write(json.dumps(data, separators=(',', ':')) + '\n')
        except OSError:
            pass


class PaintDeduper:
    """Avoid submitting the same overlay pixels to Qt every capture frame."""
    def __init__(self):
        self._last = object()

    def invalidate(self):
        self._last = object()

    def changed(self, overlay):
        keys = ('text', 'subtitle', 'target', 'target_name',
                'next_target', 'next_target_name', 'third_target', 'third_target_name')
        def freeze(value):
            if isinstance(value, (tuple, list)):
                return tuple(freeze(x) for x in value)
            if isinstance(value, dict):
                return tuple(sorted((k, freeze(v)) for k, v in value.items()))
            return value
        state = tuple(freeze(getattr(overlay, key, None)) for key in keys)
        if state == self._last:
            return False
        self._last = state
        return True
