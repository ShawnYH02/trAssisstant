"""Briefly retain a verified ghost across a small visual pose dropout."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class PoseSnapshot:
    current: str
    queue: tuple
    hold: object
    labels: np.ndarray
    when: float
    has_ghost: bool


class PoseGapGuard:
    def __init__(self, grace_ms: int = 90, max_changed_cells: int = 8):
        self.grace_s = max(0.0, min(0.2, grace_ms / 1000.0))
        self.max_changed_cells = max(0, min(8, max_changed_cells))
        self.last: PoseSnapshot | None = None
        self.kept = 0
        self.rejected = 0

    def observe(self, labels, current, queue, hold, *, ghost_verified=False,
                now=None):
        self.last = PoseSnapshot(
            str(current), tuple(queue or ()), hold,
            np.asarray(labels).copy(),
            time.monotonic() if now is None else now,
            bool(ghost_verified))

    def keep(self, labels, current, queue, hold, *, now=None):
        now = time.monotonic() if now is None else now
        previous = self.last
        if (previous is None or not previous.has_ghost or
                now - previous.when > self.grace_s):
            self.rejected += 1
            return False
        if (previous.current != current or
                previous.queue != tuple(queue or ()) or
                previous.hold != hold):
            self.rejected += 1
            return False
        observed = np.asarray(labels)
        if observed.shape != previous.labels.shape:
            self.rejected += 1
            return False
        # One moving tetromino can alter up to eight old/new cell positions.
        # Larger changes indicate a lock, line clear, garbage, or bad scan.
        if int(np.count_nonzero(observed != previous.labels)) > self.max_changed_cells:
            self.rejected += 1
            return False
        self.kept += 1
        return True

    def invalidate(self):
        self.last = None
