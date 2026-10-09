"""NEXT-first piece tracker for trAssisstant (default colored TETR.IO previews).

The NEXT queue contains future pieces, never the active piece. After observing
stable [A,B,C,...] -> [B,C,D,...], A has just entered play. An optional HOLD
capture handles swaps that do not consume a queued piece. No keyboard hooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from tetris_core import ActivePiece, ROTATIONS, hue_to_piece, normalize, same_color_components

PIECES = frozenset("IJLOSTZ")
UNOBSERVED = object()


def _piece_votes(bgr_or_bgra: np.ndarray, *, min_pixels: int = 12,
                 saturation_min: int = 95, value_min: int = 90) -> Optional[str]:
    """Majority color vote over a preview slot; None means unreadable/empty.

    Vote on bright saturated pixels, not cell centers: NEXT previews are much
    smaller than board cells and their geometry differs from the playfield.
    """
    if bgr_or_bgra.size == 0:
        return None
    if bgr_or_bgra.ndim != 3 or bgr_or_bgra.shape[2] not in (3, 4):
        raise ValueError("Expected BGR/BGRA preview image")
    bgr = cv2.cvtColor(bgr_or_bgra, cv2.COLOR_BGRA2BGR) if bgr_or_bgra.shape[2] == 4 else bgr_or_bgra
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= saturation_min) & (hsv[:, :, 2] >= value_min)
    hues = hsv[:, :, 0][mask]
    if len(hues) < min_pixels:
        return None
    votes = {}
    for h in hues:
        name = hue_to_piece(float(h))
        if name in PIECES:
            votes[name] = votes.get(name, 0) + 1
    if not votes:
        return None
    best, count = max(votes.items(), key=lambda item: item[1])
    # A dominant color is more reliable than a single noisy pixel.
    if count < min_pixels or count < 0.65 * sum(votes.values()):
        return None
    return best


def read_next_queue(image: np.ndarray, slots: int = 5, *,
                    saturation_min: int = 95, value_min: int = 90,
                    min_pixels: int = 12) -> Optional[tuple[str, ...]]:
    """Read NEXT previews enclosed tightly in a *vertical, equally spaced* ROI.

    A bad slot rejects the entire observation instead of corrupting tracking.
    Adjust ROI to cover the icons, excluding NEXT text and unrelated UI.
    """
    if not 2 <= slots <= 8:
        raise ValueError("Queue slots must be between 2 and 8")
    h, w = image.shape[:2]
    if h < 3 * slots or w < 8:
        return None
    names = []
    for i in range(slots):
        # Remove a small inter-slot margin, avoiding spillover from neighbors.
        y0 = round(h * (i + 0.12) / slots)
        y1 = round(h * (i + 0.88) / slots)
        name = _piece_votes(image[y0:y1], saturation_min=saturation_min,
                            value_min=value_min, min_pixels=min_pixels)
        if name is None:
            return None
        names.append(name)
    return tuple(names)


def read_hold_piece(image: np.ndarray, *, saturation_min: int = 95,
                    value_min: int = 90, min_pixels: int = 12) -> Optional[str]:
    """Read HOLD preview color; None means empty OR undetected.

    For reliable HOLD-swap recognition, use the default skin and a tight ROI.
    """
    return _piece_votes(image, saturation_min=saturation_min,
                         value_min=value_min, min_pixels=min_pixels)


def find_active_matching(labels: np.ndarray, expected: str) -> Optional[ActivePiece]:
    """Determine *pose* from board geometry, but *identity* from NEXT history.

    Ignores the detected component's hue identity. For instance, a purple T
    accidentally classified as red is still a T if its geometry matches.
    Never invents a pose when the cells are occluded or touch the stack.
    """
    if expected not in ROTATIONS:
        return None
    candidates = []
    for _color, cells in same_color_components(labels):
        if len(cells) != 4:
            continue
        shape = normalize(cells)
        try:
            rotation = ROTATIONS[expected].index(shape)
        except ValueError:
            continue
        x = min(x for x, _ in cells)
        y = min(y for _, y in cells)
        candidates.append(ActivePiece(expected, rotation, x, y, tuple(sorted(cells))))
    if not candidates:
        return None
    candidates.sort(key=lambda a: (a.y, a.x))
    if len(candidates) > 1 and candidates[0].y == candidates[1].y:
        return None
    return candidates[0]


@dataclass
class QueueTracker:
    """Debounces NEXT/HOLD observations and advances on confirmed queue shifts.

    Start intentionally unsynchronized: initial NEXT does NOT identify CURRENT.
    One confirmed queue shift is needed. Never advance simply because the
    active piece disappears for a frame (e.g. line-clear animation).
    """
    stable_frames: int = 3
    current: Optional[str] = None
    hold: Optional[str] = None
    queue: Optional[tuple[str, ...]] = None
    status: str = "Waiting for NEXT queue calibration"

    def __post_init__(self):
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self._candidate = None
        self._candidate_count = 0
        self._hold_observed = False

    def observe(self, queue: Optional[tuple[str, ...]], hold=UNOBSERVED):
        """One camera-frame observation; queue must be entirely readable."""
        if queue is None or len(queue) < 2 or any(p not in PIECES for p in queue):
            self.status = "NEXT preview not readable — check queue ROI"
            return None
        if hold is not UNOBSERVED and hold not in PIECES and hold is not None:
            self.status = "Invalid HOLD preview"
            return self.current
        observation = (tuple(queue), hold)
        if observation == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = observation
            self._candidate_count = 1
        if self._candidate_count < self.stable_frames:
            if self.queue is not None and (observation[0] != self.queue or
                    (hold is not UNOBSERVED and self._hold_observed and hold != self.hold)):
                self.status = "Confirming queue/HOLD change"
                return None  # No stale advice while the HUD is transitioning.
            return self.current

        new_queue = observation[0]
        if self.queue is None:
            self.queue = new_queue
            if hold is not UNOBSERVED:
                self.hold = hold
                self._hold_observed = True
            self.status = "Synchronized NEXT; place one piece to identify CURRENT"
            return self.current

        # If queue updated, recognize the piece popped from its head. For fast
        # play we can recognize 2+ skipped shifts if enough overlap remains.
        if new_queue != self.queue:
            shifts = [k for k in range(1, len(self.queue))
                      if self.queue[k:] == new_queue[:len(self.queue) - k]]
            if shifts:
                delta = min(shifts)  # largest unambiguous overlap
                self.current = self.queue[delta - 1]
                self.status = f"Queue advanced {delta}: CURRENT={self.current}"
            else:
                self.current = None
                self.status = "Queue changed unexpectedly; waiting to resynchronize"
            self.queue = new_queue
        elif (hold is not UNOBSERVED and self._hold_observed
              and hold != self.hold):
            if self.hold is not None and hold is None:
                # A colored HOLD may turn grey while disabled; that cannot be
                # interpreted safely as a swap. Suppress advice until resync.
                self.current = None
                self.status = "HOLD became unreadable; CURRENT needs resync"
            elif self.hold is not None:
                self.current = self.hold
                self.status = f"HOLD swap: CURRENT={self.current}"
            else:
                # First-time HOLD consumes the NEXT preview. Wait for its shift.
                self.current = None
                self.status = "Empty HOLD used; waiting for NEXT to advance"

        if hold is not UNOBSERVED:
            self.hold = hold
            self._hold_observed = True
        return self.current
