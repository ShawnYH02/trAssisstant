"""Capture board, NEXT, and HOLD from one physical MSS frame when practical."""
from __future__ import annotations

import numpy as np


def _rectangle(roi):
    if not roi:
        return None
    return {name: int(roi[name])
            for name in ('left', 'top', 'width', 'height')}


class CoherentFrame:
    def __init__(self, capture, rois, max_pixels=3_000_000):
        self.capture = capture
        rectangles = [_rectangle(roi) for roi in rois if roi]
        self.area = None
        self.image = None
        self.coherent = False
        if not rectangles:
            return
        left = min(rect['left'] for rect in rectangles)
        top = min(rect['top'] for rect in rectangles)
        right = max(rect['left'] + rect['width'] for rect in rectangles)
        bottom = max(rect['top'] + rect['height'] for rect in rectangles)
        area = {'left': left, 'top': top, 'width': right - left,
                'height': bottom - top}
        if area['width'] <= 0 or area['height'] <= 0:
            return
        if area['width'] * area['height'] <= max_pixels:
            self.image = np.asarray(capture.grab(area))
            self.area = area
            self.coherent = True

    def crop(self, roi):
        rectangle = _rectangle(roi)
        if rectangle is None:
            raise ValueError('Cannot crop an empty ROI')
        if self.image is None:
            return np.asarray(self.capture.grab(rectangle))
        x = rectangle['left'] - self.area['left']
        y = rectangle['top'] - self.area['top']
        width, height = rectangle['width'], rectangle['height']
        if (x < 0 or y < 0 or x + width > self.area['width'] or
                y + height > self.area['height']):
            raise ValueError('ROI outside captured union')
        return self.image[y:y + height, x:x + width]
