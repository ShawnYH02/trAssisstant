"""Calibrate NEXT / HOLD preview rectangles without modifying board calibration."""
from __future__ import annotations
import ctypes
import json
from pathlib import Path

import cv2
import numpy as np
from mss import MSS


def select_rect(image, title, *, left, top, scale):
    print(title)
    x, y, w, h = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w < 8 or h < 8:
        return None
    return {"left": round(left + x / scale), "top": round(top + y / scale),
            "width": round(w / scale), "height": round(h / scale)}


def main():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass
    config_path = Path(__file__).resolve().parent / 'config.json'
    if not config_path.exists():
        raise SystemExit("config.json is missing. Run the original calibrate.py first.")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    print("TETR.IO: display NEXT queue with the default colored skin.")
    print("First select ONLY the preview icons, including all visible slots; exclude text.")
    with MSS() as capture:
        virtual = capture.monitors[0]
        bgra = np.asarray(capture.grab(virtual))
    screen = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    h, w = screen.shape[:2]
    scale = min(1.0, 1450 / w, 900 / h)
    display = cv2.resize(screen, None, fx=scale, fy=scale) if scale < 1 else screen
    nxt = select_rect(display, 'Select NEXT previews, press ENTER',
                      left=virtual['left'], top=virtual['top'], scale=scale)
    if nxt is None:
        raise SystemExit("NEXT ROI not selected. No config changes.")
    if nxt['height'] < 1.5 * nxt['width']:
        raise SystemExit(
            "NEXT selection is too wide. Select only the narrow vertical column "
            "of colored preview pieces, then rerun calibration. No config changes.")
    cfg['next_queue_roi'] = nxt
    slots = input("How many NEXT preview pieces fit in selection? [5]: ").strip()
    try:
        cfg['next_queue_slots'] = int(slots or 5)
    except ValueError:
        raise SystemExit("Slot count must be a number from 2 to 8. No config changes.")
    if not 2 <= cfg['next_queue_slots'] <= 8:
        raise SystemExit("Slot count must be 2–8. No config changes.")
    print("Optional: select the piece image in HOLD. ESC or CANCEL skips.")
    hold = select_rect(display, 'Select HOLD preview, or ESC to skip',
                       left=virtual['left'], top=virtual['top'], scale=scale)
    if hold is not None and hold['height'] > 1.5 * hold['width']:
        raise SystemExit(
            "HOLD selection is too tall. Select only the single HOLD piece icon, "
            "or cancel to skip HOLD. No config changes.")
    if hold is not None:
        cfg['hold_piece_roi'] = hold
    else:
        cfg.pop('hold_piece_roi', None)
    cfg.setdefault('queue_stable_frames', 3)
    config_path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    print('Saved NEXT ROI' + (' and HOLD ROI' if hold else '') + ' to config.json.')


if __name__ == '__main__':
    main()
