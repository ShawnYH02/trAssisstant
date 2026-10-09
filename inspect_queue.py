"""One-shot diagnosis: print queue reader output and save a cropped image."""
from pathlib import Path
import json

import cv2
import numpy as np
from mss import MSS
from queue_first import read_next_queue, read_hold_piece


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    roi = cfg.get('next_queue_roi')
    if roi is None:
        raise SystemExit('Missing NEXT ROI. Run python calibrate_queue.py first.')
    with MSS() as capture:
        frame = np.asarray(capture.grab(roi))
        held = (np.asarray(capture.grab(cfg['hold_piece_roi']))
                if 'hold_piece_roi' in cfg else None)
    result = read_next_queue(frame, slots=int(cfg.get('next_queue_slots', 5)),
        saturation_min=cfg.get('queue_saturation_min', 95),
        value_min=cfg.get('queue_value_min', 90))
    print('NEXT:', ' '.join(result) if result else 'UNREADABLE')
    name = (read_hold_piece(
        held,
        saturation_min=cfg.get('queue_saturation_min', 95),
        value_min=cfg.get('queue_value_min', 90),
        min_pixels=cfg.get('queue_min_pixels', 12))
        if held is not None else 'not calibrated')
    print('HOLD:', name if name is not None else 'empty/unreadable')
    out = root / 'debug_queue.png'
    cv2.imwrite(str(out), frame)
    print('Saved:', out)


if __name__ == '__main__':
    main()
