"""Read saved V4 preview rectangles once and print recognized NEXT/HOLD."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from mss import MSS
from vision_v4 import read_queue_region,read_queue_rois,read_hold_view


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--delay',type=int,default=3,
                        help='seconds to switch focus back to TETR.IO before capture')
    args=parser.parse_args()
    cfg=json.loads(Path('config.json').read_text(encoding='utf-8'))
    region=cfg.get('next_queue_region')
    rois=cfg.get('next_piece_rois') or []
    if region is None and len(rois)<2:
        raise SystemExit('Missing NEXT region; run python calibrate_all.py')
    if region is not None:
        bounds={key:int(region[key]) for key in ('left','top','width','height')}
    else:
        xs=[r['left'] for r in rois]; ys=[r['top'] for r in rois]
        xe=[r['left']+r['width'] for r in rois]
        ye=[r['top']+r['height'] for r in rois]
        bounds={'left':int(min(xs)), 'top':int(min(ys)),
                'width':int(max(xe)-min(xs)), 'height':int(max(ye)-min(ys))}
    delay=max(0,args.delay)
    if delay:
        print(f'Switch to TETR.IO now; capturing in {delay} seconds...')
        time.sleep(delay)
    with MSS() as capturer:
        img=np.asarray(capturer.grab(bounds))
        if region is not None:
            upcoming=read_queue_region(
                img,saturation_min=cfg.get('queue_saturation_min',65),
                value_min=cfg.get('queue_value_min',55))
        else:
            upcoming=read_queue_rois(img,rois,(bounds['left'],bounds['top']))
        held_roi=cfg.get('hold_piece_roi')
        if held_roi is not None:
            held_img=np.asarray(capturer.grab(held_roi))
            held=read_hold_view(held_img)
        else:
            held_img=None
            held=None
    print('NEXT:', ' '.join(upcoming) if upcoming else 'unreadable')
    print('HOLD:',f'{held.piece or "empty/unreadable"} ({held.appearance})' if held else 'not configured')
    print('HOLD ROI inferred during calibration:',cfg.get('hold_roi_inferred',False))
    # Save only selected HUD regions, not the entire personal desktop.
    cv2.imwrite('debug_v4_next.png',img)
    if held_img is not None:
        cv2.imwrite('debug_v4_hold.png',held_img)
    print('Saved debug_v4_next.png and (if configured) debug_v4_hold.png locally.')

if __name__=='__main__':main()
