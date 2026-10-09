"""One full-screen grab, detect board + NEXT + HOLD, then confirm on screenshot."""
from __future__ import annotations
import argparse
import ctypes
import json
from pathlib import Path
import cv2
import numpy as np
from mss import MSS
from auto_layout_v4 import detect_layout, rect


def draw_layout(screen, data):
    im=screen.copy()
    targets=[('BOARD',data['board'],(90,255,90))]
    targets += [(f'NEXT {i+1}',r,(255,180,50)) for i,r in enumerate(data['next_rois'])]
    if data['hold']:
        targets.append(('HOLD (check)' if data['hold_inferred'] else 'HOLD',data['hold'],(255,80,240)))
    for name,r,color in targets:
        x,y,w,h=(r[k] for k in ('left','top','width','height'))
        cv2.rectangle(im,(x,y),(x+w,y+h),color,2)
        cv2.putText(im,name,(max(0,x),max(20,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    cv2.putText(im,'ENTER accept | R manual edit | ESC cancel',(15,32),
                cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    return im


def select(screen,title):
    box=cv2.selectROI(title,screen,showCrosshair=True,fromCenter=False)
    cv2.destroyWindow(title)
    return rect(*map(int,box)) if box[2]>0 and box[3]>0 else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--monitor',type=int,default=1)
    args=ap.parse_args()
    if hasattr(ctypes,'windll'):
        try: ctypes.windll.user32.SetProcessDPIAware()
        except Exception: pass
    with MSS() as capturer:
        if not 1<=args.monitor<len(capturer.monitors):
            raise SystemExit('Invalid monitor number')
        mon=capturer.monitors[args.monitor]
        screenshot=np.array(capturer.grab(mon))[:,:,:3].copy()
    layout=detect_layout(screenshot)
    if layout is None:
        print('Could not find board; manual selection on same full-screen image.')
        board=select(screenshot,'Select board (10x20 playfield)')
        if board is None: raise SystemExit('No board selected')
        layout=dict(board=board,next_rois=[],hold=None,hold_inferred=True,confidence=0)
    while True:
        cv2.imshow('One-screen calibration',draw_layout(screenshot,layout))
        key=cv2.waitKey(0)&255
        cv2.destroyWindow('One-screen calibration')
        if key in (13,10,32):
            if len(layout['next_rois'])<2 or layout['hold'] is None:
                print('NEXT/HOLD could not be inferred; press R to mark them.')
                continue
            break
        if key==27: raise SystemExit('Cancelled; config unchanged.')
        if key in (ord('r'),ord('R')):
            b=select(screenshot,'Board (ESC to retain automatic)')
            if b is not None: layout['board']=b
            print('Draw NEXT previews in order. Press ESC when finished.')
            slots=[]
            for i in range(1,6):
                r=select(screenshot,f'NEXT #{i} (ESC to stop)')
                if r is None:break
                slots.append(r)
            if len(slots)>=2:layout['next_rois']=slots
            hold=select(screenshot,'HOLD area (select even if empty/dark)')
            if hold is not None:
                layout['hold']=hold
                layout['hold_inferred']=False
    cfg_path=Path('config.json')
    old=json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}
    ox,oy=mon['left'],mon['top']
    board=layout['board']
    saved={k:int(board[k]+(ox if k=='left' else oy if k=='top' else 0)) for k in ('left','top','width','height')}
    def offset(r):
        return rect(r['left']+ox,r['top']+oy,r['width'],r['height'])
    old.update(saved)
    old['spawn_rows']=old.get('spawn_rows',4)
    old['next_piece_rois']=[offset(r) for r in layout['next_rois']]
    old['hold_piece_roi']=offset(layout['hold'])
    old['hold_roi_inferred']=layout['hold_inferred']
    old['queue_stable_frames']=old.get('queue_stable_frames',3)
    old.setdefault('saturation_min',72)
    old.setdefault('value_min',70)
    old.setdefault('gray_value_min',108)
    if cfg_path.exists():
        import shutil
        shutil.copy2(cfg_path,'config.json.before_v4.bak')
    cfg_path.write_text(json.dumps(old,indent=2),encoding='utf-8')
    print('Saved config.json. Board:',saved)
    print('NEXT slots:',len(layout['next_rois']), '| HOLD inferred:',layout['hold_inferred'])
    print('No screen image was uploaded. Verify preview rectangles before running the overlay.')

if __name__=='__main__':main()
