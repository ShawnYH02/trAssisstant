"""Single-frame automatic layout detector for 10x20 board and colorful HUD.

Layout is only a *proposal*: a human reviews rectangles before saving.
A screenshot showing a dark/empty HOLD can only infer its ROI geometrically.
"""
from __future__ import annotations
import cv2
import numpy as np
from tetris_core import hue_to_piece


def rect(left,top,width,height):
    return dict(left=int(left),top=int(top),width=int(width),height=int(height))


def detect_next_group(frame, group, saturation_min=65, value_min=55):
    """Split one manually selected NEXT column into per-piece rectangles."""
    fh, fw = frame.shape[:2]
    gx = max(0, int(group['left']))
    gy = max(0, int(group['top']))
    gw = min(int(group['width']), fw - gx)
    gh = min(int(group['height']), fh - gy)
    if gw < 8 or gh < 20:
        return []
    hsv = cv2.cvtColor(frame[gy:gy + gh, gx:gx + gw, :3], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] >= saturation_min) &
            (hsv[:, :, 2] >= value_min)).astype(np.uint8) * 255
    kernel_size = max(3, round(gw * .03)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        patch = hsv[y:y + height, x:x + width]
        colored = ((patch[:, :, 1] >= saturation_min) &
                   (patch[:, :, 2] >= value_min))
        hues = patch[:, :, 0][colored]
        if len(hues) < max(9, int(gw * gh * .0005)):
            continue
        votes = {}
        for hue in hues:
            piece = hue_to_piece(float(hue))
            if piece:
                votes[piece] = votes.get(piece, 0) + 1
        if not votes:
            continue
        dominant = max(votes.values())
        if dominant < .62 * sum(votes.values()):
            continue
        pad = max(2, round(min(gw, gh) * .01))
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(gw, x + width + pad)
        bottom = min(gh, y + height + pad)
        found.append(rect(gx + left, gy + top, right - left, bottom - top))
    found.sort(key=lambda roi: roi['top'])
    return found if 2 <= len(found) <= 8 else []


def detect_board_fullscreen(frame:np.ndarray):
    """Return board rectangle, confidence; None for no credible 10x20 region."""
    h,w=frame.shape[:2]
    bgr=frame[:,:,:3]
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    edge=cv2.Canny(blur,30,100)
    masks=[edge]
    for inverse in (False,True):
        flag=cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
        _,binary=cv2.threshold(blur,0,255, flag|cv2.THRESH_OTSU)
        masks.append(binary)
    candidates=[]
    min_h=max(200,int(h*0.27))
    for mask in masks:
        joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
        contours,_=cv2.findContours(joined,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,bw,bh=cv2.boundingRect(cnt)
            if bh<min_h or bw<100 or bw>0.66*w or bh>0.95*h:
                continue
            ratio=bw/max(bh,1)
            if not .425<=ratio<=.575:
                continue
            cx=x+bw/2
            cy=y+bh/2
            centered=1-abs(cx-w/2)/(w/2)
            vcentered=1-abs(cy-h/2)/(h/2)
            ratio_score=1-abs(ratio-.5)/.075
            # Check that the rectangle has a relatively strong physical border.
            inset=max(2,round(bw/100))
            x0=max(0,x-inset); x1=min(w,x+bw+inset)
            y0=max(0,y-inset); y1=min(h,y+bh+inset)
            crop=edge[y0:y1,x0:x1]
            if crop.size==0: continue
            rim=np.concatenate([crop[:max(2,inset*2)].ravel(),
                                crop[-max(2,inset*2):].ravel(),
                                crop[:, :max(2,inset*2)].ravel(),
                                crop[:, -max(2,inset*2):].ravel()])
            border=float(np.count_nonzero(rim))/max(1,rim.size)
            score=4*ratio_score+1.5*centered+.5*vcentered+min(bh/h,1)*2+min(border,0.5)
            candidates.append((score,rect(x,y,bw,bh)))
    if not candidates:
        return None,0.
    candidates.sort(key=lambda p:p[0],reverse=True)
    score,roi=candidates[0]
    return roi,round(min(1.,score/8),3)


def _colored_regions(frame, board, side):
    """Find piece-colored contiguous blobs outside the playfield."""
    fh,fw=frame.shape[:2]
    bgr=frame[:,:,:3]
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    x,y,bw,bh=(board[k] for k in ('left','top','width','height'))
    cell=bw/10
    if side=='right':
        x0=min(fw,max(0,x+bw+int(.04*bw)))
        x1=min(fw,x+bw+int(1.1*bw))
    else:
        x0=max(0,x-int(1.1*bw))
        x1=max(0,x-int(.04*bw))
    y0=max(0,int(y-.22*bh))
    y1=min(fh,int(y+.82*bh))
    if x1<=x0 or y1<=y0:
        return []
    crop=hsv[y0:y1,x0:x1]
    valid=(crop[:,:,1]>=70)&(crop[:,:,2]>=70)
    # Get rid of isolated colorful text pixels, join touching blocks.
    mask=(valid.astype(np.uint8)*255)
    k=max(3,int(round(cell*.19))|1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for contour in contours:
        bx,by,bwidth,bheight=cv2.boundingRect(contour)
        area=bwidth*bheight
        if area<cell*cell*.22 or area>cell*cell*35:
            continue
        if not .42<=bwidth/max(bheight,1)<=5:
            continue
        roi_hsv=crop[by:by+bheight,bx:bx+bwidth]
        good=(roi_hsv[:,:,1]>=70)&(roi_hsv[:,:,2]>=70)
        votes={}
        for hv in roi_hsv[:,:,0][good]:
            name=hue_to_piece(float(hv))
            if name: votes[name]=votes.get(name,0)+1
        if not votes:
            continue
        piece,n=max(votes.items(),key=lambda item:item[1])
        if n<max(8,int(cell*cell*.08)) or n/sum(votes.values())<.68:
            continue
        pad=max(2,round(cell*.12))
        r=rect(x0+bx-pad,y0+by-pad,bwidth+2*pad,bheight+2*pad)
        result.append((piece,r))
    return result


def detect_hud(frame,board):
    """Return individual NEXT slots, HOLD ROI and whether HOLD is inferred."""
    right=_colored_regions(frame,board,'right')
    left=_colored_regions(frame,board,'left')
    x,y,bw,bh=(board[k] for k in ('left','top','width','height'))
    # Prefer a vertical column; any point in the same column can be a peer.
    best=[]
    for _,r in right:
        center=r['left']+r['width']/2
        cohort=[p for p in right if abs(p[1]['left']+p[1]['width']/2-center)<bw*.22]
        cohort.sort(key=lambda p:p[1]['top'])
        filtered=[]
        for item in cohort:
            if not filtered or item[1]['top']-filtered[-1][1]['top']>bw*.07:
                filtered.append(item)
        if len(filtered)>len(best):
            best=filtered
    best=best[:5]
    if len(best)<3:
        return [],None,False
    previews=[r for _,r in best]
    first=previews[0]
    hold_candidates=[r for piece,r in left
                     if abs((r['top']+r['height']/2)-(first['top']+first['height']/2))<bh*.25]
    if hold_candidates:
        hold=min(hold_candidates,key=lambda r:abs((r['top']+r['height']/2)-(first['top']+first['height']/2)))
        inferred=False
    else:
        # Empty HOLD: estimate a mirror-image slot; confirm manually on screen.
        gap=first['left']-(x+bw)
        hold=rect(x-gap-first['width'],first['top'],first['width'],first['height'])
        inferred=True
    return previews,hold,inferred


def detect_layout(frame):
    board,confidence=detect_board_fullscreen(frame)
    if board is None: return None
    next_rois,hold,inferred=detect_hud(frame,board)
    return dict(board=board, next_rois=next_rois, hold=hold,
                hold_inferred=inferred,confidence=confidence)
