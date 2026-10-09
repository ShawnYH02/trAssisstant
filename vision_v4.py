"""Piece pose detection and HOLD/NEXT vision (offline/local learning prototype).

Do not interpret a low-brightness HOLD preview as an empty slot. A separate
tracker stores the piece identity and its HOLD cooldown state.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from tetris_core import ActivePiece, ROTATIONS, SPAWN, hue_to_piece
from queue_first import _piece_votes
from auto_layout_v4 import detect_next_group

PIECES = set('IJLOSTZ')


def read_queue_rois(screen_bgra: np.ndarray, rois: list[dict], origin: tuple[int,int]=(0,0),
                    saturation_min: int=70, value_min: int=60) -> Optional[tuple[str,...]]:
    """Individually detected preview ROIs, no equal-height slot assumption."""
    ox, oy = origin
    ans=[]
    for roi in rois:
        x,y = int(roi['left'])-ox, int(roi['top'])-oy
        w,h = int(roi['width']), int(roi['height'])
        patch=screen_bgra[max(0,y):max(0,y)+h, max(0,x):max(0,x)+w]
        p=_piece_votes(patch, min_pixels=9, saturation_min=saturation_min, value_min=value_min)
        if p is None:
            return None
        ans.append(p)
    return tuple(ans) if len(ans)>=2 else None


def read_queue_region(image: np.ndarray, saturation_min: int = 65,
                      value_min: int = 55) -> Optional[tuple[str, ...]]:
    """Find and read all preview pieces inside one live NEXT-region crop."""
    if image.size == 0 or image.ndim != 3 or image.shape[2] not in (3, 4):
        return None
    h, w = image.shape[:2]
    group = {'left': 0, 'top': 0, 'width': w, 'height': h}
    rois = detect_next_group(image, group, saturation_min, value_min)
    if len(rois) < 2:
        return None
    return read_queue_rois(image, rois, saturation_min=saturation_min,
                           value_min=value_min)


@dataclass(frozen=True)
class HoldView:
    piece: Optional[str]
    appearance: str  # bright/dim/empty/unknown


def _gray_preview_piece(image: np.ndarray) -> Optional[str]:
    """Identify a dim preview from its 4-square silhouette, not its color.

    Uses a tight HOLD icon ROI: no text, border, or background art. Ambiguous
    silhouettes return None so we do not corrupt known piece identity.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
    grey=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    h,w=grey.shape
    if min(h,w)<18:return None
    margin=max(2,min(h,w)//9)
    corners=np.concatenate((grey[:margin,:margin].ravel(),
                            grey[:margin,-margin:].ravel(),
                            grey[-margin:,:margin].ravel(),
                            grey[-margin:,-margin:].ravel()))
    bg=float(np.median(corners))
    binary=(grey.astype(np.float32)>bg+max(13.,bg*.16)).astype(np.uint8)
    # Ignore ROI border (where icon frames are sometimes drawn).
    binary[:margin,:]=0;binary[-margin:,:]=0
    binary[:,:margin]=0;binary[:,-margin:]=0
    if np.count_nonzero(binary)<max(35,int(grey.size*.035)):
        return None
    count,component,stats, _=cv2.connectedComponentsWithStats(binary,8)
    if count<=1:return None
    region_idx=max(range(1,count),key=lambda i:int(stats[i,cv2.CC_STAT_AREA]))
    x,y,bw,bh=map(int,stats[region_idx,:4])
    if bw<12 or bh<6:return None
    data=(component[y:y+bh,x:x+bw]==region_idx).astype(np.uint8)
    scored=[]
    for piece,shape in SPAWN.items():
        norm=_normalize(shape)
        sw=max(dx for dx,dy in norm)+1
        sh=max(dy for dx,dy in norm)+1
        square_ratio=(bw/sw)/(bh/sh)
        if not .62<=square_ratio<=1.6:continue
        predicted=np.zeros((bh,bw),dtype=np.uint8)
        for dx,dy in norm:
            xa=round(dx*bw/sw);xb=round((dx+1)*bw/sw)
            ya=round(dy*bh/sh);yb=round((dy+1)*bh/sh)
            predicted[ya:yb,xa:xb]=1
        intersection=np.count_nonzero((predicted>0)&(data>0))
        union=np.count_nonzero((predicted>0)|(data>0))
        score=intersection/max(1,union)-.06*abs(np.log(square_ratio))
        scored.append((score,piece))
    scored.sort(reverse=True)
    if not scored or scored[0][0]<.60:
        return None
    # Do not force a classification when two different shapes fit similarly.
    if len(scored)>1 and scored[0][0]-scored[1][0]<.08:
        return None
    return scored[0][1]


def read_hold_view(image: np.ndarray, previous: Optional[str] = None,
                   proposed_current: Optional[str] = None,
                   saturation_min: int=85, value_min: int=75) -> HoldView:
    """Read bright/dim HOLD while treating cooldown separately from identity."""
    if image.size == 0:
        return HoldView(previous, 'unknown')
    bright=_piece_votes(image, min_pixels=10, saturation_min=saturation_min,
                        value_min=value_min)
    if bright:
        return HoldView(bright, 'bright')
    dim=_piece_votes(image, min_pixels=8, saturation_min=24, value_min=24)
    if dim and dim == previous:
        return HoldView(dim,'dim')
    silhouette=_gray_preview_piece(image)
    if silhouette:
        if previous is None and proposed_current in PIECES:
            return HoldView(proposed_current,'dim')
        if previous is not None and silhouette != previous and silhouette != proposed_current:
            # Reject an unrelated gray silhouette; it is more likely HUD noise.
            return HoldView(previous,'unknown')
        return HoldView(silhouette,'dim')
    if dim and previous is None:
        return HoldView(dim,'dim')
    if previous:
        return HoldView(previous, 'dim' if dim else 'unknown')
    if proposed_current in PIECES:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
        grey=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        h,w=grey.shape
        margin=max(2,min(h,w)//9)
        bg=float(np.median(np.concatenate((grey[:margin,:margin].ravel(),
                  grey[:margin,-margin:].ravel(),grey[-margin:,:margin].ravel(),
                  grey[-margin:,-margin:].ravel()))))
        center=grey[margin:h-margin,margin:w-margin]
        foreground=int(np.count_nonzero(center.astype(float)>bg+max(13,bg*.16)))
        if foreground>=max(35,int(center.size*.07)):
            return HoldView(proposed_current, 'dim')
    return HoldView(None, 'empty')


def _normalize(cells):
    x0=min(x for x,_ in cells)
    y0=min(y for _,y in cells)
    return tuple(sorted((x-x0,y-y0) for x,y in cells))


def locate_expected_piece(labels: np.ndarray, expected: str,
                          previous: Optional[ActivePiece]=None) -> Optional[ActivePiece]:
    """Shape-fit the current tetromino using 3+ observed colored cells.

    The queue supplies identity. Hue/segmentation provide supporting evidence,
    not an exact-four-component gate. Returns None on weak/ambiguous evidence.
    """
    if expected not in ROTATIONS or labels.ndim != 2 or labels.shape[1] != 10:
        return None
    h,w=labels.shape
    candidates=[]
    for r,shape in enumerate(ROTATIONS[expected]):
        sh=max(y for _,y in shape)+1
        sw=max(x for x,_ in shape)+1
        for y in range(0,h-sh+1):
            for x in range(0,w-sw+1):
                labs=[str(labels[y+dy,x+dx]) for dx,dy in shape]
                color_count=sum(p in PIECES for p in labs)
                if color_count < 3:
                    continue
                correct=sum(p==expected for p in labs)
                missing=4-color_count
                # Require a cluster that is mostly consistent with piece hue,
                # or support from the previous position when color is unreliable.
                if correct < 2 and previous is None and color_count < 4:
                    continue
                positions=tuple(sorted((x+dx,y+dy) for dx,dy in shape))
                score=5.0*correct + 2.0*(color_count-correct) - 3.3*missing
                # The highest colored piece is usually active; avoid stack matches.
                score-=0.22*y
                if previous is not None and previous.name==expected:
                    drift=abs(x-previous.x)+abs(y-previous.y)
                    if drift<=5:
                        score+=4.0 - 0.45*drift
                    else:
                        score-=1.0*min(drift,12)
                candidates.append((score,ActivePiece(expected,r,x,y,positions)))
    if not candidates:
        return None
    candidates.sort(key=lambda item:item[0], reverse=True)
    top_score,top=candidates[0]
    if len(candidates)>1:
        runner=next((c for s,c in candidates[1:] if set(c.cells)!=set(top.cells)),None)
        runner_score=next((s for s,c in candidates[1:] if set(c.cells)!=set(top.cells)),None)
        if runner is not None and runner_score is not None and top_score-runner_score<0.8:
            # A one-cell horizontal/vertical shift of identical colors may be
            # legitimate; previous position helps break such ties, otherwise wait.
            if previous is None:
                return None
    return top


@dataclass
class PieceTrackerV4:
    """Debounce piece CONTENT independently of HOLD brightness/cooldown."""
    stable_frames: int = 3
    current: Optional[str] = None
    hold: Optional[str] = None
    queue: Optional[tuple[str, ...]] = None
    can_hold: Optional[bool] = None
    status: str = 'Learning NEXT queue'

    def __post_init__(self):
        if self.stable_frames < 1: raise ValueError('stable_frames must be >=1')
        self._queue_candidate = None
        self._queue_frames = 0
        self._hold_candidate = None
        self._hold_frames = 0
        self._hold_initialized = False
        self._pending_empty_hold = False

    def observe(self, next_queue: Optional[tuple[str,...]], hold: Optional[HoldView]=None):
        # Process HOLD before NEXT: first-time HOLD and its accompanying NEXT
        # shift often appear in the same rendered frame.
        if hold is not None and hold.piece in PIECES | {None} and hold.appearance != 'unknown':
            observed=hold.piece
            if observed==self._hold_candidate:
                self._hold_frames+=1
            else:
                self._hold_candidate=observed
                self._hold_frames=1
            if self._hold_frames>=self.stable_frames:
                if not self._hold_initialized:
                    # Merely starting the program with a pre-filled HOLD is not
                    # a swap event.
                    self.hold=observed
                    self._hold_initialized=True
                elif observed is not None and observed!=self.hold:
                    prior=self.hold
                    prior_current=self.current
                    self.hold=observed
                    if prior is None:
                        self._pending_empty_hold=True
                        self.current=None
                        self.can_hold=False
                        self.status='First HOLD: awaiting NEXT shift'
                    elif prior_current is not None and observed==prior_current:
                        self.current=prior
                        self.can_hold=False
                        self.status=f'HOLD swapped: CURRENT={prior}'
                    else:
                        self.current=None
                        self.can_hold=False
                        self.status='Unexpected HOLD contents; resynchronizing'
                # If HOLD becomes dim, the piece remains the same. A dark/blank
                # detection never clears an already-known piece.
        if next_queue is not None and len(next_queue)>=2 and all(p in PIECES for p in next_queue):
            nxt=tuple(next_queue)
            if nxt==self._queue_candidate:
                self._queue_frames+=1
            else:
                self._queue_candidate=nxt
                self._queue_frames=1
            if self._queue_frames>=self.stable_frames:
                if self.queue is None:
                    self.queue=nxt
                    self.status='NEXT learned; place one piece to synchronize'
                elif nxt!=self.queue:
                    shifts=[i for i in range(1,len(self.queue))
                            if self.queue[i:]==nxt[:len(self.queue)-i]]
                    if shifts:
                        n=min(shifts)
                        self.current=self.queue[n-1]
                        self.can_hold=not self._pending_empty_hold
                        self.status=f'Queue advanced {n}: CURRENT={self.current}'
                        self._pending_empty_hold=False
                    else:
                        self.current=None
                        self.can_hold=None
                        self.status='Queue discontinuity; awaiting resync'
                    self.queue=nxt
        return self.current
