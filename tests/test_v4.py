import cv2
import numpy as np
from tetris_core import ROTATIONS, ActivePiece
from vision_v4 import (HoldView, PieceTrackerV4, read_hold_view,
                       locate_expected_piece,read_queue_region,read_queue_rois)
from auto_layout_v4 import (detect_layout, detect_board_fullscreen,
                            detect_next_group)

HUES={'Z':0,'L':15,'O':30,'S':60,'I':90,'J':115,'T':150}

def color(name,value=230, sat=230):
    return cv2.cvtColor(np.uint8([[[HUES[name],sat,value]]]),cv2.COLOR_HSV2BGR)[0,0]

def feed(t,q,hold,n=2):
    for _ in range(n):t.observe(tuple(q),hold)


def test_t_detected_with_missing_fourth_cell():
    labels=np.full((24,10),'.',dtype='<U1')
    cells=[(4+x,1+y) for x,y in ROTATIONS['T'][0]]
    for x,y in cells[:3]:labels[y,x]='T'
    previous=ActivePiece('T',0,4,1,tuple(sorted(cells)))
    result=locate_expected_piece(labels,'T',previous)
    assert result is not None
    assert set(result.cells)==set(cells)


def test_t_detected_when_wrong_color_classified():
    labels=np.full((24,10),'.',dtype='<U1')
    cells=[(3+x,2+y) for x,y in ROTATIONS['T'][0]]
    for x,y in cells:labels[y,x]='Z'
    r=locate_expected_piece(labels,'T')
    assert r is not None and set(r.cells)==set(cells)


def test_previously_known_hold_survives_darkness():
    img=np.zeros((70,90,3),np.uint8)
    cv2.rectangle(img,(20,20),(65,48),tuple(map(int,color('T',value=48))),-1)
    result=read_hold_view(img,previous='T')
    assert result.piece=='T'
    assert result.appearance in ('dim','unknown')


def test_empty_hold_and_next_first_hold():
    t=PieceTrackerV4(stable_frames=2)
    feed(t,'IOTSL', HoldView(None,'empty'))
    feed(t,'OTSLJ', HoldView(None,'empty'))
    assert t.current=='I' and t.can_hold is True
    feed(t,'OTSLJ', HoldView('I','bright'))
    assert t.hold=='I' and t.current is None and t.can_hold is False
    feed(t,'TSLJZ', HoldView('I','dim'))
    assert t.current=='O' and t.hold=='I' and t.can_hold is False
    feed(t,'SLJZT', HoldView('I','dim'))
    assert t.current=='T' and t.can_hold is True


def test_swap_filled_hold_without_queue_shift():
    t=PieceTrackerV4(stable_frames=2)
    feed(t,'IOTSL',HoldView('Z','bright'))
    assert t.hold=='Z' and t.current is None  # initial hold is not an event
    feed(t,'OTSLJ',HoldView('Z','dim'))
    assert t.current=='I'
    feed(t,'OTSLJ',HoldView('I','bright'))
    assert t.current=='Z' and t.hold=='I' and t.can_hold is False
    feed(t,'OTSLJ',HoldView('I','unknown'))
    assert t.hold=='I' and t.current=='Z'


def test_invalid_queue_does_not_erase_hold():
    t=PieceTrackerV4(stable_frames=2)
    feed(t,'IOTSL',HoldView('T','bright'))
    assert t.hold=='T'
    feed(t,'IOTSL',HoldView('T','unknown'))
    assert t.hold=='T'
    t.observe(None, HoldView('T','unknown'))
    assert t.hold=='T'


def synthetic_screen():
    img=np.full((900,1600,3), (8,10,14),np.uint8)
    bx,by,bw,bh=660,120,260,520
    cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(80,100,115),3)
    for i in range(1,10):
        x=bx+round(i*bw/10)
        cv2.line(img,(x,by),(x,by+bh),(25,29,35),1)
    for i in range(1,20):
        y=by+round(i*bh/20)
        cv2.line(img,(bx,y),(bx+bw,y),(25,29,35),1)
    for i,p in enumerate('IOTSL'):
        px=1010;py=185+i*75
        cv2.rectangle(img,(px,py),(px+40,py+27),tuple(map(int,color(p))),-1)
    cv2.rectangle(img,(535,185),(575,212),tuple(map(int,color('Z'))),-1)
    return img


def test_fullscreen_board_and_queue():
    img=synthetic_screen()
    layout=detect_layout(img)
    assert layout is not None
    board=layout['board']
    assert abs(board['left']-660)<=8,board
    assert abs(board['top']-120)<=8,board
    assert len(layout['next_rois'])==5,layout
    assert layout['hold'] is not None and not layout['hold_inferred']
    q=read_queue_rois(img,layout['next_rois'])
    assert q==tuple('IOTSL'),q


def test_fullscreen_infers_empty_hold_roi():
    img=synthetic_screen()
    img[150:230,500:610]= (8,10,14)
    layout=detect_layout(img)
    assert layout is not None and layout['hold_inferred']
    assert layout['hold'] is not None


def test_manual_next_column_is_split_into_individual_previews():
    img=synthetic_screen()
    group={'left':995,'top':175,'width':70,'height':350}
    rois=detect_next_group(img,group)
    assert len(rois)==5,rois
    assert read_queue_rois(img,rois)==tuple('IOTSL')
    crop=img[group['top']:group['top']+group['height'],
             group['left']:group['left']+group['width']]
    assert read_queue_region(crop)==tuple('IOTSL')

def test_first_hold_grayscale_detected_using_current_identity():
    img=np.full((65,90,3), (12,12,12),np.uint8)
    cv2.rectangle(img,(19,20),(66,48),(53,53,53),-1)
    view=read_hold_view(img, previous=None, proposed_current='T')
    assert view==HoldView('T','dim')


def test_empty_hold_stays_empty():
    img=np.full((65,90,3), (12,12,12),np.uint8)
    assert read_hold_view(img,proposed_current='S')==HoldView(None,'empty')

def gray_piece(name,scale=13):
    from tetris_core import SPAWN
    shape=SPAWN[name]
    out=np.full((90,105,3),(11,11,11),np.uint8)
    w=max(x for x,y in shape)+1
    h=max(y for x,y in shape)+1
    x0=round((105-w*scale)/2)
    y0=round((90-h*scale)/2)
    for x,y in shape:
        cv2.rectangle(out,(x0+x*scale,y0+y*scale),
                     (x0+(x+1)*scale-1,y0+(y+1)*scale-1),(54,54,54),-1)
    return out


def test_held_gray_t_shape_instead_of_previous_i():
    assert read_hold_view(gray_piece('T'),previous='I',proposed_current='T') == HoldView('T','dim')


def test_all_grayscale_tetromino_silhouettes():
    for piece in 'IJLOSTZ':
        observed=read_hold_view(gray_piece(piece),previous=None)
        assert observed.piece==piece, (piece,observed)
