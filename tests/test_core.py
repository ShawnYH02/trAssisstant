import numpy as np
from tetris_core import (
    ROWS, COLS, ROTATIONS, read_cells, find_active, find_best,
    occupied_without_active, apply_placement, hard_drop, normalize, shift_active,
    ActivePiece, Recommendation, finesse_actions,
)


def test_rotations_are_tetrominoes():
    assert len(ROTATIONS) == 7
    for rotations in ROTATIONS.values():
        assert len(rotations) == 4
        for shape in rotations:
            assert len(set(shape)) == 4
            assert all(x >= 0 and y >= 0 for x, y in shape)
            assert shape == normalize(shape)


def test_drop_and_line_clear():
    stack = np.zeros((ROWS, COLS), dtype=bool)
    stack[-1, :6] = True
    assert hard_drop(stack, ROTATIONS["I"][0], 6) == 19
    candidate = find_best(stack, "I")
    assert candidate is not None
    assert candidate.cleared == 1
    assert set(candidate.cells) == {(6, 19), (7, 19), (8, 19), (9, 19)}
    final, lines = apply_placement(stack, candidate.cells)
    assert lines == 1
    assert not final.any()


def test_recognize_falling_piece():
    labels = np.full((ROWS, COLS), ".", dtype="<U1")
    labels[19, 0:3] = ["G", "G", "G"]
    coords = ((3, 2), (2, 3), (3, 3), (4, 3))
    for x, y in coords:
        labels[y, x] = "T"
    active = find_active(labels)
    assert active is not None
    assert active.name == "T"
    assert active.rotation == 0
    board = occupied_without_active(labels, active)
    assert board.sum() == 3
    assert all(not board[y, x] for x, y in coords)


def test_screen_recognizer_synthetic_colors():
    import cv2
    img = np.zeros((400, 200, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    for x, y in ((3, 2), (2, 3), (3, 3), (4, 3)):
        # magenta BGR: represents T
        img[y*20:(y+1)*20, x*20:(x+1)*20, :3] = (210, 30, 210)
    cells = read_cells(img)
    active = find_active(cells)
    assert active is not None
    assert active.name == "T"
    assert set(active.cells) == {(3, 2), (2, 3), (3, 3), (4, 3)}


def test_unrecognized_shape_is_ignored():
    labels = np.full((ROWS, COLS), ".", dtype="<U1")
    for x, y in ((0, 0), (1, 0), (2, 0), (3, 1)):
        labels[y, x] = "T"
    assert find_active(labels) is None


def test_piece_can_be_detected_above_visible_board():
    spawn_rows = 4
    labels = np.full((ROWS + spawn_rows, COLS), ".", dtype="<U1")
    for x, y in ((3, 2), (2, 3), (3, 3), (4, 3)):
        labels[y, x] = "T"
    captured = find_active(labels)
    assert captured is not None
    active = shift_active(captured, -spawn_rows)
    assert active.y == -2
    assert min(y for _, y in active.cells) == -2
    assert not occupied_without_active(labels[spawn_rows:], active).any()


def test_finesse_uses_das_and_a_correction_for_distant_target():
    active = ActivePiece("T", 0, 3, -1, ())
    best = Recommendation("T", 0, 6, 18, (), 0.0, 0)
    assert finesse_actions(active, best) == ("DAS R", "Tap L", "Hard drop")


def test_finesse_combines_rotation_and_das():
    active = ActivePiece("T", 0, 3, -1, ())
    best = Recommendation("T", 1, 8, 17, (), 0.0, 0)
    assert finesse_actions(active, best) == ("CW", "DAS R", "Hard drop")
