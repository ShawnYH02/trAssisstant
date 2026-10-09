"""Interactive one-time crop of the visible board; use TETR.IO windowed mode."""
import argparse
import ctypes
import json
from pathlib import Path

import cv2
import numpy as np
from mss import MSS


BOARD_RATIO = 10 / 20


def dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def _aspect_fit(rect):
    """Return the largest centered 1:2 rectangle contained in rect."""
    left, top, original_width, original_height = map(int, rect)
    width, height = original_width, original_height
    if width / height > BOARD_RATIO:
        width = max(10, round(height * BOARD_RATIO))
    else:
        height = max(20, round(width / BOARD_RATIO))
    return (left + (original_width - width) // 2,
            top + (original_height - height) // 2, width, height)


def detect_board(screenshot, rough):
    """Find a board-shaped rectangle inside a user-supplied rough crop."""
    rx, ry, rw, rh = map(int, rough)
    roi = screenshot[ry:ry + rh, rx:rx + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blurred, 24, 80)
    masks = [edge]
    for invert in (False, True):
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)
        masks.append(mask)

    candidates = []
    rough_area = float(rw * rh)
    for mask, kernel_size in zip(masks, ((3, 3), (5, 5), (3, 3))):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(joined, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area_fraction = width * height / rough_area
            ratio = width / max(height, 1)
            if (area_fraction < 0.16 or width < 70 or height < 140 or
                    not 0.38 <= ratio <= 0.62):
                continue

            ratio_score = max(0.0, 1.0 - abs(ratio - BOARD_RATIO) / 0.12)
            center_error = (abs((x + width / 2) - rw / 2) / rw +
                            abs((y + height / 2) - rh / 2) / rh)
            center_score = max(0.0, 1.0 - center_error)
            area_score = min(area_fraction / 0.72, 1.0)
            score = 4 * ratio_score + 2 * area_score + center_score
            candidates.append((score, (x, y, width, height)))

    if not candidates:
        return _aspect_fit(rough), False

    _, local = max(candidates, key=lambda item: item[0])
    fitted = _aspect_fit(local)
    return (rx + fitted[0], ry + fitted[1], fitted[2], fitted[3]), True


def choose_board(screenshot):
    """Ask for a rough area and let the user confirm the detected board."""
    while True:
        print("Draw a LOOSE rectangle around the board, including a little padding.")
        print("Press Enter/Space to search; press C to cancel.")
        rough = tuple(map(int, cv2.selectROI(
            "Rough TETR.IO board area", screenshot,
            showCrosshair=True, fromCenter=False)))
        cv2.destroyAllWindows()
        if rough[2] <= 0 or rough[3] <= 0:
            raise SystemExit("No area selected. Re-run calibrate.py")

        board, detected = detect_board(screenshot, rough)
        left, top, width, height = board
        preview = screenshot.copy()
        cv2.rectangle(preview, (left, top), (left + width, top + height),
                      (80, 255, 80), 3)
        status = "AUTO-DETECTED" if detected else "CENTERED 1:2 FALLBACK"
        cv2.putText(preview, status, (max(8, left), max(28, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2,
                    cv2.LINE_AA)
        cv2.putText(preview,
                    "ENTER = accept    R = draw again    C = cancel",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Confirm detected TETR.IO board", preview)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (10, 13, 32):
                cv2.destroyAllWindows()
                return board
            if key in (ord("r"), ord("R")):
                cv2.destroyAllWindows()
                break
            if key in (ord("c"), ord("C"), 27):
                cv2.destroyAllWindows()
                raise SystemExit("Calibration cancelled")


def main():
    dpi_awareness()
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1,
                        help="MSS monitor index, 1=first display, 2=second, etc.")
    args = parser.parse_args()
    with MSS() as sct:
        if not 1 <= args.monitor < len(sct.monitors):
            raise SystemExit(f"Choose --monitor between 1 and {len(sct.monitors)-1}")
        mon = sct.monitors[args.monitor]
        grab = sct.grab(mon)
        screenshot = np.array(grab)[:, :, :3].copy()
        left, top, width, height = choose_board(screenshot)
        config = {"left": mon["left"] + left, "top": mon["top"] + top,
                  "width": width, "height": height,
                  "spawn_rows": 4,
                  "saturation_min": 72, "value_min": 70,
                  "gray_value_min": 108}
        Path("config.json").write_text(json.dumps(config, indent=2),
                                       encoding="utf-8")
        print("Saved config.json:", config)


if __name__ == "__main__":
    main()
