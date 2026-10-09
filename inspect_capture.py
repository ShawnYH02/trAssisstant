"""Save a one-frame playfield crop and print what the recognizer sees."""
import json
from pathlib import Path

import cv2
import numpy as np
from mss import MSS

from tetris_core import (read_cells, find_active, find_best,
                         occupied_without_active, describe_action, shift_active)


def main():
    cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
    spawn_rows = max(0, int(cfg.get("spawn_rows", 4)))
    spawn_pixels = round(cfg["height"] / 20 * spawn_rows)
    region = {"left": int(cfg["left"]),
              "top": int(cfg["top"] - spawn_pixels),
              "width": int(cfg["width"]),
              "height": int(cfg["height"] + spawn_pixels)}
    with MSS() as sct:
        screenshot = np.asarray(sct.grab(region))
    cv2.imwrite("debug_board.png", screenshot)
    cells = read_cells(screenshot,
                       cfg.get("saturation_min", 72),
                       cfg.get("value_min", 70),
                       cfg.get("gray_value_min", 108),
                       rows=20 + spawn_rows)
    print(f"Detected {20 + spawn_rows}x10 capture; first {spawn_rows} rows are above the board.")
    print("'.' is empty and 'G' is gray garbage:")
    print("\n".join("".join(row) for row in cells))
    active = find_active(cells)
    if active:
        active = shift_active(active, -spawn_rows)
        best = find_best(occupied_without_active(cells[spawn_rows:].copy(), active),
                         active.name)
        print("Detected:", active)
        print("Recommendation:", describe_action(active, best))
    else:
        print("No intact 4-cell active piece found.")
    print("Saved debug_board.png to inspect crop alignment and colors.")


if __name__ == "__main__":
    main()
