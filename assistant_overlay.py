"""Windows-only visual training prototype. No inputs are sent to the game."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import signal
import sys
from pathlib import Path

import numpy as np
from mss import MSS
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from tetris_core import (read_cells, find_active, find_best,
                         occupied_without_active, describe_action, shift_active)


class Overlay(QWidget):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.target = None
        self.text = "Looking for a complete falling tetromino..."
        self.subtitle = "Default skin / one piece / hard drops only"
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool |
                            Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(QApplication.primaryScreen().virtualGeometry())
        self.show()
        self.exclude_from_capture()

    def exclude_from_capture(self):
        """Best effort: omit overlay from Windows screen captures (Win 10 2004+)."""
        try:
            func = ctypes.windll.user32.SetWindowDisplayAffinity
            func.argtypes = (wintypes.HWND, wintypes.DWORD)
            func.restype = wintypes.BOOL
            ok = func(int(self.winId()), 0x11)
            if not ok:
                print("Note: capture exclusion unavailable; only cell outlines are drawn.")
        except (AttributeError, OSError):
            print("Note: capture exclusion unavailable; only cell outlines are drawn.")

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left = self.cfg["left"] - self.x()
        top = self.cfg["top"] - self.y()
        cw = self.cfg["width"] / 10
        ch = self.cfg["height"] / 20
        if self.target is not None:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 255, 212, 245), max(2.0, min(cw, ch) * 0.10)))
            for x, y in self.target:
                painter.drawRoundedRect(QRectF(left + x*cw + 2, top + y*ch + 2,
                                                cw - 4, ch - 4), 2, 2)
        # Keep text outside both the playfield and the extra spawn-row capture.
        spawn_height = self.cfg["height"] / 20 * self.cfg.get("spawn_rows", 4)
        label_y = (top - spawn_height - 63 if top >= spawn_height + 67
                   else top + self.cfg["height"] + 8)
        label_y = max(0, min(label_y, self.height()-57))
        label_x = max(0, min(left, self.width()-440))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 13, 21, 222))
        painter.drawRoundedRect(QRectF(label_x, label_y, 440, 56), 8, 8)
        painter.setPen(QColor(239, 249, 251))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(QRectF(label_x+12, label_y+4, 418, 25),
                         Qt.AlignmentFlag.AlignVCenter, self.text)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(161, 184, 188))
        painter.drawText(QRectF(label_x+12, label_y+29, 418, 20),
                         Qt.AlignmentFlag.AlignVCenter, self.subtitle)
        painter.end()


def dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def main():
    if sys.platform != "win32":
        raise SystemExit("This overlay starter targets Windows only.")
    dpi_awareness()
    path = Path("config.json")
    if not path.exists():
        raise SystemExit("Missing config.json. Run 'python calibrate.py' first.")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    app = QApplication(sys.argv)
    overlay = Overlay(cfg)
    capture = MSS()
    spawn_rows = max(0, int(cfg.get("spawn_rows", 4)))
    spawn_pixels = round(cfg["height"] / 20 * spawn_rows)
    region = {"left": int(cfg["left"]),
              "top": int(cfg["top"] - spawn_pixels),
              "width": int(cfg["width"]),
              "height": int(cfg["height"] + spawn_pixels)}
    timer = QTimer()

    def tick():
        try:
            screenshot = np.asarray(capture.grab(region))
            labels = read_cells(screenshot,
                                saturation_min=cfg.get("saturation_min", 72),
                                value_min=cfg.get("value_min", 70),
                                gray_value_min=cfg.get("gray_value_min", 108),
                                rows=20 + spawn_rows)
            active = find_active(labels)
            if active is None:
                overlay.target = None
                overlay.text = "Waiting: piece not detected with confidence"
                overlay.subtitle = "Try default skin, visible board and no bloom effects"
            else:
                active = shift_active(active, -spawn_rows)
                visible_labels = labels[spawn_rows:].copy()
                stack = occupied_without_active(visible_labels, active)
                best = find_best(stack, active.name)
                if best is None:
                    overlay.target = None
                    overlay.text = "No safe hard-drop found"
                else:
                    overlay.target = best.cells
                    overlay.text = describe_action(active, best)
                    overlay.subtitle = (f"Heuristic score {best.score:.1f} | "
                                        f"{best.cleared} lines | not exact SRS+ instructions")
            overlay.update()
        except Exception as exc:
            overlay.target = None
            overlay.text = f"Capture error: {type(exc).__name__}"
            overlay.subtitle = str(exc)[:80]
            overlay.update()

    timer.timeout.connect(tick)
    timer.start(80)  # ~12.5 Hz; lower than 60 FPS to keep CPU and UI responsive
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    app.aboutToQuit.connect(capture.close)
    print("Running. Ctrl+C in this terminal stops the overlay.")
    tick()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
