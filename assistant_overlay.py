"""Windows-only visual training prototype. No inputs are sent to the game."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import signal
import sys
import threading
from pathlib import Path

import numpy as np
from mss import MSS
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from tetris_core import read_cells, occupied_without_active, shift_active
from solver_v2 import find_best_v2, describe_action_v2
from queue_first import (QueueTracker, UNOBSERVED, find_active_matching,
                         read_hold_piece, read_next_queue)


class Overlay(QWidget):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.target = None
        self.text = "Looking for a complete falling tetromino..."
        self.subtitle = "Solver V2 / collision-checked path / no lookahead"
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
        label_width = min(620, self.width())
        label_x = max(0, min(left, self.width() - label_width))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 13, 21, 222))
        painter.drawRoundedRect(QRectF(label_x, label_y, label_width, 56), 8, 8)
        painter.setPen(QColor(239, 249, 251))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(QRectF(label_x+12, label_y+4, label_width-24, 25),
                         Qt.AlignmentFlag.AlignVCenter, self.text)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(161, 184, 188))
        painter.drawText(QRectF(label_x+12, label_y+29, label_width-24, 20),
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
    tracker = QueueTracker(stable_frames=max(1, int(cfg.get("queue_stable_frames", 3))))
    queue_roi = cfg.get("next_queue_roi")
    hold_roi = cfg.get("hold_piece_roi")
    queue_slots = int(cfg.get("next_queue_slots", 5))
    timer = QTimer()
    shutdown_timer = QTimer()
    stop_requested = threading.Event()

    def tick():
        try:
            screenshot = np.asarray(capture.grab(region))
            labels = read_cells(screenshot,
                                saturation_min=cfg.get("saturation_min", 72),
                                value_min=cfg.get("value_min", 70),
                                gray_value_min=cfg.get("gray_value_min", 108),
                                rows=20 + spawn_rows)
            if queue_roi is None:
                overlay.target = None
                overlay.text = "NEXT not calibrated: run python calibrate_queue.py"
                overlay.subtitle = "Board calibration is preserved; NEXT needs its own crop"
            else:
                next_image = np.asarray(capture.grab(queue_roi))
                upcoming = read_next_queue(
                    next_image, slots=queue_slots,
                    saturation_min=cfg.get("queue_saturation_min", 95),
                    value_min=cfg.get("queue_value_min", 90),
                    min_pixels=cfg.get("queue_min_pixels", 12))
                held = (read_hold_piece(
                    np.asarray(capture.grab(hold_roi)),
                    saturation_min=cfg.get("queue_saturation_min", 95),
                    value_min=cfg.get("queue_value_min", 90),
                    min_pixels=cfg.get("queue_min_pixels", 12))
                    if hold_roi is not None else UNOBSERVED)
                current = tracker.observe(upcoming, held)
                queue_text = " ".join(tracker.queue or ()) or "unreadable"
                if current is None:
                    overlay.target = None
                    overlay.text = tracker.status
                    overlay.subtitle = f"NEXT: {queue_text} | CURRENT unknown"
                else:
                    active = find_active_matching(labels, current)
                    if active is None:
                        overlay.target = None
                        overlay.text = f"CURRENT={current} | cannot locate four cells"
                        overlay.subtitle = (f"NEXT: {queue_text} | identity known; "
                                            "pose not visible")
                    else:
                        active = shift_active(active, -spawn_rows)
                        visible_labels = labels[spawn_rows:].copy()
                        stack = occupied_without_active(visible_labels, active)
                        best = find_best_v2(stack, active)
                        if best is None:
                            overlay.target = None
                            overlay.text = f"CURRENT={current} | no reachable placement"
                            overlay.subtitle = tracker.status
                        else:
                            overlay.target = best.cells
                            overlay.text = describe_action_v2(active, best)
                            overlay.subtitle = (f"NEXT: {queue_text} | "
                                                f"score {best.score:.1f} | "
                                                f"{best.cleared} lines")
            overlay.update()
        except Exception as exc:
            overlay.target = None
            overlay.text = f"Capture error: {type(exc).__name__}"
            overlay.subtitle = str(exc)[:80]
            overlay.update()

    timer.timeout.connect(tick)
    timer.start(80)  # ~12.5 Hz; lower than 60 FPS to keep CPU and UI responsive

    def request_stop(*_args):
        stop_requested.set()

    def stop_if_requested():
        if stop_requested.is_set():
            shutdown_timer.stop()
            app.quit()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    # Python signals can be delayed while Qt owns the Windows event loop. A
    # native console callback only sets a thread-safe flag; Qt performs the
    # actual shutdown on its main thread during the next poll.
    console_handler = None
    try:
        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        @handler_type
        def console_handler(event):
            if event in (0, 1):  # CTRL_C_EVENT / CTRL_BREAK_EVENT
                stop_requested.set()
                return True
            return False

        ctypes.windll.kernel32.SetConsoleCtrlHandler(console_handler, True)
    except (AttributeError, OSError):
        console_handler = None

    shutdown_timer.timeout.connect(stop_if_requested)
    shutdown_timer.start(50)
    print("Running. Ctrl+C or Ctrl+Break in this terminal stops the overlay.")
    tick()
    exit_code = 0
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        request_stop()
    finally:
        timer.stop()
        shutdown_timer.stop()
        overlay.close()
        capture.close()
        if console_handler is not None:
            try:
                ctypes.windll.kernel32.SetConsoleCtrlHandler(console_handler, False)
            except (AttributeError, OSError):
                pass
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
