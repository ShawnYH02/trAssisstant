"""Windows-only visual training prototype. No inputs are sent to the game."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from mss import MSS
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from tetris_core import read_cells, occupied_without_active, shift_active
from solver_v6 import SearchSettingsV6, describe_action_v6, find_best_v6
from solver_v7 import (SearchSettingsV7, describe_action_v7, find_best_v7,
                       native_path)
from queue_first import read_next_queue
from vision_v4 import (PieceTrackerV4, locate_expected_piece, read_hold_view,
                       read_queue_region, read_queue_rois)


GHOST_COLORS = {
    "I": (35, 220, 235),
    "O": (245, 210, 40),
    "T": (175, 80, 225),
    "S": (65, 205, 90),
    "Z": (235, 65, 70),
    "J": (65, 105, 230),
    "L": (245, 145, 35),
}


class Overlay(QWidget):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.physical_monitor, self.board_screen = self._find_board_screen()
        self.screen_scale = float(self.board_screen.devicePixelRatio())
        self.target = None
        self.target_name = None
        self.next_target = None
        self.next_target_name = None
        self.third_target = None
        self.third_target_name = None
        self.text = "Looking for a complete falling tetromino..."
        self.subtitle = "Solver V6 / root-diverse lookahead / collision-checked CURRENT"
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool |
                            Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # Keep this top-level window wholly on the board's screen. A desktop-wide
        # Qt window is rendered at one device-pixel ratio, which offsets drawings
        # when Windows monitors use different display scaling percentages.
        self.setGeometry(self.board_screen.geometry())
        self.show()
        if self.windowHandle() is not None:
            self.windowHandle().setScreen(self.board_screen)
            self.setGeometry(self.board_screen.geometry())
        self.exclude_from_capture()

        left, top, width, height = self.board_geometry()
        print(f"Overlay mapped to {self.board_screen.name()}: "
              f"board=({left:.1f}, {top:.1f}, {width:.1f}, {height:.1f}), "
              f"scale={self.screen_scale:.2f}")

    def _find_board_screen(self):
        """Match the physical MSS monitor containing the board to a Qt screen."""
        center_x = self.cfg["left"] + self.cfg["width"] / 2
        center_y = self.cfg["top"] + self.cfg["height"] / 2
        with MSS() as capture:
            monitors = [dict(monitor) for monitor in capture.monitors[1:]]
        monitor = next(
            (item for item in monitors
             if (item["left"] <= center_x < item["left"] + item["width"] and
                 item["top"] <= center_y < item["top"] + item["height"])),
            None)
        if monitor is None:
            raise ValueError("Calibrated board is outside the current monitor layout")

        def match_score(screen):
            geometry = screen.geometry()
            scale = float(screen.devicePixelRatio())
            physical_width = round(geometry.width() * scale)
            physical_height = round(geometry.height() * scale)
            return (abs(geometry.x() - monitor["left"]) +
                    abs(geometry.y() - monitor["top"]) +
                    abs(physical_width - monitor["width"]) +
                    abs(physical_height - monitor["height"]))

        screens = QApplication.instance().screens()
        if not screens:
            raise RuntimeError("Qt did not report any screens")
        return monitor, min(screens, key=match_score)

    def board_geometry(self):
        """Return the physical board rectangle in this screen's Qt coordinates."""
        scale = self.screen_scale
        monitor = self.physical_monitor
        return ((self.cfg["left"] - monitor["left"]) / scale,
                (self.cfg["top"] - monitor["top"]) / scale,
                self.cfg["width"] / scale,
                self.cfg["height"] / scale)

    def clear_targets(self):
        self.target = None
        self.target_name = None
        self.next_target = None
        self.next_target_name = None
        self.third_target = None
        self.third_target_name = None

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
        left, top, width, height = self.board_geometry()
        cw = width / 10
        ch = height / 20
        # Draw later projections first so the immediate move remains clearest
        # where planned placements overlap.
        ghosts = ((self.third_target_name, self.third_target,
                   Qt.PenStyle.DotLine, 4),
                  (self.next_target_name, self.next_target,
                   Qt.PenStyle.DashLine, 3),
                  (self.target_name, self.target,
                   Qt.PenStyle.SolidLine, 2))
        for name, cells, line_style, inset in ghosts:
            if cells is None:
                continue
            red, green, blue = GHOST_COLORS.get(name, (220, 235, 240))
            painter.setBrush(QColor(red, green, blue, 25))
            pen = QPen(QColor(red, green, blue, 245),
                       max(2.0, min(cw, ch) * 0.09))
            pen.setStyle(line_style)
            painter.setPen(pen)
            for x, y in cells:
                painter.drawRoundedRect(
                    QRectF(left + x*cw + inset, top + y*ch + inset,
                           cw - 2*inset, ch - 2*inset), 2, 2)
        # Keep text outside both the playfield and the extra spawn-row capture.
        spawn_height = height / 20 * self.cfg.get("spawn_rows", 4)
        label_y = (top - spawn_height - 63 if top >= spawn_height + 67
                   else top + height + 8)
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


def main():
    if sys.platform != "win32":
        raise SystemExit("This overlay starter targets Windows only.")
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
    tracker = PieceTrackerV4(
        stable_frames=max(1, int(cfg.get("queue_stable_frames", 3))))
    next_region = cfg.get("next_queue_region")
    next_rois = cfg.get("next_piece_rois") or []
    queue_roi = cfg.get("next_queue_roi")
    hold_roi = cfg.get("hold_piece_roi")
    queue_slots = int(cfg.get("next_queue_slots", 5))
    previous_active = None
    previous_name = None
    previous_queue = None
    if next_rois:
        lefts = [roi["left"] for roi in next_rois]
        tops = [roi["top"] for roi in next_rois]
        rights = [roi["left"] + roi["width"] for roi in next_rois]
        bottoms = [roi["top"] + roi["height"] for roi in next_rois]
        hud_box = {"left": int(min(lefts)), "top": int(min(tops)),
                   "width": int(max(rights) - min(lefts)),
                   "height": int(max(bottoms) - min(tops))}
    else:
        hud_box = None
    native_enabled = native_path().is_file()
    if native_enabled:
        search_settings = SearchSettingsV7(
            depth=int(cfg.get("search_depth", 5)),
            beam_width=int(cfg.get("beam_width", 36)),
            time_budget_ms=float(cfg.get("search_budget_ms", 250)),
            allow_hold=bool(cfg.get("search_allow_hold", True)))
        solve_function = find_best_v7
        describe_function = describe_action_v7
        overlay.subtitle = "Solver V7 native / reachable future search"
        print(f"Native V7 engine enabled: {native_path()}")
    else:
        search_settings = SearchSettingsV6(
            depth=int(cfg.get("search_depth", 5)),
            beam_width=int(cfg.get("beam_width", 24)),
            time_budget_ms=float(cfg.get("search_budget_ms", 250)),
            allow_hold=bool(cfg.get("search_allow_hold", True)))
        solve_function = find_best_v6
        describe_function = describe_action_v6
        overlay.subtitle = "Solver V6 fallback / build native_v7 for V7"
        print(f"Native V7 engine not built; using V6 fallback. Expected: {native_path()}")
    solver_pool = ThreadPoolExecutor(max_workers=1,
                                     thread_name_prefix="tetris-solver")
    solver_future = None
    pending_key = None
    result_key = None
    result = None
    timer = QTimer()
    shutdown_timer = QTimer()
    stop_requested = threading.Event()

    def tick():
        nonlocal previous_active, previous_name, previous_queue
        nonlocal solver_future, pending_key, result_key, result
        try:
            screenshot = np.asarray(capture.grab(region))
            labels = read_cells(screenshot,
                                saturation_min=cfg.get("saturation_min", 72),
                                value_min=cfg.get("value_min", 70),
                                gray_value_min=cfg.get("gray_value_min", 108),
                                rows=20 + spawn_rows)
            if next_region is None and not next_rois and queue_roi is None:
                overlay.clear_targets()
                overlay.text = "NEXT not calibrated: run python calibrate_all.py"
                overlay.subtitle = "Confirm BOARD, individual NEXT pieces, and HOLD"
            else:
                if next_region is not None:
                    upcoming = read_queue_region(
                        np.asarray(capture.grab(next_region)),
                        saturation_min=cfg.get("queue_saturation_min", 65),
                        value_min=cfg.get("queue_value_min", 55))
                elif next_rois:
                    upcoming = read_queue_rois(
                        np.asarray(capture.grab(hud_box)), next_rois,
                        (hud_box["left"], hud_box["top"]),
                        saturation_min=cfg.get("queue_saturation_min", 70),
                        value_min=cfg.get("queue_value_min", 60))
                else:
                    upcoming = read_next_queue(
                        np.asarray(capture.grab(queue_roi)), slots=queue_slots,
                        saturation_min=cfg.get("queue_saturation_min", 95),
                        value_min=cfg.get("queue_value_min", 90),
                        min_pixels=cfg.get("queue_min_pixels", 12))
                hold_view = (read_hold_view(
                    np.asarray(capture.grab(hold_roi)),
                    previous=tracker.hold, proposed_current=tracker.current,
                    saturation_min=cfg.get("queue_saturation_min", 85),
                    value_min=cfg.get("queue_value_min", 75))
                    if hold_roi is not None else None)
                current = tracker.observe(upcoming, hold_view)
                queue_text = " ".join(tracker.queue or ()) or "unreadable"
                if upcoming is None:
                    overlay.clear_targets()
                    overlay.text = "NEXT preview unreadable; advice paused"
                    overlay.subtitle = f"Last stable NEXT: {queue_text}"
                    previous_active = None
                    previous_name = None
                elif current is None:
                    overlay.clear_targets()
                    overlay.text = tracker.status
                    overlay.subtitle = (f"NEXT: {queue_text} | "
                                        f"HOLD: {tracker.hold or 'empty'}")
                    previous_active = None
                    previous_name = None
                else:
                    if previous_name != current or previous_queue != tracker.queue:
                        previous_active = None
                    active = locate_expected_piece(labels, current, previous_active)
                    previous_queue = tracker.queue
                    previous_name = current
                    if active is None:
                        overlay.clear_targets()
                        overlay.text = f"CURRENT={current} | pose not confidently visible"
                        overlay.subtitle = "Tolerant 3-of-4 detection; check board crop/theme"
                        previous_active = None
                    else:
                        previous_active = active
                        active = shift_active(active, -spawn_rows)
                        visible_labels = labels[spawn_rows:].copy()
                        stack = occupied_without_active(visible_labels, active)
                        state_key = (stack.tobytes(), active.name,
                                     active.rotation, active.x,
                                     tuple(tracker.queue or ()), tracker.hold,
                                     tracker.can_hold, search_settings)
                        if solver_future is not None and solver_future.done():
                            completed = solver_future
                            completed_key = pending_key
                            solver_future = None
                            pending_key = None
                            result = completed.result()
                            result_key = completed_key
                        if result_key != state_key and solver_future is None:
                            pending_key = state_key
                            solver_future = solver_pool.submit(
                                solve_function, stack.copy(), active,
                                tuple(tracker.queue or ()), tracker.hold,
                                tracker.can_hold, search_settings)
                        best = result if result_key == state_key else None
                        if result_key != state_key:
                            overlay.clear_targets()
                            overlay.text = f"CURRENT={current} | planning lookahead..."
                            overlay.subtitle = (f"NEXT: {queue_text} | "
                                                "capture remains responsive")
                        elif best is None:
                            overlay.clear_targets()
                            overlay.text = f"CURRENT={current} | no reachable placement"
                            overlay.subtitle = tracker.status
                        elif any(x < 0 or x >= 10 or y < 0 or y >= 20
                                 for x, y in best.cells):
                            overlay.clear_targets()
                            overlay.text = "Solver returned an out-of-board placement"
                            overlay.subtitle = f"Suppressed cells: {best.cells}"
                        else:
                            overlay.target = best.cells
                            overlay.target_name = best.name
                            next_cells = best.next_cells
                            if (next_cells is not None and
                                    all(0 <= x < 10 and 0 <= y < 20
                                        for x, y in next_cells)):
                                overlay.next_target = next_cells
                                overlay.next_target_name = best.next_name
                            else:
                                overlay.next_target = None
                                overlay.next_target_name = None
                            third_cells = best.third_cells
                            if (third_cells is not None and
                                    all(0 <= x < 10 and 0 <= y < 20
                                        for x, y in third_cells)):
                                overlay.third_target = third_cells
                                overlay.third_target_name = best.third_name
                            else:
                                overlay.third_target = None
                                overlay.third_target_name = None
                            overlay.text = describe_function(active, best)
                            cooldown = ("ready" if tracker.can_hold else
                                        "used" if tracker.can_hold is False else "?")
                            plan_names = [name for name in
                                          (best.name, best.next_name,
                                           best.third_name) if name]
                            plan = (f" | PLAN: {' > '.join(plan_names)}"
                                    if len(plan_names) > 1 else "")
                            overlay.subtitle = (f"HOLD: {tracker.hold or 'empty'} "
                                                f"({cooldown}) | NEXT: {queue_text} | "
                                                f"D{best.depth_used} / "
                                                f"{best.elapsed_ms:.0f}ms | "
                                                f"score {best.score:.1f}"
                                                f"{plan}")
            overlay.update()
        except Exception as exc:
            overlay.clear_targets()
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
        solver_pool.shutdown(wait=False, cancel_futures=True)
        if console_handler is not None:
            try:
                ctypes.windll.kernel32.SetConsoleCtrlHandler(console_handler, False)
            except (AttributeError, OSError):
                pass
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
