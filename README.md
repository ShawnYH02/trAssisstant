# TETR.IO Board-Analysis Overlay (Windows learning prototype)

> **Fair-play reminder:** TETR.IO rules prohibit third-party bots/assistance tools
> for an advantage, explicitly including solution finders. Do not use this during
> live competitive matches or to gain an advantage on the service.
> Rules: https://tetr.io/about/rules/

A non-controlling experiment that captures a visible Tetris board, detects a
falling tetromino using default colors, searches collision-checked placements,
and paints a click-through landing outline plus a finesse-style route. Runs
locally.

## Install

Open PowerShell in this folder, preferably with Python 3.11 or 3.12 installed:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks venv activation, use `.venv\Scripts\python.exe` instead
of `python` below (or run the commands in Command Prompt).

## Calibrate

1. Open the game in a window with a default color theme, minimal effects, and
   a clean 10-column × 20-row visible playfield.
2. Pause on a frame where the playfield borders are visible, then run:

```powershell
python calibrate.py
```

3. Drag a loose crop around the board with a little padding. The calibrator
   detects and snaps to a 10×20 playfield, then shows a green confirmation box.
   Press Enter to accept or R to draw the rough area again. On multiple
   monitors, use `python calibrate.py --monitor 2` as necessary.
4. Check `config.json`, which uses *physical desktop pixels*.

## Calibrate NEXT and optional HOLD

The queue tracker needs a separate crop around the preview icons:

```powershell
python calibrate_queue.py
```

Draw a tight rectangle around all vertically stacked NEXT piece icons, excluding
the NEXT label and other colored UI. Enter the number of visible preview slots
(normally five). You may then select the HOLD piece, or press Escape to skip it.

Verify the preview reader before starting the overlay:

```powershell
python inspect_queue.py
```

It should print all NEXT identities instead of `UNREADABLE`. The diagnostic
saves `debug_queue.png` locally; the file is excluded from Git.

## Run

```powershell
python assistant_overlay.py
```

The overlay is click-through and does not send keyboard inputs. Stop with
`Ctrl+C` in the launching terminal. For the initial test, use a static or
slow-moving board and inspect output before trying motion.

The preview contains future pieces, not the current piece. On startup the
tracker first stabilizes the NEXT queue, then needs one normal placement and a
confirmed queue shift before it can identify CURRENT. If the queue changes too
quickly or unexpectedly, advice is suspended until tracking resynchronizes.

## Troubleshooting recognition

Run `python inspect_capture.py` to save `debug_board.png` and print the detected
10-column grid, including the four capture rows above the board. If the cells do
not match the game, correct the crop in `config.json` or experiment with the
three thresholds. Do not share captured screenshots that contain private
information.

For NEXT/HOLD problems, rerun `python inspect_queue.py`. Recalibrate after
moving or resizing the game window, and use the default colored preview skin.

## Tests

```powershell
python -m pytest -q
```

## Scope and limitations

- Searches from the currently detected position and rotation. Taps,
  hold-to-obstacle moves, soft drops, 90° rotations, and hard drops are checked
  for collisions before a route is suggested.
- Uses standard JLSTZ SRS kicks and a symmetric I-piece kick approximation.
  TETR.IO-specific I kicks and 180° kicks are not modeled exactly.
- Ranks placements using line clears, height, holes, covered holes, surface
  bumpiness, wells, and row transitions. It is not provably optimal.
- Reads the 10×20 playfield plus four capture rows above it so a complete
  tetromino can be recognized before it enters the visible board.
- Reads the colored NEXT queue and can optionally track HOLD. A greyed-out HOLD
  icon may be unreadable, and queue tracking intentionally waits for a confirmed
  shift instead of guessing the initial current piece.
- Does not yet recognize gray ghost cells or garbage reliably across all themes,
  special skins, spins, or 180° rotations.
- Suggested routes are collision-checked in a turn-based model, but gravity,
  frame timing, auto-shift charge, and lock delay are not simulated. A legal
  route may still require faster execution than the current game state allows.
- Detects an active piece as the uppermost **isolated four-cell colored group**.
  If a piece touches a locked group of the same color, lies partly offscreen,
  or the skin is unusual, it intentionally suspends suggestions or may err.
- HUD animation, flashing lines, bloom, scaling, custom shaders, DPI and
  multi-monitor geometry can affect reliability. Test carefully.
- A Windows display-affinity flag attempts to hide the overlay from MSS
  captures. As backup the overlay draws only cell outlines, avoiding cell-center
  sampling regions.

For serious analysis, next steps are: per-game color calibration and temporal
piece tracking; exact SRS+ and 180° kick tables; lookahead search using the
recognized queue; confidence indicators; saved-frame regression tests.
