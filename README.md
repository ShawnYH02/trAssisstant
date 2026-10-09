# TETR.IO Board-Analysis Overlay (Windows learning prototype)

> **Fair-play reminder:** TETR.IO rules prohibit third-party bots/assistance tools
> for an advantage, explicitly including solution finders. Do not use this during
> live competitive matches or to gain an advantage on the service.
> Rules: https://tetr.io/about/rules/

A non-controlling experiment that captures a visible Tetris board, detects a
falling tetromino using default colors, ranks hard-drop landings, and paints a
click-through outline plus a minimum-input finesse-style route. Runs locally.

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

## Run

```powershell
python assistant_overlay.py
```

The overlay is click-through and does not send keyboard inputs. Stop with
`Ctrl+C` in the launching terminal. For the initial test, use a static or
slow-moving board and inspect output before trying motion.

## Troubleshooting recognition

Run `python inspect_capture.py` to save `debug_board.png` and print a 20x10
ASCII representation of detected cells. If the cells do not match the actual
board, correct the crop in `config.json` or experiment with the three thresholds.
Do not share captured screenshots that contain private information.

## Tests

```powershell
python -m pytest -q
```

## Scope and limitations

- Samples one piece at a time and ranks simple **straight hard-drop** placements
  by line clears, heights, holes and bumpiness. It is *not* provably optimal.
- Reads the 10×20 playfield plus four capture rows above it so a complete
  tetromino can be recognized before it enters the visible board.
- Does not yet recognize HOLD/NEXT, gray ghost cells, garbage well enough for all
  themes, special skins, spins, 180/SRS+ kicks or actual input-path reachability.
- Finesse suggestions minimize taps, rotations and DAS-to-wall inputs in this
  prototype's bounding-box model. They remain **approximate**, because SRS+
  rotation centers, kicks, collision paths and frame-perfect DAS are not modeled.
- Detects an active piece as the uppermost **isolated four-cell colored group**.
  If a piece touches a locked group of the same color, lies partly offscreen,
  or the skin is unusual, it intentionally suspends suggestions or may err.
- HUD animation, flashing lines, bloom, scaling, custom shaders, DPI and
  multi-monitor geometry can affect reliability. Test carefully.
- A Windows display-affinity flag attempts to hide the overlay from MSS
  captures. As backup the overlay draws only cell outlines, avoiding cell-center
  sampling regions.

For serious analysis, next steps are: per-game color calibration and temporal
piece tracking; next queue / hold OCR; exact SRS+ kick/action search; beam search
with next 3-5 pieces; confidence indicators; dedicated offline test harness.
