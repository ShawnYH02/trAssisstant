# TETR.IO Board-Analysis Overlay (Windows learning prototype)

> **Fair-play reminder:** TETR.IO rules prohibit third-party bots/assistance tools
> for an advantage, explicitly including solution finders. Do not use this during
> live competitive matches or to gain an advantage on the service.
> Rules: https://tetr.io/about/rules/

A non-controlling experiment that captures a visible Tetris board, detects a
falling tetromino using default colors, searches collision-checked placements
with queue/HOLD lookahead, and paints a click-through landing outline plus a
finesse-style route. Runs locally.

## Install

Open PowerShell in this folder, preferably with Python 3.11 or 3.12 installed:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks venv activation, use `.venv\Scripts\python.exe` instead
of `python` below (or run the commands in Command Prompt).

## Calibrate (recommended)

Open the game in a window with its board, colored NEXT previews, and HOLD area
visible, then run the one-screen calibrator:

```powershell
python calibrate_all.py --monitor 1
```

Switch focus back to TETR.IO during the three-second capture delay. Use
`--monitor 2` when the game is on the second display. The preview uses:

- Green for the 10×20 board.
- Orange for each individual NEXT piece.
- Pink for HOLD.

Press Enter only when all rectangles are correct. Press R to make just three
selections—BOARD, the whole NEXT preview column, and HOLD—on the same captured
frame. During play, the overlay re-detects and reads the pieces inside the saved
NEXT region on every frame; calibration-time internal boxes are diagnostic only.
An inferred empty HOLD rectangle needs especially careful review. Existing
unrelated config values are preserved and a local backup is made.

Verify recognition before starting the overlay:

```powershell
python inspect_v4.py
```

Switch back to TETR.IO during its three-second delay as well. It should print
readable NEXT and HOLD results. Diagnostic crops stay local and are excluded
from Git.

The older separate calibrators remain available if needed:

```powershell
python calibrate.py --monitor 1
python calibrate_queue.py
python inspect_queue.py
```

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

The V6 solver runs on a background worker so capture and Ctrl+C remain
responsive. Its default target is depth 5 with a 250 ms budget; the HUD reports
the depth actually completed and elapsed search time. Tune `search_depth`,
`beam_width`, and `search_budget_ms` in `config.json` if needed. For an offline
comparison against V5, run `python benchmark_v6.py`; `selfplay_v6.py` provides
a longer same-bag offline simulation.

## Troubleshooting recognition

Run `python inspect_capture.py` to save `debug_board.png` and print the detected
10-column grid, including the four capture rows above the board. If the cells do
not match the game, correct the crop in `config.json` or experiment with the
three thresholds. Do not share captured screenshots that contain private
information.

For NEXT/HOLD problems, rerun `python inspect_v4.py`. Recalibrate after moving
or resizing the game window, and use the default colored preview skin.

## Tests

```powershell
python -m pytest -q
```

## Scope and limitations

- Searches from the currently detected position and rotation. Taps,
  hold-to-obstacle moves, soft drops, 90° rotations, and hard drops are checked
  for collisions before a route is suggested.
- Plans up to five pieces using the recognized NEXT queue, beam pruning, and an
  S1-inspired heuristic for Quads, T-spins, B2B chains, and combos. HOLD is
  considered only when the tracker reports it available.
- V6 preserves diverse first-move candidates, caps continuations per parent,
  rewards perfect clears, and evaluates nonlinear height danger, buried holes,
  transitions, roughness, and accessible wells.
- Uses standard JLSTZ SRS kicks and a symmetric I-piece kick approximation.
  TETR.IO-specific I kicks and 180° kicks are not modeled exactly.
- Ranks placements using line clears, height, holes, covered holes, surface
  bumpiness, wells, and row transitions. It is not provably optimal.
- Reads the 10×20 playfield plus four capture rows above it so a complete
  tetromino can be recognized before it enters the visible board.
- Reads each colored NEXT preview from its own rectangle. HOLD identity persists
  when its icon dims, while hold availability is tracked separately. Queue
  tracking intentionally waits for a confirmed shift instead of guessing the
  initial current piece.
- Does not yet recognize gray ghost cells or garbage reliably across all themes,
  special skins, spins, or 180° rotations.
- Suggested routes are collision-checked in a turn-based model, but gravity,
  frame timing, auto-shift charge, and lock delay are not simulated. A legal
  route may still require faster execution than the current game state allows.
- Only CURRENT receives a collision-checked action path. Future placements are
  forecasting candidates, not guaranteed input routes, and the scoring is an
  approximation rather than an exact attack simulator.
- Offline benchmark/self-play metrics are heuristic comparisons, not evidence
  of multiplayer strength or parity with mature Tetris engines.
- Fits the queue-identified active shape against the board and can tolerate one
  missing or misclassified cell when recent pose evidence resolves ambiguity.
  It suspends suggestions when position or rotation is not sufficiently clear.
- HUD animation, flashing lines, bloom, scaling, custom shaders, DPI and
  multi-monitor geometry can affect reliability. Test carefully.
- A Windows display-affinity flag attempts to hide the overlay from MSS
  captures. As backup the overlay draws only cell outlines, avoiding cell-center
  sampling regions.

For serious analysis, next steps are: per-game color calibration and temporal
piece tracking; exact SRS+ and 180° kick tables; exact attack/B2B/combo state
tracking; confidence indicators; saved-frame regression tests.
