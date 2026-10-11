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

The solver runs on a background worker so capture and Ctrl+C remain responsive.
CC2 returns the first nonempty suggestion within the recovery ceiling, then
Python performs an early-exit route search for CC2's preferred landing. It does
not enumerate every possible lock first. The HUD reports total decision time.

A lightweight completion timer checks the background Future every 12 ms. It
requests the normal capture-and-validation path only when a result has just
finished, reducing the wait for the next 50 ms capture tick without increasing
idle screen captures. Repeated unchanged overlay states are not repainted.

Accepted plans remain visible while the falling piece moves. The route text is
marked stale when it was calculated from an earlier pose, but the landing guide
is retained. After a planned lock or HOLD, the overlay validates the board,
queue, and HOLD transition before promoting the next ghost immediately while a
fresh background search replenishes the plan. Unexpected transitions discard
the cached preview and trigger normal replanning.

## Cold Clear 2 backend (required)

The adapter uses a persistent Cold Clear 2 process for move selection
while retaining this project's vision, HOLD/queue tracking, click-through
overlay, and collision-checked current-piece routes. CC2 suggestions are shown
only when they match a locally reachable placement. This build fails closed:
if CC2 is missing, times out, or raises an error, it clears the ghost and shows
`COLD CLEAR UNAVAILABLE` instead of silently changing evaluators.

After a suggestion matches a legal current-piece route, CC2 speculatively
advances that move and searches the likely next position while the player is
still moving. When the lock occurs, the observed board, NEXT queue, HOLD
contents, and cooldown must all match the prediction before that prefetched
tree is used. Any mismatch safely restarts from the observed state. Python
route validation waits for a concrete CC2 target instead of speculatively
enumerating every possible lock for the predicted next piece.

Cold Clear 2 is third-party MIT/Apache-2.0 software and is intentionally not
vendored into this repository. Build it beside `assistant_overlay.py`:

```powershell
git clone https://github.com/MinusKelvin/cold-clear-2.git cold_clear_2
cargo build --release --locked --manifest-path cold_clear_2\Cargo.toml
python smoke_cc2.py
```

Enable it in the machine-local `config.json`:

```json
{
  "solver_engine": "coldclear2",
  "cc2_budget_ms": 350,
  "cc2_current_states": 5000,
  "cc2_strict_spin": true,
  "capture_interval_ms": 50,
  "completion_interval_ms": 12,
  "pose_grace_ms": 90
}
```

An absolute `coldclear2_exe` path can be supplied when the executable is
elsewhere. CC2's TBP response contains alternative immediate moves, not a
sequential plan, so the overlay intentionally shows one verified ghost.

The adapter records local timing samples in `cc2_latency.jsonl`. After playing
a few pieces, run `python perf_report.py` to see p50/p95 engine, targeted-route,
total latency, prefetch hit rate, and restart reasons. Set
`TRASSIST_CC2_METRICS=off` to disable this log. Native CC2 tree prefetch remains
enabled.

Completed-result delivery timings are written separately to
`cc2_delivery.jsonl`. `python perf_report.py` includes them automatically when
the file exists. Set `TRASSIST_DELIVERY_LOG=off` to disable this log, or use the
environment variable to select another filename. Delivery timing ends when all
overlay fields are assigned; it does not measure the Windows compositor or the
monitor.

Up to 250 rejected prefetch transitions are recorded locally in
`cc2_transitions.jsonl`, using only 10-bit board occupancy rows and piece/queue
names—never screenshots or input history. After a practice session, run:

```powershell
python transition_report.py
python perf_report.py
```

The transition report distinguishes a geometrically compatible alternative
placement from a board change that no visible four-cell lock explains. It does
not prove movement reachability, so do not loosen synchronization checks based
only on this report. Set `TRASSIST_CC2_TRACE=off` to disable tracing.

CC2 suggestion requests use first-available polling: empty or unanswered
replies are retried until the configured search deadline, and the first
nonempty move list is still checked against a reachable local route before it
is displayed. A repeated observation of the same piece reuses its validated
reply without polling again. The metrics log records first reply time, first
nonempty time, poll count, and unanswered polls; `perf_report.py` summarizes
them alongside visible latency.

Transient lock frames receive a bounded 130 ms coherence window before an
unexpected board/NEXT/HOLD transition resets CC2. If the complete transition
appears during that window, the verified prefetched tree is retained. Advice
is briefly suppressed while a transition is pending instead of showing a
fallback based on mixed frames.

Measure CC2 readiness independently from capture and route enumeration with:

```powershell
python benchmark_cc2.py --trials 8 --budget-ms 800
```

The standalone first-readiness benchmark accepts the experimental
`TRASSIST_CC2_FIRST_POLL_MS` (default `8`) and
`TRASSIST_CC2_FIRST_INTERVAL_MS` (default `5`) controls. The live recovery
client deliberately uses the conservative cadence described below. The
transition window is controlled by `TRASSIST_CC2_SETTLE_MS` (default `130`).

### No-placement recovery

The runtime uses a conservative 25 ms request cadence and permits up to 700 ms
for a late initial CC2 answer, while still returning immediately when a result
is available. If the earliest result cannot be matched to a reachable local
route, CC2 receives another 250 ms to improve its candidates. No unvalidated
placement is ever displayed; failure clears the ghost and remains fail-closed.

Recovery events are written locally to `cc2_recovery.jsonl`. To distinguish an
engine timeout from rejected candidate geometry, run:

```powershell
python recovery_report.py
```

`TRASSIST_CC2_RECOVERY_BUDGET_MS` controls the initial availability ceiling
(default `700`). Set `TRASSIST_CC2_RECOVERY_LOG=off` to disable this diagnostic
log. Reduce the ceiling only after the report shows that valid replies reliably
arrive sooner.

Targeted route validation uses a 90 ms upper bound, controlled by
`TRASSIST_CC2_ROUTE_MS`. If the first CC2 reply has no executable candidate,
`TRASSIST_CC2_RETRY_MS` controls the later-candidate recovery window (default
`250`). These are ceilings rather than mandatory waits.

## Troubleshooting recognition

Run `python diagnose_runtime.py` first to verify that the active configuration
uses Cold Clear 2, coherent frame capture, the pose guard, and fail-closed
solver behavior. This check does not start the overlay or modify your files.

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
- The CC2 backend assumes guideline TBP semantics and cannot express
  the live HOLD cooldown directly. Every displayed CC2 placement is therefore
  filtered against the observed cooldown and the local reachable-route search.
- CC2 returns the first available candidate within the recovery ceiling. It
  searches a validated suggested move early, but uses that prefetched tree only
  when the observed lock, queue shift, HOLD state, and cooldown prove the
  displayed suggestion was followed; otherwise it resets.
- Uses standard JLSTZ SRS kicks and a symmetric I-piece kick approximation.
  TETR.IO-specific I kicks and 180° kicks are not modeled exactly.
- Reads the 10×20 playfield plus four capture rows above it so a complete
  tetromino can be recognized before it enters the visible board.
- Board, NEXT, and HOLD are normally cropped from one coherent MSS frame. If
  their union exceeds three million pixels, capture safely falls back to the
  individual rectangles.
- Reads each colored NEXT preview from its own rectangle. HOLD identity persists
  when its icon dims, while hold availability is tracked separately. Queue
  tracking intentionally waits for a confirmed shift instead of guessing the
  initial current piece.
- Does not yet recognize gray ghost cells or garbage reliably across all themes,
  special skins, spins, or 180° rotations.
- Suggested routes are collision-checked in a turn-based model, but gravity,
  frame timing, auto-shift charge, and lock delay are not simulated. A legal
  route may still require faster execution than the current game state allows.
- Moving after a route is calculated keeps the landing guide stable but can make
  the original finesse directions stale or the landing unreachable. The HUD
  warns when directions came from an earlier pose.
- A verified ghost can remain for at most `pose_grace_ms` during a small pose
  dropout when CURRENT, NEXT, HOLD, and nearly all cells remain unchanged.
  Directions are paused; larger changes immediately clear it.
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
