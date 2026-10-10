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
Its default target is depth 5 with a 250 ms budget; the HUD reports
the depth actually completed and elapsed search time. Tune `search_depth`,
`beam_width`, and `search_budget_ms` in `config.json` if needed.

Accepted plans remain visible while the falling piece moves. The route text is
marked stale when it was calculated from an earlier pose, but the landing guide
is retained. After a planned lock or HOLD, the overlay validates the board,
queue, and HOLD transition before promoting the next ghost immediately while a
fresh background search replenishes the plan. Unexpected transitions discard
the cached preview and trigger normal replanning.

## Optional Cold Clear 2 backend (speculative prefetch)

The adapter uses a persistent Cold Clear 2 process for move selection
while retaining this project's vision, HOLD/queue tracking, click-through
overlay, and collision-checked current-piece routes. CC2 suggestions are shown
only when they match a locally reachable placement. If CC2 times out or returns
only an unavailable HOLD/unverified route, the overlay safely falls back to the
native V11 solver (or V6 when V11 is not built).

After a suggestion matches a legal current-piece route, CC2 speculatively
advances that move and searches the likely next position while the player is
still moving. When the lock occurs, the observed board, NEXT queue, HOLD
contents, and cooldown must all match the prediction before that prefetched
tree is used. Any mismatch safely restarts from the observed state. Optional
Python legal-path precomputation for the predicted next piece is available,
but disabled by default because local measurements showed no route-cache hits.

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
  "capture_interval_ms": 50
}
```

Use `"solver_engine": "legacy"` to select V11/V6 directly. An absolute
`coldclear2_exe` path can be supplied when the executable is elsewhere. CC2's
TBP response contains alternative immediate moves, not a sequential plan, so
the CC2 path intentionally shows one ghost; the three-ghost display remains
available whenever the V11 fallback supplies the recommendation.

The adapter records local timing samples in `cc2_latency.jsonl`. After playing
a few pieces, run `python perf_report.py` to see p50/p95 latency, prefetch hit
rate, and restart reasons. Set `TRASSIST_CC2_METRICS=off` to disable this log.
Set `TRASSIST_PREFETCH_ROUTES=1` to experiment with speculative Python route
generation. Native CC2 tree prefetch remains enabled either way.

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

Optional polling controls are `TRASSIST_CC2_FIRST_POLL_MS` (default `8`) and
`TRASSIST_CC2_FIRST_INTERVAL_MS` (default `5`). The transition window is
controlled by `TRASSIST_CC2_SETTLE_MS` (default `130`).

## Optional native V11 solver

V11 uses a dependency-free Rust executable for reachable future-piece search.
It retains queue-aware structural hints and adds bounded tactical proof for
reachable full T-spin Doubles when T is immediate, held, or one known piece
away. Only movement-verified proofs receive the tactical priority bonus.
If Cargo is installed, build and verify it with:

```powershell
cargo build --release --manifest-path native_v7\Cargo.toml
cargo test --manifest-path native_v7\Cargo.toml
python smoke_v7.py
```

On the next launch, the overlay detects
`native_v7\target\release\trassist-v7.exe` and enables V11 automatically. If it
is absent, the overlay prints a notice and safely continues with the tested V6
solver. The executable name remains V7-compatible. Use `benchmark_native.py` for
side-by-side latency/depth measurement against a saved native baseline.

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
- Native V11 extends future-piece search to collision-checked BFS with 90°
  SRS-style kicks. Python still supplies the exact reachable CURRENT routes and
  HOLD roots, while Rust selects among them using deeper continuations.
- Its T-slot, Kaidan-like, STMB-like, and STSD-like scores are structural
  search hints, not guarantees that a named setup or spin is executable. Actual
  attack credit still requires the native movement search to produce a spin.
- V11 temporarily prioritizes branches only after its bounded native movement
  search proves a reachable full T-spin Double. Negative probes can still miss
  180-kick, gravity-timed, or deeper tactical continuations.
- The optional CC2 backend assumes guideline TBP semantics and cannot express
  the live HOLD cooldown directly. Every displayed CC2 placement is therefore
  filtered against the observed cooldown and the local reachable-route search.
- CC2 uses the full configured budget after a fresh/reset state. It starts
  searching the validated suggested move early, but uses that prefetched tree
  only when the observed lock, queue shift, HOLD state, and cooldown prove the
  displayed suggestion was followed; otherwise it resets.
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
- Moving after a route is calculated keeps the landing guide stable but can make
  the original finesse directions stale or the landing unreachable. The HUD
  warns when directions came from an earlier pose.
- Only CURRENT receives a collision-checked action path. Future placements are
  forecasting candidates, not guaranteed input routes, and the scoring is an
  approximation rather than an exact attack simulator.
- Offline benchmark/self-play metrics are heuristic comparisons, not evidence
  of multiplayer strength or parity with mature Tetris engines.
- V11 requires a locally built Rust executable and starts one native process per
  decision. It intentionally falls back to V6 in the overlay when unbuilt;
  native-specific benchmarks fail instead of silently substituting another engine.
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
