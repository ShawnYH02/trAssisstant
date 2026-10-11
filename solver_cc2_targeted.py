"""Bounded route validation for Cold Clear's preferred placements.

Cold Clear remains responsible for strategy.  This module validates only the
ranked placements returned by the engine instead of enumerating every locking
placement before looking at the reply.  Every displayed placement still needs
a collision-checked route under the existing V5 movement model.
"""
from __future__ import annotations

from collections import deque
import os
import time
from typing import Optional

import numpy as np

import solver_cc2 as cc2
import solver_cc2_reliable as reliable
import solver_v5 as v5
from solver_cc2_resync import TransitionPending
from solver_v2 import Pose, active_pose, legal, pose_cells


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return min(high, max(low, float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _goal_poses(name: str, cells):
    """Return V5 pose representations whose minos exactly match ``cells``."""
    wanted = frozenset(cells)
    results = set()
    for rotation, shape in enumerate(v5.STATES[name]):
        dx = min(x for x, _ in shape)
        dy = min(y for _, y in shape)
        pose = Pose(rotation,
                    min(x for x, _ in cells) - dx,
                    min(y for _, y in cells) - dy)
        if frozenset(pose_cells(name, pose)) == wanted:
            results.add(pose)
    return results


def targeted_route(board: np.ndarray, active, target_cells,
                   expected_spin: str = "none", *, max_states: int = 5000,
                   budget_ms: float = 90.0):
    """Return ``(move, examined_states)`` for one exact locking placement.

    The breadth-first search uses the same collision and rotation transitions
    as the prior whole-board validator, but stops as soon as the requested
    landing is proven reachable.
    """
    if active is None or active.name not in cc2.PIECES:
        return None, 0
    target = tuple(sorted(map(tuple, target_cells)))
    if (len(set(target)) != 4 or
            any(not (0 <= x < 10 and 0 <= y < 20) for x, y in target)):
        return None, 0
    goals = _goal_poses(active.name, target)
    if not goals:
        return None, 0
    start = active_pose(active)
    if (not legal(board, active.name, start) or
            set(pose_cells(active.name, start)) != set(active.cells)):
        return None, 0

    rows = v5.to_rows(board)
    indexed = {}
    for goal in goals:
        indexed.setdefault((goal.r, goal.bx), []).append(goal)
    seen = {(start, False): None}
    queue = deque([(start, False)])
    deadline = time.perf_counter() + max(0.001, float(budget_ms) / 1000.0)
    searched = 0

    while (queue and len(seen) <= max_states and
           time.perf_counter() < deadline):
        pose, last_rotate = queue.popleft()
        searched += 1
        for goal in indexed.get((pose.r, pose.bx), ()):
            if pose.by > goal.by:
                continue
            if (expected_spin in {"mini", "full"} and
                    (pose != goal or not last_rotate)):
                continue
            if not all(legal(board, active.name,
                             Pose(pose.r, pose.bx, y))
                       for y in range(pose.by, goal.by + 1)):
                continue
            if legal(board, active.name,
                     Pose(goal.r, goal.bx, goal.by + 1)):
                continue
            spin = (v5.corner_tspin(
                rows, goal, last_rotation=(last_rotate and pose == goal))
                if active.name == "T" else "none")
            if expected_spin in {"mini", "full"} and spin != expected_spin:
                continue
            locked = v5.lock(rows, target)
            if locked is None:
                continue
            after, cleared = locked
            actions = []
            key = (pose, last_rotate)
            while seen[key] is not None:
                parent, label = seen[key]
                actions.append(label)
                key = parent
            actions.reverse()
            offset_x, offset_y = v5.OFFSETS[active.name][goal.r]
            return (v5._Move(
                after, active.name, goal.r, goal.bx + offset_x,
                goal.by + offset_y, target, cleared, spin,
                tuple(actions) + ("Hard drop",)), searched)

        for label, next_pose in v5.transitions_v5(board, active.name, pose):
            next_key = (next_pose, label in {"CW", "CCW", "180"})
            if next_key not in seen and len(seen) < max_states:
                seen[next_key] = ((pose, last_rotate), label)
                queue.append(next_key)
    return None, searched


def match_suggestions(board, active, message, queue_pieces, hold, can_hold,
                      settings, *, budget_ms=90.0):
    """Validate Cold Clear candidates in rank order within one total budget."""
    started = time.perf_counter()
    searched = 0
    for choice in (message.get("moves") or [])[:8]:
        remaining = budget_ms - 1000.0 * (time.perf_counter() - started)
        if remaining <= 0:
            break
        location = choice.get("location") or {}
        name = location.get("type")
        if name not in cc2.PIECES:
            continue
        hold_used = name != active.name
        held = hold or (queue_pieces[0] if queue_pieces else None)
        if (hold_used and
                (can_hold is not True or not settings.allow_hold or
                 name != held)):
            continue
        try:
            cells = cc2.tbp_cells(location)
        except (KeyError, TypeError, ValueError):
            continue
        origin = v5._make_spawn(name) if hold_used else active
        if origin is None:
            continue
        expected_spin = (choice.get("spin", "none")
                         if settings.strict_spin else "none")
        move, states = targeted_route(
            board, origin, cells, expected_spin,
            max_states=settings.max_current_states, budget_ms=remaining)
        searched += states
        if move is not None:
            return (choice, move, hold_used), searched
    return None, searched


def find_best_cc2_targeted(
        board: np.ndarray, active, next_queue=(), hold=None, can_hold=False,
        settings=None, *, initial_b2b=0, initial_combo=-1
) -> Optional[cc2.CC2Recommendation]:
    """Ask CC2 first, then validate only its preferred current placements."""
    settings = settings or cc2.SearchSettingsCC2()
    if active is None or active.name not in cc2.PIECES:
        return None
    overall = time.perf_counter()
    queue_pieces = tuple(p for p in (next_queue or ()) if p in cc2.PIECES)
    client = reliable.get_reliable_client(cc2.cc2_path(settings))

    # Do not turn a mixed lock-animation frame into a cached no-placement.
    # Raising here lets assistant_overlay retry this same state on its next tick.
    preflight = getattr(client, "preflight", None)
    if (preflight is not None and
            preflight(board, active.name, queue_pieces, hold, can_hold)):
        raise TransitionPending(
            "Waiting for coherent board / NEXT / HOLD after lock")

    engine_started = time.perf_counter()
    message = client.query(
        board, active.name, queue_pieces, hold,
        combo=max(0, initial_combo), b2b=bool(initial_b2b),
        budget_ms=settings.time_budget_ms, can_hold=can_hold)
    engine_ms = 1000.0 * (time.perf_counter() - engine_started)

    validation_started = time.perf_counter()
    validation_budget = _env_float(
        "TRASSIST_CC2_ROUTE_MS", 90.0, 10.0, 500.0)
    result, searched = match_suggestions(
        board, active, message, queue_pieces, hold, can_hold, settings,
        budget_ms=validation_budget)
    attempts = 1

    # Preserve the reliable adapter's recovery ceiling. This is only used
    # after the first reply has no executable candidate and returns early as
    # soon as a later candidate validates.
    retry_ms = _env_float("TRASSIST_CC2_RETRY_MS", 250.0, 0.0, 500.0)
    if result is None and retry_ms > 0:
        def valid(choice):
            nonlocal searched
            trial, extra = match_suggestions(
                board, active, {"moves": [choice]}, queue_pieces, hold,
                can_hold, settings, budget_ms=min(30.0, retry_ms))
            searched += extra
            return trial is not None

        newer = client.retry_for_valid(valid, budget_ms=retry_ms)
        attempts += 1
        if newer is not None:
            message = newer
            result, extra = match_suggestions(
                board, active, message, queue_pieces, hold, can_hold,
                settings, budget_ms=validation_budget)
            searched += extra

    validation_ms = 1000.0 * (time.perf_counter() - validation_started)
    elapsed_ms = 1000.0 * (time.perf_counter() - overall)
    try:
        from solver_cc2_fast import emit_metric
        emit_metric(dict(
            event="solver", pipeline="targeted", mode=client.mode,
            elapsed_ms=round(elapsed_ms, 3),
            engine_ms=round(engine_ms, 3),
            engine_first_nonempty_ms=round(engine_ms, 3),
            pathfinding_ms=round(validation_ms, 3),
            targeted_validation_ms=round(validation_ms, 3),
            states=searched, route_states=searched,
            engine_polls=getattr(client, "last_first_metrics", {}).get("polls"),
            validated=result is not None, reply_rounds=attempts,
            reset_reason=getattr(client, "last_restart_reason", None),
            prefetch_hits=getattr(client, "prefetch_hits", 0),
            prefetch_misses=getattr(client, "prefetch_misses", 0)))
    except (ImportError, AttributeError):
        pass

    if result is None:
        reliable.report_unmatched(active.name, message, states=searched,
                                  strict_spin=settings.strict_spin)
        client.forget()
        return None

    raw, move, hold_used = result
    client.remember(raw, move.cells, hold_used)
    actions = (("HOLD",) if hold_used else ()) + move.actions
    return cc2.CC2Recommendation(
        name=move.name, rotation=move.r, x=move.x, y=move.y,
        cells=move.cells, score=0.0, cleared=move.lines, actions=actions,
        visited_states=searched, spin=move.spin, depth_used=0,
        nodes_expanded=message.get("move_info", {}).get("nodes", 0),
        elapsed_ms=elapsed_ms, hold_used=hold_used)
