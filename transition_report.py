"""Summarize CC2 reset causes from recorded occupancy snapshots."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from solver_cc2 import CC2_CELLS
from solver_cc2_fast import applied_lock
from transition_debug import unmask


def candidate_placements(before, after, kinds):
    """Yield shape-compatible locks; this does not prove route reachability."""
    for name in kinds:
        if name not in CC2_CELLS:
            continue
        offsets = CC2_CELLS[name]
        for rotation in range(4):
            for y in range(20):
                for x in range(10):
                    cells = tuple((x + dx, 19 - (y + dy))
                                  for dx, dy in offsets)
                    if all(0 <= cx < 10 and 0 <= cy < 20
                           for cx, cy in cells):
                        future = applied_lock(before, cells)
                        if future is not None and np.array_equal(future, after):
                            yield name, cells, rotation
            offsets = tuple((dy, -dx) for dx, dy in offsets)


def classify(record):
    old, new = record['old'], record['new']
    if old['board'] is None or new['board'] is None:
        return 'missing_board_snapshot', None
    before, after = unmask(old['board']), unmask(new['board'])
    predicted = (unmask(record['predicted'])
                 if record.get('predicted') is not None else None)
    if predicted is not None and np.array_equal(after, predicted):
        return 'board_actually_matches', None
    if np.array_equal(before, after):
        return 'board_unchanged_or_premature_transition', None

    selected = record.get('selected')
    kinds = []
    if selected and selected.get('type'):
        kinds.append(selected['type'])
    for name in (old.get('current'), old.get('hold')):
        if name and name not in kinds:
            kinds.append(name)
    for name in old.get('queue', [])[:2]:
        if name not in kinds:
            kinds.append(name)
    candidates = list(candidate_placements(
        before, after, kinds or list(CC2_CELLS)))
    if not candidates:
        return ('no_four_cell_lock_explains_scan',
                int(np.count_nonzero(before != after)))
    if selected and any(name == selected.get('type')
                        for name, *_ in candidates):
        return 'different_location_or_pose_for_selected_piece', len(candidates)
    return 'matches_different_piece_identity_or_hold', len(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', nargs='?',
                        default='cc2_transitions.jsonl')
    options = parser.parse_args()
    path = Path(options.logfile)
    if not path.is_file():
        parser.exit(1, f'No trace found: {path}\n')
    reasons, groups, examples = Counter(), Counter(), {}
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
            kind, detail = classify(record)
            reasons[record.get('reason', 'unknown')] += 1
            groups[kind] += 1
            examples.setdefault(kind, (count + 1, detail))
            count += 1
        except (ValueError, TypeError, KeyError, IndexError):
            groups['unreadable_record'] += 1
    print(f'Transition trace: {count} mismatches analyzed')
    print('Original reasons:')
    for key, quantity in reasons.most_common():
        print(f'  {key}: {quantity}')
    print('Board mismatch classification:')
    for key, quantity in groups.most_common():
        first, detail = examples.get(key, (None, None))
        print(f'  {key}: {quantity} '
              f'(first entry {first}, detail={detail})')
    print('NOTE: different_location means geometry-compatible, not a proven '
          'reachable move.')
    print('Do not increase prefetch tolerance based on this report alone.')


if __name__ == '__main__':
    main()
