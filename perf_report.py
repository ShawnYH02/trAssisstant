"""Report measured CC2 latency and speculative-prefetch effectiveness."""
import argparse
from collections import Counter
import json
from pathlib import Path


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return (ordered[lower] * (upper - position) +
            ordered[upper] * (position - lower))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='cc2_latency.jsonl')
    options = parser.parse_args()
    path = Path(options.file)
    if not path.is_file():
        parser.exit(1, f'File not found: {path}\n')
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
        except json.JSONDecodeError:
            continue
    print('CC2 actual-machine performance measurements')
    for kind, field in [('visible', 'latency_ms'), ('solver', 'elapsed_ms'),
                        ('solver', 'pathfinding_ms'), ('solver', 'engine_ms')]:
        values = [float(row[field]) for row in records
                  if row.get('event') == kind and
                  isinstance(row.get(field), (int, float))]
        if values:
            print(f'  {kind}.{field}: count={len(values)}, '
                  f'p50={percentile(values, 50):.1f} ms, '
                  f'p95={percentile(values, 95):.1f} ms, '
                  f'max={max(values):.1f} ms')
    decisions = [row for row in records if row.get('event') == 'solver']
    modes = Counter(row.get('mode') for row in decisions)
    if modes:
        print('  search modes:', dict(modes))
        hits = modes.get('prefetch_hit', 0)
        print(f'  observed prefetch handoffs: {hits}/{sum(modes.values())} '
              f'= {hits / sum(modes.values()):.1%}')
    reasons = Counter(row.get('reset_reason') for row in decisions
                      if row.get('mode') == 'start' and
                      row.get('reset_reason'))
    if reasons:
        print('  restart reasons:')
        for name, count in reasons.most_common():
            print(f'    {name}: {count}')
    if decisions:
        newest = decisions[-1]
        print('  cumulative speculative prefetch hits:',
              newest.get('prefetch_hits', 'not recorded'))
        print('  cumulative speculative misses:',
              newest.get('prefetch_misses', 'not recorded'))
        print('  cumulative prefetched path cache hits:',
              newest.get('route_cache_hits', 'not recorded'))
    first_events = [row for row in records
                    if row.get('event') == 'cc2_first_suggestion']
    if first_events:
        print('  first-available suggestion timing:')
        for mode in ('all', 'start', 'prefetch_hit',
                     'cached_same_piece', 'refresh', 'advance'):
            selected = [row for row in first_events
                        if mode == 'all' or row.get('mode') == mode]
            if not selected:
                continue
            values = [float(row['first_nonempty_ms']) for row in selected
                      if isinstance(row.get('first_nonempty_ms'), (int, float))]
            if values:
                print(f'    {mode}: count={len(selected)}, '
                      f'p50={percentile(values, 50):.1f} ms, '
                      f'p95={percentile(values, 95):.1f} ms, '
                      f'missing={len(selected) - len(values)}')
            else:
                print(f'    {mode}: count={len(selected)}, no valid replies')
        unanswered = sum(int(row.get('unanswered_polls', 0))
                         for row in first_events)
        empty = sum(int(row.get('empty_replies', 0))
                    for row in first_events)
        print(f'    polling: unanswered={unanswered}, empty={empty}')
    print('Visible timing starts after the scanner recognizes a stable state; '
          'initial and restarted positions can still be slow.')


if __name__ == '__main__':
    main()
