"""Summarize Cold Clear availability and route-recovery diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(path):
    counts = Counter()
    response_times = []
    examples = []
    if not path.exists():
        print('No recovery log found:', path)
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get('event') != 'cc2_recovery':
            continue
        status = entry.get('status', 'unknown')
        counts[status] += 1
        if status == 'engine_reply':
            response_times.append(float(entry.get('elapsed_ms', 0)))
        if (status in {'unmatched_root', 'no_legal_route', 'engine_timeout'}
                and len(examples) < 4):
            examples.append(entry)
    print('CC2 recovery diagnostic')
    if not counts:
        print('No recovery events recorded.')
        return
    for status, quantity in counts.most_common():
        print(f'  {status}: {quantity}')
    if response_times:
        response_times.sort()
        print('  engine-response median:',
              f'{response_times[len(response_times) // 2]:.1f} ms')
    if examples:
        print('  Example errors:')
        for entry in examples:
            print('    ', json.dumps(entry, ensure_ascii=False)[:230])
    print('A timeout means no engine move arrived by the deadline; an '
          'unmatched root means route validation rejected the engine move.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='cc2_recovery.jsonl')
    options = parser.parse_args()
    summarize(Path(options.log))
