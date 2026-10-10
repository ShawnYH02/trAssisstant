"""Controlled slow or initially invalid TBP engine for recovery tests."""
import json
import sys
import time

scenario = sys.argv[1]
started = time.monotonic()
print(json.dumps({'type': 'info', 'name': 'recovery-fake',
                  'version': '1', 'features': []}), flush=True)
for raw in sys.stdin:
    try:
        message = json.loads(raw)
    except ValueError:
        continue
    kind = message.get('type')
    if kind == 'rules':
        print('{"type":"ready"}', flush=True)
    elif kind in ('start', 'play'):
        started = time.monotonic()
    elif kind == 'suggest':
        age = time.monotonic() - started
        if scenario == 'slow' and age < .36:
            continue
        if scenario == 'late_legal' and age < .12:
            piece = 'I'
            x = 4
        else:
            piece = 'O'
            x = 0
        result = {
            'type': 'suggestion',
            'moves': [{'location': {'type': piece, 'orientation': 'north',
                                    'x': x, 'y': 0}, 'spin': 'none'}],
            'move_info': {'nodes': 5},
        }
        print(json.dumps(result), flush=True)
    elif kind == 'quit':
        break
