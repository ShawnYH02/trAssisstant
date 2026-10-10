"""TBP fake returning no reply, an empty reply, then a recommendation."""
import json
import sys
import time

ready_after = float(sys.argv[1])
log = sys.argv[2]
started = time.monotonic()
print(json.dumps({'type': 'info', 'name': 'fixture', 'version': '1',
                  'features': []}), flush=True)
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    kind = message.get('type')
    with open(log, 'a', encoding='utf-8') as output:
        output.write(json.dumps(message) + '\n')
    if kind == 'rules':
        print('{"type":"ready"}', flush=True)
    elif kind in ('start', 'play'):
        started = time.monotonic()
    elif kind == 'suggest':
        elapsed = time.monotonic() - started
        if elapsed < ready_after / 2:
            continue
        if elapsed < ready_after:
            print('{"type":"suggestion","moves":[],"move_info":{"nodes":1}}',
                  flush=True)
        else:
            print(json.dumps({
                'type': 'suggestion',
                'moves': [{'location': {'type': 'O',
                                        'orientation': 'north',
                                        'x': 0, 'y': 0},
                           'spin': 'none'}],
                'move_info': {'nodes': 5},
            }), flush=True)
    elif kind == 'quit':
        break
