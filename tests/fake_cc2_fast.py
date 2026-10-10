"""Deterministic fake TBP executable with a persistent-command audit trail."""
import json
import sys

log = sys.argv[1]


def record(message):
    with open(log, 'a', encoding='utf-8') as output:
        output.write(json.dumps(message, sort_keys=True) + '\n')


print(json.dumps({'type': 'info', 'name': 'fake', 'version': 'test',
                  'features': []}), flush=True)
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    kind = message.get('type')
    record(message)
    if kind == 'rules':
        print(json.dumps({'type': 'ready'}), flush=True)
    elif kind == 'suggest':
        move = {'location': {'type': 'O', 'orientation': 'north',
                             'x': 0, 'y': 0}, 'spin': 'none'}
        print(json.dumps({'type': 'suggestion', 'moves': [move],
                          'move_info': {'nodes': 11}}), flush=True)
    elif kind == 'quit':
        break
