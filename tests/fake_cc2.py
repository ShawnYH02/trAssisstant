"""Test-only imitation of Cold Clear 2's TBP handshake."""
import json
import sys

print(json.dumps({'type': 'info', 'name': 'fake', 'features': []}), flush=True)
suggestions = 0
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    kind = message.get('type')
    if kind == 'rules':
        print(json.dumps({'type': 'ready'}), flush=True)
    elif kind == 'start':
        pieces = message.get('queue', [])
        assert len(message['board']) == 40
        assert all(len(row) == 10 for row in message['board'])
        current = pieces[0] if pieces else 'O'
    elif kind == 'suggest':
        suggestions += 1
        if current == 'O':
            move = {'location': {'type': 'O', 'orientation': 'north',
                                 'x': 0, 'y': 0}, 'spin': 'none'}
        else:
            move = {'location': {'type': current, 'orientation': 'north',
                                 'x': 4, 'y': 0}, 'spin': 'none'}
        print(json.dumps({'type': 'suggestion', 'moves': [move],
                          'move_info': {'nodes': suggestions * 111}}), flush=True)
    elif kind == 'quit':
        break
