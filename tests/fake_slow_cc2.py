"""Mock TBP engine that needs ~140ms of thinking after start/play."""
import json
import sys
import time

out = sys.argv[1]
thinking_start = None


def emit(value):
    print(json.dumps(value),flush=True)


def log(value):
    with open(out,'a',encoding='utf-8') as h:
        h.write(json.dumps(value)+'\n')


emit({'type':'info','name':'slow-fake','features':[]})
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    log(m)
    t = m.get('type')
    if t == 'rules':
        emit({'type':'ready'})
    elif t in ('start','play'):
        thinking_start = time.monotonic()
    elif t == 'suggest':
        if thinking_start is not None and time.monotonic()-thinking_start >= .14:
            emit({'type':'suggestion','moves':[{'location':{'type':'O','orientation':'north','x':0,'y':0},'spin':'none'}]})
    elif t == 'quit':
        break
