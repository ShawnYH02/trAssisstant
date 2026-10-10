"""Deterministic offline search-latency/depth comparison, NOT win-rate testing."""
import argparse
from pathlib import Path
import statistics
import subprocess

H=20
FULL=(1<<10)-1

def boards():
    # Test positions are synthetic but deterministic; all root lines/spins = 0.
    empty=[0]*H
    well=[0]*16+[FULL^1]*4
    staircase=[0]*15+[1<<6, (1<<5)|(1<<6), (1<<4)|(1<<5)|(1<<6), (1<<4)|(1<<5)|(1<<6), (1<<4)|(1<<5)|(1<<6)]
    cave=[0]*14+[(1<<2)|(1<<6)]*3 + [(1<<2)|(1<<3)|(1<<4)|(1<<5)|(1<<6)]*3
    mixed=[0]*13+[0b0011001100,0b0011100110,0b0111100110,0b0111100111,0b0111100111,0b0111101111,0b0111101111]
    for name,b in [('empty',empty),('side_well',well),('staircase',staircase),('cave',cave),('mixed',mixed)]:
        assert len(b)==H
        yield name,b

def request(board,queue='TSZILOT',budget=250,width=24):
    rows=','.join(f'{v:03x}' for v in board)
    return f'V7 {queue} 5 {width} {budget} 0 -1 1 1 0 {rows} 0 n - 0\n'

def run(path, msg):
    p=subprocess.run([str(path)],input=msg,text=True,capture_output=True,timeout=15)
    if p.returncode:
        raise RuntimeError(f'{path}: {p.stderr[:300]}')
    result=p.stdout.strip().split()
    if not result or result[0]!='OK' or len(result)<6:
        raise RuntimeError(f'{path}: {p.stdout[:300]}')
    return dict(root=int(result[1]),depth=int(result[3]),nodes=int(result[4]),ms=float(result[5]))

def main():
    ap=argparse.ArgumentParser(description='V7 / V9 native engine comparison')
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--budget-ms',type=int,default=250)
    ap.add_argument('--beam',type=int,default=24)
    args=ap.parse_args()
    for p in (args.baseline,args.candidate):
        if not p.is_file():ap.error(f'Executable not found: {p}')
    old=[];new=[]
    print('case        | baseline depth / ms / nodes | V9 depth / ms / nodes')
    for name,b in boards():
        msg=request(b,budget=args.budget_ms,width=args.beam)
        a=run(args.baseline,msg);v=run(args.candidate,msg)
        old.append(a);new.append(v)
        print(f'{name:11} | {a["depth"]:2} / {a["ms"]:7.1f} / {a["nodes"]:5} | {v["depth"]:2} / {v["ms"]:7.1f} / {v["nodes"]:5}')
    print(f'Average depth: {statistics.mean(x["depth"] for x in old):.2f} -> {statistics.mean(x["depth"] for x in new):.2f}')
    print(f'Median ms: {statistics.median(x["ms"] for x in old):.1f} -> {statistics.median(x["ms"] for x in new):.1f}')
    print('Search depth/latency does NOT measure playing strength. Different evaluation scores are not directly comparable.')
if __name__=='__main__': main()
