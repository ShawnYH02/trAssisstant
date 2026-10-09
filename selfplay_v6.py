"""Offline same-bag V5/V6 simulation; reports survival and S1-inspired reward.

Not a TETR.IO rules engine. No versus garbage or frame-exact 180 kicks.
Use --pieces 100 --seeds 10 for a stronger (slower) comparison.
"""
from __future__ import annotations
import argparse
from collections import deque
import random
import statistics
import numpy as np
from solver_v5 import find_best_v5, SearchSettings, lock, to_rows, from_rows, s1_reward, _make_spawn
from solver_v6 import find_best_v6, SearchSettingsV6

NAMES='IOTSZJL'


def bag_sequence(seed: int, count: int):
    rng = random.Random(seed)
    bag=[]
    while len(bag) < count:
        seven=list(NAMES)
        rng.shuffle(seven)
        bag.extend(seven)
    return deque(bag)


def run_one(seed: int, engine: str, pieces: int, budget: float):
    pending=bag_sequence(seed, pieces+15)
    current=pending.popleft()
    hold=None
    board=np.zeros((20,10),bool)
    lines=0; quads=0; spins=0; attack_score=0.0; times=[]; depths=[]
    b2b=0; combo=-1
    for step in range(pieces):
        active = _make_spawn(current)
        if active is None:
            break
        if engine=='v5':
            ans=find_best_v5(board,active,tuple(list(pending)[:5]),hold,True,
                    SearchSettings(depth=5,beam_width=30,time_budget_ms=budget),
                    initial_b2b=b2b, initial_combo=combo)
        else:
            ans=find_best_v6(board,active,tuple(list(pending)[:5]),hold,True,
                    SearchSettingsV6(depth=5,beam_width=24,time_budget_ms=budget),
                    initial_b2b=b2b, initial_combo=combo)
        if ans is None:
            break
        played=current
        if ans.hold_used:
            if hold is None:
                hold=current
                played=pending.popleft()
            else:
                played,hold=hold,current
        assert ans.name == played, (step,current,ans.name,played)
        result=lock(to_rows(board),ans.cells)
        if result is None:
            break
        rows,clear=result
        board=from_rows(rows)
        lines+=clear
        quads+=(clear==4)
        spins+=(ans.spin!='none' and clear>0)
        attack,b2b,combo=s1_reward(clear,ans.spin,b2b,combo)
        attack_score+=attack
        times.append(ans.elapsed_ms)
        depths.append(ans.depth_used)
        if step + 1 < pieces:
            current=pending.popleft()
    return {'pieces':len(times),'lines':lines,'quads':quads,'spins':spins,
            'attack_proxy':round(attack_score,1),
            'median_ms':round(statistics.median(times),1) if times else 0,
            'mean_depth':round(statistics.mean(depths),2) if depths else 0}


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--pieces',type=int,default=40)
    cli.add_argument('--seeds',type=int,default=3)
    cli.add_argument('--budget',type=float,default=120)
    args=cli.parse_args()
    for seed in range(args.seeds):
        res5=run_one(seed+7300,'v5',args.pieces,args.budget)
        res6=run_one(seed+7300,'v6',args.pieces,args.budget)
        print(f'bag {seed+7300} | V5 {res5} | V6 {res6}',flush=True)

if __name__ == '__main__':
    main()
