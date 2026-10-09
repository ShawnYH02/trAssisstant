"""Offline deterministic 7-bag proxy self-play: V6 vs V7. NOT a win-rate test."""
import argparse
import random
import statistics
import time
import numpy as np
import solver_v5 as v5
import solver_v6 as v6
import solver_v7 as v7


def bags(seed, count):
    rng=random.Random(seed)
    result=[]
    while len(result)<count:
        bag=list('IOTSZJL'); rng.shuffle(bag); result.extend(bag)
    return result


def play(engine, seed, count, ms):
    queue=bags(seed,count+10)
    board=np.zeros((20,10),dtype=bool)
    idx=0; held=None; b2b=0; combo=-1
    attacks=0; quads=0; spins=0; cleared_total=0; elapsed=[]; depth=[]
    for turn in range(count):
        piece=queue[idx]
        active=v5._make_spawn(piece)
        if active is None: break
        kwargs=dict(next_queue=queue[idx+1:idx+8], hold=held,
                    can_hold=True, initial_b2b=b2b, initial_combo=combo)
        if engine=='V6':
            best=v6.find_best_v6(board,active,settings=v6.SearchSettingsV6(
                depth=5,beam_width=24,time_budget_ms=ms),**kwargs)
        else:
            best=v7.find_best_v7(board,active,settings=v7.SearchSettingsV7(
                depth=5,beam_width=36,time_budget_ms=ms),**kwargs)
        if best is None: break
        elapsed.append(best.elapsed_ms); depth.append(best.depth_used)
        after=v5.lock(v5.to_rows(board), best.cells)
        if after is None: raise RuntimeError('Root placement invalid: '+str(best.cells))
        rows,lines=after
        assert lines==best.cleared
        score,b2b,combo=v5.s1_reward(lines,best.spin,b2b,combo)
        attacks+=score
        quads+=(lines==4); spins+=(best.spin!='none' and lines>0); cleared_total+=lines
        board=v5.from_rows(rows)
        if best.hold_used:
            if held is None:
                held=piece
                idx+=2
            else:
                held=piece
                idx+=1
        else: idx+=1
    return (turn+1 if best is not None else turn, round(attacks,1), quads, spins,
            cleared_total,round(statistics.median(elapsed),1) if elapsed else 0,
            round(statistics.mean(depth),2) if depth else 0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--pieces',type=int,default=50)
    ap.add_argument('--budget-ms',type=int,default=200)
    args=ap.parse_args()
    if not v7.native_path().is_file():
        raise SystemExit('Build the Rust executable first: cd native_v7 && cargo build --release')
    print('Synthetic 7-bag offline replay; attack is an S1-inspired proxy, not true match attack.')
    print('Solver | seed | pieces | attack_proxy | quads | spin_clears | cleared | median_ms | avg_depth')
    for seed in range(args.seeds):
        for name in ('V6','V7'):
            r=play(name,seed,args.pieces,args.budget_ms)
            print(name,seed,*r,sep=' | ')


if __name__=='__main__': main()
