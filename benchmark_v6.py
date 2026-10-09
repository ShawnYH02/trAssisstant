"""Fixed-state, deterministic CPU/beam benchmark; no game screen involved.

Measures time and completed depth, NOT competitive playing strength.
"""
import argparse
import statistics
import numpy as np
from tetris_core import ActivePiece
from solver_v2 import Pose, pose_cells
from solver_v5 import SearchSettings, find_best_v5
from solver_v6 import SearchSettingsV6, find_best_v6, board_value
from solver_v5 import future_locks, eval_board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=12)
    parser.add_argument('--budget',type=float,default=250)
    parser.add_argument('--depth',type=int,default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    queue = ('O','T','I','J','L','S')
    boards=[]
    for i in range(args.samples):
        b=np.zeros((20,10),bool)
        for x in range(10):
            height=int(rng.integers(0,6))
            if height:
                b[-height:,x] = True
        boards.append(b)
    pose=Pose(0,3,-2)
    cells=pose_cells('T',pose)
    active=ActivePiece('T',0,min(x for x,y in cells),min(y for x,y in cells),cells)
    for label, func, settings in (
        ('V5',find_best_v5,SearchSettings(depth=args.depth,beam_width=30,
                                        time_budget_ms=args.budget)),
        ('V6',find_best_v6,SearchSettingsV6(depth=args.depth,beam_width=24,
                                         time_budget_ms=args.budget)),
    ):
        elapsed=[]; depths=[]
        for board in boards:
            # Fair cold-cache comparison; do not reuse future results from V5 in V6.
            future_locks.cache_clear()
            eval_board.cache_clear()
            board_value.cache_clear()
            answer=func(board,active,queue,settings=settings)
            if answer:
                elapsed.append(answer.elapsed_ms)
                depths.append(answer.depth_used)
        if elapsed:
            print(f'{label}: n={len(elapsed)} median_ms={statistics.median(elapsed):.1f} '
                  f'depth{args.depth}={depths.count(args.depth)}/{len(depths)} '
                  f'mean_depth={statistics.mean(depths):.2f}')

if __name__ == '__main__':
    main()
