"""Offline reproducible benchmark. No game window needed.

Run from project folder: python benchmark_v5.py
"""
import statistics
import numpy as np
from tetris_core import ActivePiece
from solver_v2 import Pose, pose_cells
from solver_v5 import SearchSettings, find_best_v5


def main():
    rng = np.random.default_rng(44)
    pose = Pose(0, 3, -2)
    cells = pose_cells('T', pose)
    active = ActivePiece('T', 0, 3, -2, cells)
    for depth in (3, 5):
        elapsed, depths = [], []
        for _ in range(20):
            board = np.zeros((20, 10), bool)
            for x in range(10):
                height = int(rng.integers(0, 6))
                if height:
                    board[-height:, x] = True
            result = find_best_v5(
                board, active, ('I', 'O', 'J', 'L', 'S'),
                settings=SearchSettings(depth=depth, beam_width=30,
                                        time_budget_ms=190),
            )
            if result:
                elapsed.append(result.elapsed_ms)
                depths.append(result.depth_used)
        if elapsed:
            print(f'depth target={depth}: {len(elapsed)} boards, '
                  f'median={statistics.median(elapsed):.1f}ms, '
                  f'target reached={depths.count(depth)}/{len(depths)}')


if __name__ == '__main__':
    main()
