"""Verify the integrated speculative bridge with your locally built CC2.

Does not require screen capture, and does not modify your game.
"""
import argparse
from pathlib import Path

import numpy as np
from solver_cc2 import SearchSettingsCC2, cc2_path
from solver_cc2_fast import applied_lock, find_best_cc2_fast
from solver_cc2_first import close_first_clients, get_first_client
from solver_v5 import _make_spawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', help='Path to cold-clear-2 executable')
    args = parser.parse_args()
    binary = Path(args.exe).resolve() if args.exe else cc2_path()
    settings = SearchSettingsCC2(executable=str(binary), time_budget_ms=1200)
    board = np.zeros((20, 10), dtype=bool)
    queue = ('I', 'O', 'L', 'S', 'J')
    try:
        first = find_best_cc2_fast(
            board, _make_spawn('T'), queue, None, True, settings)
        if first is None:
            raise SystemExit('CC2 returned no locally reachable recommendation')
        print('Engine:', binary)
        print('First verified recommendation:', first)

        # Feed the exact predicted lock back as a synthetic observation. This
        # proves that the bridge sends play/new_piece and reuses the CC2 tree.
        next_board = applied_lock(board, first.cells)
        if first.hold_used:
            current, next_queue, hold = 'O', ('L', 'S', 'J', 'Z', 'T'), 'T'
        else:
            current, next_queue, hold = 'I', ('O', 'L', 'S', 'J', 'Z'), None
        second = find_best_cc2_fast(
            next_board, _make_spawn(current), next_queue, hold, True, settings)
        client = get_first_client(binary)
        if second is None or client.mode != 'prefetch_hit':
            raise SystemExit(
                f'CC2 tree-advance verification failed: '
                f'mode={client.mode}, result={second}')
        print('Tree reuse mode:', client.mode)
        print('Second verified recommendation:', second)
    finally:
        close_first_clients()


if __name__ == '__main__':
    main()
