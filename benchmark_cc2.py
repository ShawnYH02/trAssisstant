"""Measure time to the first nonempty CC2 result without screen capture."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import numpy as np

from solver_cc2 import SearchSettingsCC2, cc2_path
from solver_cc2_first import FirstTBPProcess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', help='Full path to cold-clear-2.exe')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--budget-ms', type=float, default=800.0)
    options = parser.parse_args()
    executable = (Path(options.engine).resolve() if options.engine
                  else cc2_path(SearchSettingsCC2()))
    client = FirstTBPProcess(executable)
    board = np.zeros((20, 10), dtype=bool)
    samples = []
    try:
        print('Cold Clear executable:', executable)
        client.open()
        for index in range(max(1, options.trials)):
            current, queue = (
                ('T', ('I', 'O', 'L', 'J', 'S', 'Z'))
                if index % 2 == 0 else
                ('I', ('O', 'T', 'Z', 'S', 'L', 'J')))
            try:
                client.query(board, current, queue, None, can_hold=True,
                             budget_ms=options.budget_ms)
                metrics = client.last_first_metrics
                elapsed = metrics['first_nonempty_ms']
                samples.append(elapsed)
                print(f'trial {index + 1}: first valid={elapsed:.1f} ms, '
                      f'first reply={metrics["first_reply_ms"]}, '
                      f'polls={metrics["polls"]}, mode={metrics["mode"]}')
            except TimeoutError as exc:
                print(f'trial {index + 1}: timed out ({exc})')
        if samples:
            print('Median first nonempty:',
                  round(statistics.median(samples), 1), 'ms')
    finally:
        client.close()


if __name__ == '__main__':
    main()
