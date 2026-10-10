"""Deterministic native search latency/depth comparison, not strength testing."""
import argparse
from pathlib import Path
import statistics
import subprocess

H = 20
FULL = (1 << 10) - 1


def boards():
    empty = [0] * H
    well = [0] * 16 + [FULL ^ 1] * 4
    staircase = ([0] * 15 + [1 << 6, (1 << 5) | (1 << 6)] +
                 [(1 << 4) | (1 << 5) | (1 << 6)] * 3)
    cave = ([0] * 14 + [(1 << 2) | (1 << 6)] * 3 +
            [(1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6)] * 3)
    mixed = ([0] * 13 + [0b0011001100, 0b0011100110, 0b0111100110,
                          0b0111100111, 0b0111100111, 0b0111101111,
                          0b0111101111])
    for name, board in [('empty', empty), ('side_well', well),
                        ('staircase', staircase), ('cave', cave),
                        ('mixed', mixed)]:
        assert len(board) == H
        yield name, board


def request(board, queue='TSZILOT', budget=250, width=24):
    rows = ','.join(f'{value:03x}' for value in board)
    return f'V7 {queue} 5 {width} {budget} 0 -1 1 1 0 {rows} 0 n - 0\n'


def run(path, message):
    process = subprocess.run([str(path)], input=message, text=True,
                             capture_output=True, timeout=15)
    if process.returncode:
        raise RuntimeError(f'{path}: {process.stderr[:300]}')
    fields = process.stdout.strip().split()
    if not fields or fields[0] != 'OK' or len(fields) < 6:
        raise RuntimeError(f'{path}: {process.stdout[:300]}')
    return dict(root=int(fields[1]), depth=int(fields[3]),
                nodes=int(fields[4]), ms=float(fields[5]))


def main():
    parser = argparse.ArgumentParser(description='Compare two native engines')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--budget-ms', type=int, default=250)
    parser.add_argument('--beam', type=int, default=24)
    args = parser.parse_args()
    for path in (args.baseline, args.candidate):
        if not path.is_file():
            parser.error(f'Executable not found: {path}')
    baseline = []
    candidate = []
    print('case        | baseline depth / ms / nodes | candidate depth / ms / nodes')
    for name, board in boards():
        message = request(board, budget=args.budget_ms, width=args.beam)
        old = run(args.baseline, message)
        new = run(args.candidate, message)
        baseline.append(old)
        candidate.append(new)
        print(f'{name:11} | {old["depth"]:2} / {old["ms"]:7.1f} / '
              f'{old["nodes"]:5} | {new["depth"]:2} / {new["ms"]:7.1f} / '
              f'{new["nodes"]:5}')
    print(f'Average depth: {statistics.mean(x["depth"] for x in baseline):.2f} '
          f'-> {statistics.mean(x["depth"] for x in candidate):.2f}')
    print(f'Median ms: {statistics.median(x["ms"] for x in baseline):.1f} '
          f'-> {statistics.median(x["ms"] for x in candidate):.1f}')
    print('Latency/depth does not measure playing strength; evaluation scores '
          'from different engines are not directly comparable.')


if __name__ == '__main__':
    main()
