"""Offline sanity check: does the Rust engine return a reachable placement?"""
import numpy as np
import solver_v5 as v5
from solver_v7 import SearchSettingsV7, describe_action_v7, find_best_v7, native_path


def main():
    print('Native engine:', native_path())
    board=np.zeros((20,10),dtype=bool)
    active=v5._make_spawn('T')
    best=find_best_v7(board,active,next_queue=list('IOLJSZ'),hold=None,
                      can_hold=True,settings=SearchSettingsV7(depth=5,beam_width=24,time_budget_ms=250))
    assert best is not None
    assert v5.lock(v5.to_rows(board),best.cells) is not None
    print(describe_action_v7(active,best))
    print('target:',best.cells)
    print(f'lookahead={best.depth_used}, native_candidates={best.nodes_expanded}, elapsed={best.elapsed_ms:.0f}ms')
    print('First placement is collision-checked and legal in the local board model.')


if __name__=='__main__': main()
