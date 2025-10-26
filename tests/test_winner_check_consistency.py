# -*- coding: utf-8 -*-
"""
Test: verify consistency of winner detection between 'scan_all' and 'last_move'.
- Simulates random games and checks outputs after each move.
- Stops each game when any win is detected (episode early stop), ensuring proper game semantics.
Run:
  python tests/test_winner_check_consistency.py [--width 15 --height 15 --n_in_row 5 --games 1000 --seed 123]
Exit status:
  0 if all consistent; 1 if mismatches found.
"""
import os
import sys
import random
import argparse
from typing import Dict, Any, List, Tuple

# Ensure repo root on path for local imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from game_core import Board


def gen_random_full_permutation(width: int, height: int, seed: int) -> List[int]:
    total = width * height
    rng = random.Random(seed)
    return rng.sample(range(total), total)


def check_game_once(cfg_base: Dict[str, Any], moves: List[int]) -> Tuple[int, List[Tuple[int, Tuple[bool, int], Tuple[bool, int], Tuple[bool, int]]]]:
    """
    Play one game by applying moves in order, stop when any win is detected.
    Returns:
      stop_step: index at which the episode stopped (1-based), or total cells if no win
      mismatches: list of (step, res_last, res_scan_es, res_scan_full) where outputs differ
    """
    # Detection board (always scan_all early stop True)
    cfg_detect = dict(cfg_base)
    cfg_detect["winner_check"] = "scan_all"
    cfg_detect["scan_all_early_stop"] = True
    b_detect = Board(config=cfg_detect)
    b_detect.init_board(start_player=0)

    # Boards under test
    cfg_scan_es = dict(cfg_base)
    cfg_scan_es["winner_check"] = "scan_all"
    cfg_scan_es["scan_all_early_stop"] = True
    b_scan_es = Board(config=cfg_scan_es)
    b_scan_es.init_board(start_player=0)

    cfg_scan_full = dict(cfg_base)
    cfg_scan_full["winner_check"] = "scan_all"
    cfg_scan_full["scan_all_early_stop"] = False
    b_scan_full = Board(config=cfg_scan_full)

    b_scan_full.init_board(start_player=0)

    cfg_last = dict(cfg_base)
    cfg_last["winner_check"] = "last_move"
    b_last = Board(config=cfg_last)
    b_last.init_board(start_player=0)

    mismatches: List[Tuple[int, Tuple[bool, int], Tuple[bool, int], Tuple[bool, int]]] = []
    stop_step = len(moves)

    for k, mv in enumerate(moves, start=1):
        b_detect.do_move(mv)
        b_scan_es.do_move(mv)
        b_scan_full.do_move(mv)
        b_last.do_move(mv)

        res_last = b_last.has_a_winner()
        res_scan_es = b_scan_es.has_a_winner()
        res_scan_full = b_scan_full.has_a_winner()

        # Compare outputs; both methods should agree in proper episode flow
        if not (res_last == res_scan_es == res_scan_full):
            mismatches.append((k, res_last, res_scan_es, res_scan_full))

        # Early stop based on detection board
        win_any, _ = b_detect.has_a_winner()
        if win_any:
            stop_step = k
            break

    return stop_step, mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=15)
    ap.add_argument("--n_in_row", type=int, default=5)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    width = args.width
    height = args.height
    n_in_row = args.n_in_row
    games = max(1, int(args.games))
    base_cfg = {"board_width": width, "board_height": height, "n_in_row": n_in_row}

    total_cells = width * height
    total_mismatches = 0
    examples: List[Dict[str, Any]] = []

    for g in range(games):
        moves = gen_random_full_permutation(width, height, seed=args.seed + g)
        stop_step, mismatches = check_game_once(base_cfg, moves)
        if mismatches:
            total_mismatches += len(mismatches)
            # Record a compact example (first mismatch only)
            step, res_last, res_scan_es, res_scan_full = mismatches[0]
            examples.append({
                "game": g + 1,
                "step": step,
                "stop_step": stop_step,
                "last_move": res_last,
                "scan_all_es": res_scan_es,
                "scan_all_full": res_scan_full,
            })
        # Optional: print simple progress every 10%
        if (g + 1) % max(1, games // 10) == 0:
            print(f"[progress] {g+1}/{games} games checked")

    if total_mismatches == 0:
        print(f"OK: All consistent across {games} games ({width}x{height}, n={n_in_row}).")
        sys.exit(0)
    else:
        print(f"FAIL: Found {total_mismatches} mismatches across {games} games.")
        # Show up to 5 examples
        for ex in examples[:5]:
            print(
                f" - game={ex['game']}, step={ex['step']}, stop_step={ex['stop_step']} | "
                f"last_move={ex['last_move']} | scan_all(es)={ex['scan_all_es']} | scan_all(full)={ex['scan_all_full']}"
            )
        # Non-zero exit to signal test failure
        sys.exit(1)


if __name__ == "__main__":
    main()