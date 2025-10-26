# -*- coding: utf-8 -*-
"""
Benchmark: winner check time vs number of stones, for scan_all and last_move methods.
Outputs:
- artifacts/benchmarks/winner_check_trend.csv
- artifacts/benchmarks/winner_check_trend.png (if matplotlib available)
Run: python tests/benchmark_winner_check_trend.py [--width 15 --height 15 --n_in_row 5 --min 5 --max 200 --step 5 --iters 200]
"""
import os
import sys
import time
import random
import csv
import argparse
import statistics
from typing import Dict, Any, List

# Ensure repo root on path for local imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from game_core import Board


def build_board_with_moves(config: Dict[str, Any], moves: List[int]) -> Board:
    b = Board(config=config)
    b.init_board(start_player=0)
    for mv in moves:
        b.do_move(mv)
    return b


def gen_random_moves(width: int, height: int, count: int, seed: int) -> List[int]:
    total = width * height
    rng = random.Random(seed)
    return rng.sample(range(total), min(count, total))


def time_fn(fn, iters: int) -> float:
    # Return total elapsed seconds across iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return time.perf_counter() - t0


def gen_random_full_permutation(width: int, height: int, seed: int) -> List[int]:
    total = width * height
    rng = random.Random(seed)
    return rng.sample(range(total), total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=15)
    ap.add_argument("--n_in_row", type=int, default=5)
    ap.add_argument("--min", dest="min_count", type=int, default=1)
    ap.add_argument("--max", dest="max_count", type=int, default=225)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sequences", type=int, default=32)
    # Episodes mode (按局早停): set --games>0 to enable
    ap.add_argument("--games", type=int, default=1000, help="Number of random games (episodes) to simulate")
    ap.add_argument("--no_episode_early_stop", action="store_true", help="Do not stop episode when any win is present")
    ap.add_argument("--scan_all_full_scan", action="store_true", help="Disable early stop inside scan_all (full scan)")
    ap.add_argument("--wandb_project", type=str, default="winner-check-bench")
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="offline")
    args = ap.parse_args()

    width = args.width
    height = args.height
    n_in_row = args.n_in_row
    min_count = args.min_count
    max_count = min(args.max_count, width * height)
    step = args.step
    iters = args.iters
    num_sequences = args.sequences
    num_games = max(0, int(args.games))
    episode_early_stop = (not args.no_episode_early_stop)
    scan_all_early_stop = (not args.scan_all_full_scan)

    # Prepare base configs
    cfg_base = {"board_width": width, "board_height": height, "n_in_row": n_in_row}
    cfg_scan = dict(cfg_base)
    cfg_scan["winner_check"] = "scan_all"
    cfg_scan["scan_all_early_stop"] = scan_all_early_stop
    cfg_last = dict(cfg_base)
    cfg_last["winner_check"] = "last_move"

    # Initialize wandb
    run = None
    if WANDB_AVAILABLE:
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or (
                f"episodes_{width}x{height}_n{n_in_row}_games{num_games}" if num_games > 0
                else f"trend_{width}x{height}_n{n_in_row}_seq{num_sequences}"
            ),
            config={
                "width": width,
                "height": height,
                "n_in_row": n_in_row,
                "min_count": min_count,
                "max_count": max_count,
                "step": step,
                "iters": iters,
                "sequences": num_sequences,
                "games": num_games,
                "episode_early_stop": episode_early_stop,
                "scan_all_full_scan": args.scan_all_full_scan,
            },
        )

    # If games > 0, run episodes mode with early-stop per episode
    if num_games > 0:
        total_cells = width * height
        # Per-step samples and valid counts
        scan_samples: List[List[float]] = [[] for _ in range(total_cells + 1)]
        last_samples: List[List[float]] = [[] for _ in range(total_cells + 1)]
        valid_counts: List[int] = [0 for _ in range(total_cells + 1)]
        # Progress tracking
        t_start = time.perf_counter()
        progress_every = max(1, num_games // 10)

        for g in range(num_games):
            seq_moves = gen_random_full_permutation(width, height, seed=args.seed + g)
            # Boards: detection uses scan_all with early stop always True
            cfg_detect = dict(cfg_base)
            cfg_detect["winner_check"] = "scan_all"
            cfg_detect["scan_all_early_stop"] = True
            b_detect = Board(config=cfg_detect)
            b_detect.init_board(start_player=0)
            # Benchmark boards
            b_scan = Board(config=cfg_scan)
            b_scan.init_board(start_player=0)
            b_last = Board(config=cfg_last)
            b_last.init_board(start_player=0)

            for k in range(1, total_cells + 1):
                mv = seq_moves[k - 1]
                b_detect.do_move(mv)
                b_scan.do_move(mv)
                b_last.do_move(mv)
                # Warm up once before timing
                _ = b_scan.has_a_winner()
                _ = b_last.has_a_winner()
                # Measure
                t_scan = time_fn(b_scan.has_a_winner, iters)
                t_last = time_fn(b_last.has_a_winner, iters)
                scan_ms = (t_scan / iters) * 1000.0
                last_ms = (t_last / iters) * 1000.0
                scan_samples[k].append(scan_ms)
                last_samples[k].append(last_ms)
                valid_counts[k] += 1
                # Episode early stop check
                win_any, _ = b_detect.has_a_winner()
                if episode_early_stop and win_any:
                    break

            # Progress print every ~10%
            done = g + 1
            if done % progress_every == 0 or done == num_games:
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / done) * (num_games - done)
                print(f"[progress] {done}/{num_games} games | elapsed={elapsed:.1f}s | eta={eta:.1f}s")

        # Aggregate results per step
        results = []  # (count, scan_mean, last_mean, scan_std, last_std, valid)
        for count in range(1, total_cells + 1):
            v = valid_counts[count]
            if v == 0:
                continue
            s_list = scan_samples[count]
            l_list = last_samples[count]
            scan_mean = statistics.mean(s_list)
            last_mean = statistics.mean(l_list)
            scan_std = statistics.pstdev(s_list) if len(s_list) > 1 else 0.0
            last_std = statistics.pstdev(l_list) if len(l_list) > 1 else 0.0
            speedup = (scan_mean / last_mean) if last_mean > 0 else float("inf")
            results.append((count, scan_mean, last_mean, scan_std, last_std, v))
            print(
                f"step={count:3d} | valid={v:6d} | scan_all(mean±std)={scan_mean:7.3f}±{scan_std:7.3f} ms | "
                f"last_move(mean±std)={last_mean:7.3f}±{last_std:7.3f} ms | speedup={speedup:6.1f}x"
            )
            if run:
                wandb.log(
                    {
                        "step": count,
                        "valid_count": v,
                        "scan_all_ms_mean": scan_mean,
                        "scan_all_ms_std": scan_std,
                        "last_move_ms_mean": last_mean,
                        "last_move_ms_std": last_std,
                        "speedup": speedup,
                    },
                    step=count,
                )

        # Ensure output dir
        out_dir = os.path.join(ROOT_DIR, "artifacts", "benchmarks")
        os.makedirs(out_dir, exist_ok=True)
        # Save CSV (episodes)
        csv_path = os.path.join(out_dir, "winner_check_episodes.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "step",
                "valid_count",
                "scan_all_ms_mean",
                "last_move_ms_mean",
                "scan_all_ms_std",
                "last_move_ms_std",
                "speedup_scan_over_last",
                "scan_all_full_scan",
                "episode_early_stop",
                "games",
            ])
            for count, scan_mean, last_mean, scan_std, last_std, v in results:
                speedup = (scan_mean / last_mean) if last_mean > 0 else float("inf")
                w.writerow([
                    count,
                    v,
                    f"{scan_mean:.6f}",
                    f"{last_mean:.6f}",
                    f"{scan_std:.6f}",
                    f"{last_std:.6f}",
                    f"{speedup:.3f}",
                    int(args.scan_all_full_scan),
                    int(episode_early_stop),
                    num_games,
                ])
        print(f"CSV saved: {csv_path}")

        # Plot PNG if matplotlib available
        try:
            import matplotlib.pyplot as plt
            counts = [r[0] for r in results]
            scan_vals = [r[1] for r in results]
            last_vals = [r[2] for r in results]
            valid_vals = [r[5] for r in results]
            plt.figure(figsize=(9, 5))
            ax1 = plt.gca()
            ax1.plot(counts, scan_vals, label="scan_all (mean)", color="C0")
            ax1.plot(counts, last_vals, label="last_move (mean)", color="C1")
            ax1.set_xlabel("Step index (stones played)")
            ax1.set_ylabel("Avg time per check (ms)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper left")
            ax2 = ax1.twinx()
            ax2.plot(counts, valid_vals, label="valid_count", color="C3", alpha=0.5)
            ax2.set_ylabel("Valid samples")
            ax2.legend(loc="upper right")
            plt.title(f"Episodes winner-check time ({width}x{height}, n={n_in_row}, games={num_games}, early_stop={episode_early_stop}, full_scan={args.scan_all_full_scan})")
            png_path = os.path.join(out_dir, "winner_check_episodes.png")
            plt.tight_layout()
            plt.savefig(png_path)
            print(f"PNG saved: {png_path}")
            if run:
                pass  # Not using wandb.save on Windows
        except Exception as e:
            print(f"matplotlib not available or plotting failed: {e}")

        if run:
            # Log table for full results
            table = wandb.Table(columns=[
                "step",
                "valid_count",
                "scan_all_ms_mean",
                "last_move_ms_mean",
                "scan_all_ms_std",
                "last_move_ms_std",
                "speedup",
            ])
            for count, scan_mean, last_mean, scan_std, last_std, v in results:
                speedup = (scan_mean / last_mean) if last_mean > 0 else float("inf")
                table.add_data(count, v, scan_mean, last_mean, scan_std, last_std, speedup)
            wandb.log({"winner_check_episodes": table})
            run.finish()
        return

    # Fallback: original trend mode (fixed counts across sequences)
    # Generate multiple full-board random sequences
    total_cells = width * height
    sequences = [gen_random_full_permutation(width, height, seed=args.seed + i) for i in range(num_sequences)]

    results = []  # (count, scan_mean_ms, last_mean_ms, scan_std_ms, last_std_ms)

    # Sweep counts
    for count in range(min_count, max_count + 1, step):
        scan_ms_list = []
        last_ms_list = []
        for seq_moves in sequences:
            moves = seq_moves[:count]
            b_scan = build_board_with_moves(cfg_scan, moves)
            b_last = build_board_with_moves(cfg_last, moves)
            # Warm-up
            _ = b_scan.has_a_winner()
            _ = b_last.has_a_winner()
            # Time
            t_scan = time_fn(b_scan.has_a_winner, iters)
            t_last = time_fn(b_last.has_a_winner, iters)
            scan_ms_list.append((t_scan / iters) * 1000.0)
            last_ms_list.append((t_last / iters) * 1000.0)

        scan_mean = statistics.mean(scan_ms_list)
        last_mean = statistics.mean(last_ms_list)
        scan_std = statistics.pstdev(scan_ms_list) if len(scan_ms_list) > 1 else 0.0
        last_std = statistics.pstdev(last_ms_list) if len(last_ms_list) > 1 else 0.0
        speedup = (scan_mean / last_mean) if last_mean > 0 else float("inf")

        results.append((count, scan_mean, last_mean, scan_std, last_std))
        print(
            f"count={count:3d} | scan_all(mean±std)={scan_mean:7.3f}±{scan_std:7.3f} ms | "
            f"last_move(mean±std)={last_mean:7.3f}±{last_std:7.3f} ms | speedup={speedup:6.1f}x"
        )

        if run:
            wandb.log(
                {
                    "count": count,
                    "scan_all_ms_mean": scan_mean,
                    "scan_all_ms_std": scan_std,
                    "last_move_ms_mean": last_mean,
                    "last_move_ms_std": last_std,
                    "speedup": speedup,
                },
                step=count,
            )

    # Ensure output dir
    out_dir = os.path.join(ROOT_DIR, "artifacts", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(out_dir, "winner_check_trend.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "count",
            "scan_all_ms_mean",
            "last_move_ms_mean",
            "scan_all_ms_std",
            "last_move_ms_std",
            "speedup_scan_over_last",
            "scan_all_full_scan",
        ])
        for count, scan_mean, last_mean, scan_std, last_std in results:
            speedup = (scan_mean / last_mean) if last_mean > 0 else float("inf")
            w.writerow([
                count,
                f"{scan_mean:.6f}",
                f"{last_mean:.6f}",
                f"{scan_std:.6f}",
                f"{last_std:.6f}",
                f"{speedup:.3f}",
                int(args.scan_all_full_scan),
            ])
    print(f"CSV saved: {csv_path}")

    # Plot PNG if matplotlib available
    try:
        import matplotlib.pyplot as plt
        counts = [r[0] for r in results]
        scan_vals = [r[1] for r in results]
        last_vals = [r[2] for r in results]
        plt.figure(figsize=(8, 5))
        plt.plot(counts, scan_vals, label="scan_all (mean)", color="C0")
        plt.plot(counts, last_vals, label="last_move (mean)", color="C1")
        plt.xlabel("Number of stones on board")
        plt.ylabel("Avg time per check (ms)")
        plt.title(f"Winner check time vs stones ({width}x{height}, n_in_row={n_in_row}, seq={num_sequences})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        png_path = os.path.join(out_dir, "winner_check_trend.png")
        plt.tight_layout()
        plt.savefig(png_path)
        print(f"PNG saved: {png_path}")
        if run:
            pass  # PNG path noted; not saving via symlink on Windows
    except Exception as e:
        print(f"matplotlib not available or plotting failed: {e}")

    if run:
        # Log table for full results
        table = wandb.Table(columns=[
            "count",
            "scan_all_ms_mean",
            "last_move_ms_mean",
            "scan_all_ms_std",
            "last_move_ms_std",
            "speedup",
        ])
        for count, scan_mean, last_mean, scan_std, last_std in results:
            speedup = (scan_mean / last_mean) if last_mean > 0 else float("inf")
            table.add_data(count, scan_mean, last_mean, scan_std, last_std, speedup)
        wandb.log({"winner_check_trend": table})
        # Avoid wandb.save on Windows due to symlink privileges; table already logged
        run.finish()


if __name__ == "__main__":
    try:
        import wandb
        WANDB_AVAILABLE = True
    except Exception:
        wandb = None
        WANDB_AVAILABLE = False
    main()