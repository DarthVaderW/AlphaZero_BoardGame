# -*- coding: utf-8 -*-
"""
Unified training runner: self-play + training in one loop, with periodic pure MCTS evaluation.
Aligned with original train.py behavior, configuration-driven via YAML.
"""
from __future__ import annotations
import os
import sys
import argparse
import random
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict

import numpy as np
import torch

# Ensure repo root is on sys.path when running as a script
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Local imports
from configs.registry import load_config, load_task
from models.cnn import PolicyValueNet
from agents.mcts_agent import MCTSAgent
from agents.pure_mcts_agent import PureMCTSAgent
from runners.replay_buffer import ReplayBuffer, augment_play_data
from env.gobang_env import GobangEnv
from game_core import Board, Game
from mcts.alpha_zero import MCTSPlayer as MCTS_AlphaZero
from mcts.pure import MCTSPlayer as MCTS_Pure


def setup_wandb(cfg: Dict[str, Any]):
    wb_cfg = (cfg.get("logging", {}) or {}).get("wandb", {})
    mode = wb_cfg.get("mode", "offline")
    project = wb_cfg.get("project", "gobang-alpha-zero")
    run_name = wb_cfg.get("run_name", "run")
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", mode)
        wandb.init(project=project, name=run_name, config=cfg, mode=mode)
        return wandb
    except Exception as e:
        print(f"[wandb] disabled ({e}).")
        return None


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainRunner:
    def __init__(self, cfg: Dict[str, Any], init_model: Optional[str] = None):
        self.cfg = cfg
        game_cfg = cfg.get("game", {})
        self.board_width = int(game_cfg.get("board_width", 15))
        self.board_height = int(game_cfg.get("board_height", 15))
        self.n_in_row = int(game_cfg.get("n_in_row", 4))

        # Board + Game: pass raw YAML game config dict directly
        self.board = Board(config=game_cfg)
        self.game = Game(self.board)
        # Env(s): RL-style interface
        env_cfg = cfg.get("env", {}) or {}
        self.num_envs = int(env_cfg.get("num_envs", 1))
        self.envs: List[GobangEnv] = [GobangEnv(self.board_width, self.board_height, self.n_in_row) for _ in range(self.num_envs)]

        train_cfg = cfg.get("training", {})
        self.learn_rate = float(train_cfg.get("lr", 2e-3))
        self.lr_multiplier = 1.0
        mcts_cfg = cfg.get("mcts", {})
        self.temp = float(mcts_cfg.get("temperature", 1.0))
        self.n_playout = int(mcts_cfg.get("n_playout", 400))
        self.c_puct = float(mcts_cfg.get("c_puct", 5.0))
        self.buffer_size = int(train_cfg.get("buffer_size", 10000))
        self.batch_size = int(train_cfg.get("batch_size", 512))
        self.play_batch_size = int(cfg.get("experiment", {}).get("play_batch_size", 1))
        self.epochs = int(train_cfg.get("epochs", 5))
        self.kl_targ = float(train_cfg.get("kl_targ", 0.02))
        self.check_freq = int(cfg.get("logging", {}).get("save_interval", 50))
        self.game_batch_num = int(cfg.get("experiment", {}).get("game_batch_num", 1500))
        self.best_win_ratio = 0.0
        self.use_gpu = bool(cfg.get("experiment", {}).get("use_gpu", torch.cuda.is_available()))

        pure_cfg = cfg.get("pure_mcts", {})
        self.pure_mcts_playout_num = int(pure_cfg.get("n_playout", 1000))
        self.pure_mcts_c_puct = float(pure_cfg.get("c_puct", 5.0))
        self.eval_n_games = int(pure_cfg.get("n_games", 10))
        # Pure MCTS upgrade schedule from YAML
        sched = pure_cfg.get("schedule", {}) or {}
        self.pure_sched_enabled = bool(sched.get("enabled", False))
        self.pure_sched_step = int(sched.get("step_n_playout", 1000))
        self.pure_sched_max = int(sched.get("max_n_playout", 5000))
        self.pure_sched_threshold = float(sched.get("win_ratio_threshold", 1.0))
        self.pure_sched_reset_best = bool(sched.get("reset_best_win_ratio", True))
        start_playout = sched.get("start_n_playout", None)
        if start_playout is not None:
            self.pure_mcts_playout_num = int(start_playout)

        # Model
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model, use_gpu=self.use_gpu)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, use_gpu=self.use_gpu)

        # Agents
        self.mcts_agent = MCTSAgent(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)
        self.data_buffer = ReplayBuffer(maxlen=self.buffer_size)
        self.episode_len = 0

        # Logging
        self.wandb = setup_wandb(cfg)
        self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", cfg.get("paths", {}).get("checkpoints_dir", "artifacts/checkpoints"))
        os.makedirs(os.path.abspath(self.checkpoints_dir), exist_ok=True)

    def _log(self, metrics: Dict[str, Any]):
        if self.wandb:
            try:
                self.wandb.log(metrics)
                print(metrics)
            except Exception:
                pass

    def collect_selfplay_data(self, n_games: int = 1) -> None:
        """Env-first rollout collection, rsl_rl/Isaac Gym style step/reset.
        Replaces Game.start_self_play to generate AlphaZero training samples via env.
        """
        total_collected = 0
        for _ in range(n_games):
            # Single-env episode (vectorization can be added later)
            env = self.envs[0]
            env.reset()
            player = self.mcts_agent.to_player()
            states: List[np.ndarray] = []
            mcts_probs: List[np.ndarray] = []
            current_players: List[int] = []
            while True:
                # Observe state and select action with MCTS
                state = env.observe()
                board = env.board
                current_players.append(board.get_current_player())
                move, move_prob = player.get_action(board, temp=self.temp, return_prob=1)
                states.append(state)
                mcts_probs.append(move_prob)
                # Step env
                _, _, done, info = env.step(move)
                if done:
                    winner = info.get("winner", -1)
                    winners_z = np.zeros(len(current_players), dtype=np.float32)
                    if winner != -1:
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0
                    player.reset_player()
                    play_data = list(zip(states, mcts_probs, winners_z))
                    self.episode_len = len(play_data)
                    aug_data = augment_play_data(play_data, self.board_width, self.board_height)
                    self.data_buffer.extend(aug_data)
                    total_collected += 1
                    self._log({"episode_len": self.episode_len, "buffer_len": len(self.data_buffer)})
                    break

    def policy_update(self) -> Tuple[float, float, float, float]:
        mini_batch = self.data_buffer.sample(self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs_t, old_v_t = self.policy_value_net.policy_value_torch(state_batch)
        kl = 0.0
        loss = 0.0
        entropy = 0.0
        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
            new_probs_t, new_v_t = self.policy_value_net.policy_value_torch(state_batch)
            kl_t = torch.mean(torch.sum(old_probs_t * (torch.log(old_probs_t + 1e-10) - torch.log(new_probs_t + 1e-10)), dim=1))
            kl = float(kl_t.item())
            if kl > self.kl_targ * 4:
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        winners_t = torch.tensor(winner_batch, dtype=torch.float32, device=self.policy_value_net.device)
        old_v_flat = old_v_t.view(-1)
        new_v_flat = new_v_t.view(-1)
        explained_var_old_t = 1.0 - torch.var(winners_t - old_v_flat, unbiased=False) / torch.var(winners_t, unbiased=False)
        explained_var_new_t = 1.0 - torch.var(winners_t - new_v_flat, unbiased=False) / torch.var(winners_t, unbiased=False)
        explained_var_old = float(explained_var_old_t.item())
        explained_var_new = float(explained_var_new_t.item())

        self._log({
            "kl": kl,
            "lr_multiplier": self.lr_multiplier,
            "loss": loss,
            "entropy": entropy,
            "explained_var_old": explained_var_old,
            "explained_var_new": explained_var_new,
        })
        return loss, entropy, explained_var_old, explained_var_new

    def policy_evaluate(self, n_games: Optional[int] = None) -> float:
        n_games = int(n_games or self.eval_n_games)
        current_mcts_player = MCTS_AlphaZero(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=self.pure_mcts_c_puct, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        self._log({"win_ratio": win_ratio, "pure_mcts_playout_num": self.pure_mcts_playout_num})
        return win_ratio

    def _save_checkpoint(self, step: int, is_best: bool = False) -> None:
        # Save model parameters
        run_name = (self.cfg.get("logging", {}) or {}).get("wandb", {}).get("run_name", "run")
        out_dir = os.path.abspath(os.path.join(self.checkpoints_dir, f"run_{run_name}", f"step_{step}"))
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "model.pth")
        torch.save(self.policy_value_net.policy_value_net.state_dict(), model_path)
        # Config snapshot
        import yaml, json
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, allow_unicode=True, sort_keys=False)
        state = {
            "lr_multiplier": self.lr_multiplier,
            "best_win_ratio": self.best_win_ratio,
            "pure_mcts_playout_num": self.pure_mcts_playout_num,
            "board_width": self.board_width,
            "board_height": self.board_height,
            "n_in_row": self.n_in_row,
            "buffer_len": len(self.data_buffer),
        }
        with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(state, f, indent=2)
        if is_best:
            best_dir = os.path.abspath(os.path.join(self.checkpoints_dir, f"run_{run_name}", "best"))
            os.makedirs(best_dir, exist_ok=True)
            torch.save(self.policy_value_net.policy_value_net.state_dict(), os.path.join(best_dir, "model.pth"))

    def run(self) -> None:
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"batch i:{i+1}, episode_len:{self.episode_len}, buffer_len:{len(self.data_buffer)}")
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                if (i + 1) % self.check_freq == 0:
                    print(f"current self-play batch: {i+1}")
                    win_ratio = self.policy_evaluate()
                    # save checkpoint
                    self._save_checkpoint(step=i+1, is_best=False)
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self._save_checkpoint(step=i+1, is_best=True)
                        # Apply YAML-driven pure MCTS upgrade schedule
                        if self.pure_sched_enabled and (self.best_win_ratio >= self.pure_sched_threshold) and (self.pure_mcts_playout_num < self.pure_sched_max):
                            new_playout = min(self.pure_mcts_playout_num + self.pure_sched_step, self.pure_sched_max)
                            print(f"[schedule] pure MCTS playout increase: {self.pure_mcts_playout_num} -> {new_playout}")
                            self.pure_mcts_playout_num = new_playout
                            if self.pure_sched_reset_best:
                                self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print("\n\rquit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--task", type=str, default=None, help="Registered task name")
    parser.add_argument("--base", type=str, default=None, help="Optional base YAML to merge")
    args = parser.parse_args()

    if args.task:
        cfg = load_task(args.task, base_path=args.base or os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    elif args.config:
        cfg = load_config(args.config, base_path=args.base)
    else:
        cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))

    set_seeds(int(cfg.get("experiment", {}).get("seed", 42)))
    runner = TrainRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()