# -*- coding: utf-8 -*-
"""
MCTS-based Agent wrapper using our self-contained AlphaZero MCTS.
Keeps AlphaZero self-play behavior (Dirichlet noise, tree reuse).
"""
from typing import Optional
from mcts.alpha_zero import MCTSPlayer

class MCTSAgent:
    def __init__(self, policy_value_fn, c_puct: float = 5.0, n_playout: int = 400, is_selfplay: int = 1):
        self.player = MCTSPlayer(policy_value_fn, c_puct=c_puct, n_playout=n_playout, is_selfplay=is_selfplay)
        self.temperature: float = 1.0

    def set_temperature(self, temp: float) -> None:
        self.temperature = float(temp)

    def reset(self) -> None:
        self.player.reset_player()

    def to_player(self) -> MCTSPlayer:
        return self.player