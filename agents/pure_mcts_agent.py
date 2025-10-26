# -*- coding: utf-8 -*-
"""
Pure MCTS Agent wrapper built on existing mcts_pure.MCTSPlayer.
Used for periodic evaluation in training.
"""
from .base import BaseAgent
from mcts.pure import MCTSPlayer as MCTS_Pure

class PureMCTSAgent(BaseAgent):
    def __init__(self, c_puct: float = 5.0, n_playout: int = 1000):
        self.player = MCTS_Pure(c_puct=c_puct, n_playout=n_playout)

    def reset(self) -> None:
        self.player.reset_player()

    def to_player(self) -> MCTS_Pure:
        return self.player