# -*- coding: utf-8 -*-
from .alpha_zero import MCTSPlayer as AlphaZeroMCTSPlayer, make_player as make_alpha_zero
from .pure import MCTSPlayer as PureMCTSPlayer, make_player as make_pure

__all__ = [
    "AlphaZeroMCTSPlayer",
    "PureMCTSPlayer",
    "make_alpha_zero",
    "make_pure",
]