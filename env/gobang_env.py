# -*- coding: utf-8 -*-
"""
Gobang environment adapter over existing game.Board/Game.
This keeps interface minimal and decoupled from self-play mechanics.
"""
from typing import Tuple, Any, List
import numpy as np
from .base_env import BaseEnv
from game_core import Board, Game

class GobangEnv(BaseEnv):
    def __init__(self, board_width: int = 15, board_height: int = 15, n_in_row: int = 4):
        # Align with Board's config-based constructor
        cfg = {"board_width": board_width, "board_height": board_height, "n_in_row": n_in_row}
        self.board = Board(config=cfg)
        self.game = Game(self.board)
        self._last_state = None

    def reset(self) -> Any:
        self.board.init_board(start_player=0)
        self._last_state = self.board.current_state()
        return self._last_state

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        self.board.do_move(action)
        state = self.board.current_state()
        end, winner = self.board.game_end()
        reward = 0.0
        if end:
            if winner == -1:
                reward = 0.0
            else:
                # reward from perspective of the player who just moved
                reward = 1.0
        self._last_state = state
        return state, reward, end, {"winner": winner}

    def legal_actions(self, state: Any = None) -> List[int]:
        return list(self.board.availables)

    def current_player(self, state: Any = None) -> int:
        return self.board.get_current_player()

    def is_terminal(self, state: Any = None) -> bool:
        end, _ = self.board.game_end()
        return bool(end)

    def result(self, state: Any = None) -> int:
        end, winner = self.board.game_end()
        if not end:
            return 0
        if winner == -1:
            return 0
        return 1 if winner == self.board.get_current_player() else -1

    def observe(self, state: Any = None) -> np.ndarray:
        return self.board.current_state()

    def render(self) -> None:
        # simple ASCII render
        w, h = self.board.width, self.board.height
        grid = [["." for _ in range(w)] for _ in range(h)]
        for move, player in self.board.states.items():
            y = move // w
            x = move % w
            grid[y][x] = "X" if player == 1 else "O"
        for row in grid:
            print(" ".join(row))