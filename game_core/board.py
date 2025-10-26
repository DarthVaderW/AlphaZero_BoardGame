# -*- coding: utf-8 -*-
"""
Self-contained Board implementation for Gobang (Gomoku), adapted from original.
No external imports from top-level game.py.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Any

class Board:
    """Board for the Gobang game.
    Manages state, legal moves, and win detection.
    """
    def __init__(self, config: Dict[str, Any]):
        # Store raw YAML game config dict
        self.config: Dict[str, Any] = dict(config or {})
        self.width = int(self.config.get('board_width', self.config.get('width', 8)))
        self.height = int(self.config.get('board_height', self.config.get('height', 8)))
        self.states: Dict[int, int] = {}
        self.n_in_row = int(self.config.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
        self.current_player = self.players[0]
        self.availables: List[int] = []
        self.last_move: int = -1
        # Winner check selection: 'scan_all' | 'last_move'
        self.winner_check_method = str(self.config.get('winner_check', 'scan_all'))
        # Scan-all early stop flag (True by default)
        self.scan_all_early_stop: bool = bool(self.config.get('scan_all_early_stop', True))
        # 2D grid (height x width), 0 empty, 1/2 players
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

    def init_board(self, start_player: int = 0) -> None:
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states.clear()
        self.last_move = -1
        # reset grid
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

    def move_to_location(self, move: int) -> List[int]:
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location: List[int]) -> int:
        if len(location) != 2:
            return -1
        h, w = location
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self) -> np.ndarray:
        """Return the board state from the perspective of the current player.
        Shape: (4, width, height), channel order matches original implementation.
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]

    def do_move(self, move: int) -> None:
        p = self.current_player
        self.states[move] = p
        # update 2D grid
        h = move // self.width
        w = move % self.width
        self.grid[h, w] = p
        if move in self.availables:
            self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move

    def has_a_winner(self):
        # Early exit: not enough moves to have a winner
        if len(self.states) < self.n_in_row * 2 - 1:
            return False, -1
        # Dispatch based on configured method
        if self.winner_check_method == 'last_move':
            return self._has_a_winner_lastmove()
        return self._has_a_winner_scan_all()

    def _has_a_winner_scan_all(self):
        """Original scan-all implementation over the whole board.
        Supports optional no-early-stop full scan for benchmarking.
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        moved = list(set(range(width * height)) - set(self.availables))
        found = False
        winner = -1
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]
            if (w in range(width - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                if self.scan_all_early_stop:
                    return True, player
                found = True
                if winner == -1:
                    winner = player
            if (h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                if self.scan_all_early_stop:
                    return True, player
                found = True
                if winner == -1:
                    winner = player
            if (w in range(width - n + 1) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                if self.scan_all_early_stop:
                    return True, player
                found = True
                if winner == -1:
                    winner = player
            if (w in range(n - 1, width) and h in range(height - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                if self.scan_all_early_stop:
                    return True, player
                found = True
                if winner == -1:
                    winner = player
        if found:
            return True, winner
        return False, -1

    def _has_a_winner_lastmove(self):
        # If no last move, no winner
        if self.last_move < 0:
            return False, -1
        width = self.width
        height = self.height
        n = self.n_in_row
        y = self.last_move // width
        x = self.last_move % width
        player = int(self.grid[y, x])
        if player == 0:
            return False, -1
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dy, dx in directions:
            count = 1
            for sign in (1, -1):
                ny, nx = y, x
                while True:
                    ny += dy * sign
                    nx += dx * sign
                    if 0 <= ny < height and 0 <= nx < width and int(self.grid[ny, nx]) == player:
                        count += 1
                        if count >= n:
                            return True, player
                    else:
                        break
        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self) -> int:
        return self.current_player