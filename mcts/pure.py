# -*- coding: utf-8 -*-
"""
Self-contained pure MCTS implementation (no network), adapted from original.
"""
from __future__ import annotations
import numpy as np
import copy
from operator import itemgetter
from typing import Callable, Iterable, List, Tuple


def rollout_policy_fn(board) -> Iterable[Tuple[int, float]]:
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board) -> Tuple[Iterable[Tuple[int, float]], float]:
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0.0


class TreeNode:
    def __init__(self, parent, prior_p: float):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = prior_p

    def expand(self, action_priors: Iterable[Tuple[int, float]]) -> None:
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct: float):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value: float) -> None:
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value: float) -> None:
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct: float) -> float:
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self) -> bool:
        return self._children == {}

    def is_root(self) -> bool:
        return self._parent is None


class MCTS:
    def __init__(self, policy_value_fn: Callable, c_puct: float = 5.0, n_playout: int = 10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state) -> None:
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        action_probs, _ = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit: int = 1000) -> int:
        player = state.get_current_player()
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")
        if winner == -1:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state) -> int:
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move: int) -> None:
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self) -> str:
        return "MCTS"


class MCTSPlayer:
    def __init__(self, c_puct: float = 5.0, n_playout: int = 2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.player = 1

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, board) -> int:
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self) -> str:
        return f"MCTS {self.player}"


def make_player(c_puct: float = 5.0, n_playout: int = 1000) -> MCTSPlayer:
    return MCTSPlayer(c_puct=c_puct, n_playout=n_playout)