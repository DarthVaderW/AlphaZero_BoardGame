# -*- coding: utf-8 -*-
"""
Self-contained AlphaZero-style MCTS implementation.
No dependency on top-level mcts_alphaZero.py.
"""
from __future__ import annotations
import numpy as np
import copy
from typing import Callable, Tuple, List


def softmax(x: np.ndarray) -> np.ndarray:
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    def __init__(self, parent, prior_p: float):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = prior_p

    def expand(self, action_priors: List[Tuple[int, float]]) -> None:
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
        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp: float = 1e-3):
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        if not act_visits:
            return [], np.array([])
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move: int) -> None:
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self) -> str:
        return "MCTS"


class MCTSPlayer:
    def __init__(self, policy_value_function: Callable, c_puct: float = 5.0, n_playout: int = 2000, is_selfplay: int = 0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.player = 1

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp: float = 1e-3, return_prob: int = 0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            if len(acts) == 0:
                # fallback: uniform over sensible moves
                probs = np.ones(len(sensible_moves)) / len(sensible_moves)
                move = np.random.choice(sensible_moves, p=probs)
                move_probs[sensible_moves] = probs
                self.mcts.update_with_move(-1)
                return (move, move_probs) if return_prob else move
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self) -> str:
        return f"MCTS {self.player}"


def make_player(policy_value_fn: Callable, c_puct: float = 5.0, n_playout: int = 400, is_selfplay: int = 1) -> MCTSPlayer:
    return MCTSPlayer(policy_value_fn, c_puct=c_puct, n_playout=n_playout, is_selfplay=is_selfplay)