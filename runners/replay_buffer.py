# -*- coding: utf-8 -*-
"""
Replay buffer and data augmentation consistent with the original TrainPipeline.
"""
from collections import deque
from typing import List, Tuple
import numpy as np

Sample = Tuple[np.ndarray, np.ndarray, float]

class ReplayBuffer:
    def __init__(self, maxlen: int = 10000):
        self.buffer = deque(maxlen=maxlen)

    def extend(self, items: List[Sample]) -> None:
        self.buffer.extend(items)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> List[Sample]:
        import random
        return random.sample(self.buffer, batch_size)


def augment_play_data(play_data: List[Sample], board_width: int, board_height: int) -> List[Sample]:
    """Augment the data set by rotation and flipping, replicating original logic.
    play_data: [(state, mcts_prob, winner_z), ...]
    """
    extend_data: List[Sample] = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state], dtype=np.float32)
            equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(board_height, board_width)), i)
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten().astype(np.float32),
                float(winner)
            ))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state], dtype=np.float32)
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten().astype(np.float32),
                float(winner)
            ))
    return extend_data