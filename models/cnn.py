# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Removed deprecated Variable import
import numpy as np
from typing import Tuple, List, Sequence, Iterable


def set_learning_rate(optimizer: optim.Optimizer, lr: float) -> None:
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width: int, board_height: int):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing (log_action_probs, state_value)."""
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        # Replace deprecated F.tanh with torch.tanh
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network wrapper encapsulating training and inference."""
    def __init__(self, board_width: int, board_height: int,
                 model_file: str = None, use_gpu: bool = False):
        # Unified device management
        self.use_gpu = bool(use_gpu) and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference for a batch of states (no grad):
        input: a batch of states, shape: (batch, 4, board_w, board_h)
        output: (action_probs, state_values)
        action_probs: numpy array of shape (batch, board_w*board_h)
        state_values: numpy array of shape (batch, 1)
        """
        # Efficient tensor creation from numpy list
        state_arr = np.asarray(state_batch, dtype=np.float32)
        state_tensor = torch.from_numpy(state_arr).to(self.device, non_blocking=True)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs_t = torch.exp(log_act_probs)
        return act_probs_t.detach().cpu().numpy(), value.detach().cpu().numpy()

    def policy_value_torch(self, state_batch: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Torch-native inference for a batch of states (no grad): returns tensors on self.device.
        action_probs: Tensor of shape (batch, board_w*board_h)
        state_values: Tensor of shape (batch, 1)
        """
        state_arr = np.asarray(state_batch, dtype=np.float32)
        state_tensor = torch.from_numpy(state_arr).to(self.device, non_blocking=True)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs_t = torch.exp(log_act_probs)
        return act_probs_t, value

    def policy_value_fn(self, board) -> Tuple[Iterable[Tuple[int, float]], float]:
        """
        A callable compatible with MCTS: given board, returns
        - iterable of (action, probability) for legal positions
        - scalar value in [-1, 1]
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height)).astype(np.float32)
        state_tensor = torch.from_numpy(current_state).to(self.device, non_blocking=True)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
            act_probs_flat = torch.exp(log_act_probs).detach().cpu().numpy().flatten()
        value_scalar = float(value.detach().cpu().numpy().ravel()[0])
        act_probs = zip(legal_positions, act_probs_flat[legal_positions])
        return act_probs, value_scalar

    def train_step(self, state_batch: Sequence[np.ndarray], mcts_probs: Sequence[np.ndarray], winner_batch: Sequence[float], lr: float) -> Tuple[float, float]:
        """Perform a single training step and return (loss, entropy)."""
        # Efficient tensor creation from python lists
        state_arr = np.asarray(state_batch, dtype=np.float32)
        mcts_arr = np.asarray(mcts_probs, dtype=np.float32)
        winner_arr = np.asarray(winner_batch, dtype=np.float32)
        state_tensor = torch.from_numpy(state_arr).to(self.device, non_blocking=True)
        mcts_tensor = torch.from_numpy(mcts_arr).to(self.device, non_blocking=True)
        winner_tensor = torch.from_numpy(winner_arr).to(self.device, non_blocking=True)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_tensor)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_tensor)
        policy_loss = -torch.mean(torch.sum(mcts_tensor*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file: str) -> None:
        """Save model params to file."""
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
