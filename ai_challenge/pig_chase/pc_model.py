# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import init

from pc_utils import ACTION_SIZE, STATE_SIZE

class ActorCritic(nn.Module):
  def __init__(self, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = STATE_SIZE[0] * STATE_SIZE[1] * STATE_SIZE[2]

    self.elu = nn.ELU(inplace=True)
    self.softmax = nn.Softmax()

    # Pass state into model body
    self.conv1 = nn.Conv2d(STATE_SIZE[0], 64, 4, stride=2)
    self.conv2 = nn.Conv2d(64, 64, 3)
    self.fc1 = nn.Linear(2304, hidden_size)
    # Pass previous action, reward and timestep directly into LSTM
    self.lstm = nn.LSTMCell(hidden_size + ACTION_SIZE + 2, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, ACTION_SIZE)
    self.fc_critic = nn.Linear(hidden_size, ACTION_SIZE)

    # Orthogonal weight initialisation
    for name, p in self.named_parameters():
      if 'weight' in name:
        init.orthogonal(p)
      elif 'bias' in name:
        init.constant(p, 0)
    # Set LSTM forget gate bias to 1
    for name, p in self.lstm.named_parameters():
      if 'bias' in name:
        n = p.size(0)
        forget_start_idx, forget_end_idx = n // 4, n // 2
        init.constant(p[forget_start_idx:forget_end_idx], 1)

  def forward(self, x, h):
    state, extra = x.narrow(1, 0, self.state_size), x.narrow(1, self.state_size, ACTION_SIZE + 2)
    state = state.view(state.size(0), STATE_SIZE[0], STATE_SIZE[1], STATE_SIZE[2])  # Restore spatial structure
    x = self.elu(self.conv1(state))
    x = self.elu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = self.elu(self.fc1(x))
    h = self.lstm(torch.cat((x, extra), 1), h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1)  # V is expectation of Q under π
    return policy, Q, V, h
