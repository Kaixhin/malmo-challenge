import numpy as np
from common import visualize_training, Entity, ENV_TARGET_NAMES, ENV_ENTITIES, ENV_AGENT_NAMES, \
    ENV_ACTIONS, ENV_CAUGHT_REWARD, ENV_BOARD_SHAPE
from six.moves import range

from malmopy.agent import AStarAgent
from malmopy.agent import BaseAgent
from agent import FocusedAgent

import random
import torch
import torch.nn as nn
from torch.nn import init


from envwrap import create_env

# Constants
STATE_SIZE = (3, 18, 18)
ACTION_SIZE = 3

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.state_size = STATE_SIZE[0] * STATE_SIZE[1] * STATE_SIZE[2]
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(STATE_SIZE[0], 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(1152, 64)
        self.lstm = nn.LSTMCell(64, 64)
        self.fc_class = nn.Linear(64, 1)

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
        state = x.narrow(1, 0, self.state_size).contiguous()
        state = state.view(state.size(0), STATE_SIZE[0], STATE_SIZE[1], STATE_SIZE[2]).contiguous()
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.elu(self.fc1(x))
        h = self.lstm(x, h)  # h is (hidden state, cell state)
        x = h[0]
        meta_policy = self.sigmoid(self.fc_class(x))
        return meta_policy, h

class OptionsStar(BaseAgent):
    """Our options star agent"""

    def __init__(self, name):

        nb_actions = len(ENV_ACTIONS)
        self._agents = []
        self._agents.append(FocusedAgent(name, 'lapis_block')) # give up agent
        self._agents.append(FocusedAgent(name, 'Pig')) # pig chase agent
        self.classifier = Classifier()

    # Return state in C H W format (as a batch)
    def _map_to_observation(self, observation):
        observation = torch.Tensor(observation)
        return observation.permute(2, 1, 0).contiguous().unsqueeze(0)


    def act(self, new_state, reward, done, lstm_state, is_training=False):
        new_state_sym, new_state_topdown = new_state
        input_state = self._map_to_observation(new_state_topdown)
        meta_policy, lstm_state = self.classifier( input_state, lstm_state )

        cls = 1 if 0.5 < meta_policy.data[0][0] else 0
        if cls == 0:
            print('0')
            return self._agents[0].act(new_state_sym, reward, done, is_training), meta_policy, lstm_state
        else:
            print('1')
            return self._agents[1].act(new_state_sym, reward, done, is_training), meta_policy, lstm_state
