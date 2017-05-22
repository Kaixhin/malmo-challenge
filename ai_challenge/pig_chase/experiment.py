from time import sleep

from agent import PigChaseChallengeAgent
from common import ENV_AGENT_NAMES
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder, PigChaseTopDownStateBuilder
from environment import MyCustomBuilder
from multiprocessing import Process
from malmopy.agent import RandomAgent
from options_star_agent import OptionsStar

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
"""
 This environment define a PigChase game with Pig Chase challenge agent
"""

class Env():
    def __init__(self, clients=[('127.0.0.1', 10000), ('127.0.0.1', 10001)], builder=None):
        self.episode = 0
        self._clients = clients
        if builder is None:
            # use Top Down state
            self._state_builder = PigChaseTopDownStateBuilder()
        else:
            self._state_builder = builder

        # build the environment
        self._env = PigChaseEnvironment(self._clients, self._state_builder,
                                  role=1, randomize_positions=True)

        self._agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0])
        # build opponent environment and connect the pigchase agent
        p = Process(target=self._run_challenge_agent, args=(self._clients, self._agent))
        p.start()
        sleep(5)

    def done(self):
        return self._env.done

    def reset(self):
        return self._env.reset()

    def step(self, action):
        env = self._env
        obs, reward, done = env.do(action)

        return obs, reward, done, self._agent.get_label()

    def _run_challenge_agent(self, clients, agent):
        builder = PigChaseSymbolicStateBuilder()
        env = PigChaseEnvironment(clients, builder, role=0, randomize_positions=True, human_speed=False)
        #self._agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0])
        #self._agent_loop(self._agent, env)
        self._agent_loop(agent, env)

    def _agent_loop(self, agent, env):
        agent_done = False
        reward = 0
        episode = 0
        obs = env.reset()

        while True:
            # check if env needs reset
            if env.done:
                obs = env.reset()
                while obs is None:
                    # this can happen if the episode ended with the first
                    # action of the other agent
                    print('Warning: received obs == None.')
                    obs = env.reset()

                episode += 1

            # select an action
            action = agent.act(obs, reward, agent_done, is_training=True)
            # take a step
            obs, reward, agent_done = env.do(action)

            #self.agent_type.set(type(agent.current_agent) == RandomAgent and PigChaseEnvironment.AGENT_TYPE_1 or PigChaseEnvironment.AGENT_TYPE_2)


if __name__ == '__main__':
    # create env
    builder = MyCustomBuilder()
    env = Env(builder=builder)
    reward = 0
    episode = 0
    obs = env.reset()
    TRAIN_EPISODE = 100
    agent_done = False
    hx, cx = Variable(torch.zeros(1, 64)), Variable(torch.zeros(1, 64))
    class_loss = 0
    train_steps = 0
    episode_reward = 0
    average_reward = 0
    # create agent
    agent = OptionsStar(ENV_AGENT_NAMES[1])

    # optimizer
    learning_rate = 1e-3
    optimiser = torch.optim.Adam(agent.classifier.parameters(), lr=learning_rate)

    while episode <  TRAIN_EPISODE:
        if env.done():
            #Backprop
            optimiser.zero_grad()
            class_loss.backward()
            # Gradient L1 norm clipping
            nn.utils.clip_grad_norm(agent.classifier.parameters(), 40, 1)
            optimiser.step()

            #reset hidden state
            hx, cx = Variable(torch.zeros(1, 64)), Variable(torch.zeros(1, 64))
            print('Episode %d (%.2f)%%' % (episode, (episode / TRAIN_EPISODE) * 100.))

            print('Classification error: ' + str(class_loss.data[0]))
            print('Episode reward: ' + str(episode_reward))
            class_loss = 0
            episode_reward = 0

            while obs is None:
                # this can happen if the episode ended with the first
                # action of the other agent
                print('Warning: received obs == None.')
                obs = env.reset()

            episode += 1
            env.reset()

        # obs
        #new_state_sym, new_state_topdown = obs
        #new_state_topdown = Variable(torch.from_numpy(new_state_topdown))
        # select an action
        #action, meta_policy, (hx, cx) = agent.act((new_state_sym, new_state_topdown), reward, agent_done, (hx, cx), is_training=True)
        action, meta_policy, (hx, cx) = agent.act(obs, reward, agent_done, (hx, cx), is_training=True)

        # take a step
        obs, reward, agent_done, label = env.step(action)

        episode_reward += reward
        average_reward += reward
        target_tensor = torch.FloatTensor().resize_as_(meta_policy.data).fill_(label)
        #compute classification loss
        class_loss += F.binary_cross_entropy(meta_policy, Variable(target_tensor))
        train_steps += 1

    print('average reward: '+ str(average_reward/100))
