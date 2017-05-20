import os
from os import path
import signal
import socket
import time
import docker
import torch
from torch import multiprocessing as mp
from common import ENV_AGENT_NAMES
from agent import PigChaseChallengeAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder, PigChaseTopDownStateBuilder


# Taken from Minecraft/launch_minecraft_in_background.py
def port_has_listener(port):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  result = sock.connect_ex(('127.0.0.1', port))
  sock.close()
  return result == 0


# Return state in C H W format (as a batch)
def _map_to_observation(observation):
  observation = torch.Tensor(observation)
  return observation.permute(2, 1, 0).contiguous().unsqueeze(0)


class Env():
  def __init__(self, rank):
    docker_client = docker.from_env()
    agent_port, partner_port = 10000 + rank, 20000 + rank
    clients = [('127.0.0.1', agent_port), ('127.0.0.1', partner_port)]
    
    # Assume Minecraft launched if port has listener, launch otherwise
    if not port_has_listener(agent_port):
      self._launch_malmo(docker_client, agent_port)
    print('Malmo running on port ' + str(agent_port))
    if not port_has_listener(partner_port):
      self._launch_malmo(docker_client, partner_port)
    print('Malmo running on port ' + str(partner_port))

    # Set up internal environment
    self._env = PigChaseEnvironment(clients, PigChaseTopDownStateBuilder(gray=False), role=1, randomize_positions=True)
    p = mp.Process(target=self._run_challenge_agent, args=(clients,))
    p.start()
    time.sleep(5)

  def reset(self, raw_observation=False):
    if raw_observation:
      return self._env.reset()
    else:
      return _map_to_observation(self._env.reset())

  def step(self, action):
    observation, reward, done = self._env.do(action)
    return _map_to_observation(observation), reward, done, None  # Do not return any extra info

  def close(self):
    return  # TODO: Kill Docker containers

  def _launch_malmo(self, client, port):
    client.containers.run('malmo', '-port ' + str(port), detach=True, network_mode='host')
    launched = False
    for _ in range(100):
      time.sleep(3)
      if port_has_listener(port):
        launched = True
        break
    # Quit if Malmo could not be launched
    if not launched:
      exit(1)

  # Run agent in loop forever
  def _agent_loop(self, agent, env):
    observation = env.reset()
    reward = 0
    done = False

    while True:
      if env.done:
        observation = env.reset()
        while observation is None:
          observation = env.reset()  # If episode ended with first action of other agent, reset

      # Select an action
      action = agent.act(observation, reward, done, is_training=True)
      # Step
      observation, reward, done = env.do(action)

  # Run challenge agent
  def _run_challenge_agent(self, clients):
    env = PigChaseEnvironment(clients, PigChaseSymbolicStateBuilder(), role=0, randomize_positions=True)
    agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0])
    self._agent_loop(agent, env)
