import os
from os import path
import signal
import socket
import subprocess
import time
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
    # Find Malmo directory
    malmo_path = path.join(path.dirname(os.environ['MALMO_XSD_PATH']), 'Minecraft', 'launchClient.sh')
    agent_port, partner_port = 10000 + rank, 20000 + rank
    clients = [('127.0.0.1', agent_port), ('127.0.0.1', partner_port)]
    self.processes = []

    # Assume Minecraft launched if port has listener, launch otherwise
    if not port_has_listener(agent_port):
      self._launch_malmo(malmo_path, agent_port)
    if not port_has_listener(partner_port):
      self._launch_malmo(malmo_path, partner_port)

    # Set up internal environment
    self._env = PigChaseEnvironment(clients, PigChaseTopDownStateBuilder(gray=False), role=1, randomize_positions=True)
    p = mp.Process(target=self._run_challenge_agent, args=(clients,))
    p.start()
    time.sleep(5)

  def get_class_label(self):
    return self._agent.current_agent_id

  def reset(self, raw_observation=False):
    if raw_observation:
      return self._env.reset()
    else:
      return _map_to_observation(self._env.reset())

  def step(self, action):
    observation, reward, done = self._env.do(action)
    return _map_to_observation(observation), reward, done, None  # Do not return any extra info

  def close(self):
    for process in self.processes:
      os.killpg(os.getpgid(process.pid), signal.SIGTERM)

  def _launch_malmo(self, malmo_path, port):
    process = subprocess.Popen(malmo_path + ' -port ' + str(port), close_fds=True, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, preexec_fn=os.setsid)
    self.processes.append(process)
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
    self._agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0])
    self._agent_loop(self._agent, env)
