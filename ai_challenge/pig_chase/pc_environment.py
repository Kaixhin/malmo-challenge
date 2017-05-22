import socket
import time
import docker
import torch
from torch import multiprocessing as mp
from common import ENV_AGENT_NAMES
from agent import PigChaseChallengeAgent, RandomAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder, PigChaseSymbolicTopDownStateBuilder

from pc_utils import GlobalVar


# Taken from Minecraft/launch_minecraft_in_background.py
def _port_has_listener(port):
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
    self.agent_type = GlobalVar()

    # Assume Minecraft launched if port has listener, launch otherwise
    if not _port_has_listener(agent_port):
      self._launch_malmo(docker_client, agent_port)
    print('Malmo running on port ' + str(agent_port))
    if not _port_has_listener(partner_port):
      self._launch_malmo(docker_client, partner_port)
    print('Malmo running on port ' + str(partner_port))

    # Set up partner agent env in separate process
    p = mp.Process(target=self._run_partner, args=(clients, ))
    p.daemon = True
    p.start()
    time.sleep(3)

    # Set up agent env
    self.env = PigChaseEnvironment(clients, PigChaseSymbolicTopDownStateBuilder(gray=False), role=1, randomize_positions=True)

  def get_class_label(self):
    return self.agent_type.value() - 1

  def reset(self):
    symbols, observation = self.env.reset()
    while observation is None:  # May happen if episode ended with first action of other agent
      symbols, observation = self.env.reset()
    return symbols, _map_to_observation(observation)

  def step(self, action):
    (symbols, observation), reward, done = self.env.do(action)
    return symbols, _map_to_observation(observation), reward, done, None  # Do not return any extra info

  def close(self):
    return  # TODO: Kill processes + Docker containers

  def _launch_malmo(self, client, port):
    # Launch Docker container
    client.containers.run('malmo', '-port ' + str(port), detach=True, network_mode='host')
    # Check for port to come up
    launched = False
    for _ in range(100):
      time.sleep(3)
      if _port_has_listener(port):
        launched = True
        break
    # Quit if Malmo could not be launched
    if not launched:
      exit(1)

  # Runs partner in separate env
  def _run_partner(self, clients):
    env = PigChaseEnvironment(clients, PigChaseSymbolicStateBuilder(), role=0, randomize_positions=True)
    agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0])
    self.agent_type.set(type(agent.current_agent) == RandomAgent and PigChaseEnvironment.AGENT_TYPE_1 or PigChaseEnvironment.AGENT_TYPE_2)
    obs = env.reset(self.agent_type)
    reward = 0
    agent_done = False
    while True:
      # Select an action
      action = agent.act(obs, reward, agent_done, is_training=True)
      # Reset if needed
      if env.done:
        self.agent_type.set(type(agent.current_agent) == RandomAgent and PigChaseEnvironment.AGENT_TYPE_1 or PigChaseEnvironment.AGENT_TYPE_2)
        obs = env.reset(self.agent_type)
      # Take a step
      obs, reward, agent_done = env.do(action)
