from time import sleep

from agent import PigChaseChallengeAgent
from common import ENV_AGENT_NAMES
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder, PigChaseTopDownStateBuilder
from multiprocessing import Process

"""
 This environment define a PigChase game with Pig Chase challenge agent
"""
class create_env():
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

        # build opponent environment and connect the pigchase agent
        p = Process(target=self._run_challenge_agent, args=(self._clients,))
        p.start()
        sleep(5)


    def reset(self):
        return self._env.reset()

    def step(self, action):
        env = self._env
        obs, reward, done = env.do(action)
        info = None

        return obs, reward, done, info

    def _run_challenge_agent(self, clients):
        builder = PigChaseSymbolicStateBuilder()
        env = PigChaseEnvironment(clients, builder, role=0, randomize_positions=True)
        agent = PigChaseChallengeAgent(ENV_AGENT_NAMES[0])
        self._agent_loop(agent, env)

    def _agent_loop(self, agent, env):
        #EVAL_EPISODES = 100
        agent_done = False
        reward = 0
        episode = 0
        obs = env.reset()

        #while episode < EVAL_EPISODES:
        while True:
            # check if env needs reset
            if env.done:
                print('Episode %d (%.2f)%%' % (episode, (episode / EVAL_EPISODES) * 100.))

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


if __name__ == '__main__':
    # train
    
