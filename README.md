# The Malmo Collaborative AI Challenge - Team Pig Catcher

## Approach

The challenge involves 2 agents who can either *cooperate* or *defect*. The optimal policy, based on stag hunt [[1]](#references), depends on the policy of the other agent. Not knowing the other agent's policy, the optimal solution is then based on *modelling* the other agent's policy. Similarly, the challenge can be considered a *sequential social dilemma* [[2]](#references), as goals could change over time.

By treating the other agent as part of the environment, we can use model-free RL, and simply aim to maximise the reward of our agent. As a baseline we take a DRL algorithm - ACER [[3]](#references) - and train it against the evaluation agent (which randomly uses a focused or random strategy every episode).

We chose to approach this challenge using *hierarchical RL*. We assume there are 2 subpolicies, one for each type of partner agent. To do so, we use option heads [[4]](#references), whereby the agent has shared features, but separate heads for different subpolicies. In this case, ACER with 2 subpolicies has 2 Q-value heads and 2 policy heads. To choose which subpolicy to use at any given time, the agent also has an additional classifier head that is trained (using an oracle) to distinguish which option to use. Therefore, we ask the following questions:

- Can the agent distinguish between the two possible behaviours of the evaluation agent?
- Does the agent learn qualitatively different subpolicies?

We experienced difficulties getting both the classification and the policies to learn, so we then turned to using 2 A* planners as our subpolicies. Full results and more details can be found in our video.

## Design Decisions

For our baseline, we implemented ACER [[3]](#references) in PyTorch based on reference code [[5, 6]](#references). In addition, we augmented the state that the agent receives with the previous action, reward and a step counter [[7]](#references). Our challenge entry augments the agent with option heads [[4]](#references), and we aim to distinguish the different policies of the evaluation agent.

We also introduce a novel contribution - a batch version of ACER - which increases stability. We sample a batch of off-policy trajectories, and then truncate them to match the smallest.

## Instructions

Dependencies:

- [Python 2](https://www.python.org/)
- [PyTorch](http://pytorch.org/)
- [Plotly](https://plot.ly/python/)
- [Docker](https://www.docker.com/) + [docker-py](https://docker-py.readthedocs.io/en/stable/)

Firstly, [build the Malmo Docker image](https://github.com/Kaixhin/malmo-challenge/tree/master/docker). Secondly, [enable running Docker as a non-root user](https://docs.docker.com/engine/installation/linux/linux-postinstall/).

Run ACER with `OMP_NUM_THREADS=1 python pc_main.py`, and A* with the `--astar` option. The code automatically opens up Minecraft (Docker) instances.

## Discussion

[![Team Pig Catcher Discussion Video](https://img.youtube.com/vi/9XRL6d-yxp4/0.jpg)](https://www.youtube.com/watch?v=9XRL6d-yxp4)

## References

[1] [Game Theory of Mind](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000254)  
[2] [Multi-agent Reinforcement Learning in Sequential Social Dilemmas](https://arxiv.org/abs/1702.03037)  
[3] [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)  
[4] [Classifying Options for Deep Reinforcement Learning](https://arxiv.org/abs/1604.08153)  
[5] [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)  
[6] [pfnet/ChainerRL](https://github.com/pfnet/chainerrl)  
[7] [Learning to Navigate in Complex Environments](https://arxiv.org/abs/1611.03673)  

---

---

This repository contains the task definition and example code for the [Malmo Collaborative AI Challenge](https://www.microsoft.com/en-us/research/academic-program/collaborative-ai-challenge/).
This challenge is organized to encourage research in collaborative AI - to work towards AI agents 
that learn to collaborate to solve problems and achieve goals. 
You can find additional details, including terms and conditions, prizes and information on how to participate at the [Challenge Homepage](https://www.microsoft.com/en-us/research/academic-program/collaborative-ai-challenge/).

[![Join the chat at https://gitter.im/malmo-challenge/Lobby](https://badges.gitter.im/malmo-challenge/Lobby.svg)](https://gitter.im/malmo-challenge/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Microsoft/malmo-challenge/blob/master/LICENSE)

----

**Notes for challenge participants:** Once you and your team decide to participate in the challenge, please make sure to register your team at our [Registration Page](https://www.surveygizmo.com/s3/3299773/The-Collaborative-AI-Challenge). On the registration form, you need to provide a link to the GitHub repository that will 
contain your solution. We recommend that you fork this repository (<a href="https://help.github.com/articles/fork-a-repo/" target="_blank">learn how</a>), 
and provide address of the forked repo. You can then update your submission as you make progress on the challenge task. 
We will consider the version of the code on branch master at the time of the submission deadline as your challenge submission. Your submission needs to contain code in working order, a 1-page description of your approach, and a 1-minute video that shows off your agent. Please see the [challenge terms and conditions]() for further details.

----

**Jump to:**

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Minimal installation](#minimal-installation)
  - [Optional extensions](#optional-extensions)

- [Getting started](#getting-started)
  - [Play the challenge task](#play-the-challenge-task)
  - [Run your first experiment](#run-your-first-experiment)

- [Next steps](#next-steps)
  - [Run an experiment in Docker on Azure](#run-an-experiment-in-docker-on-azure)
  - [Resources](#resources)

# Installation

## Prerequisites

- [Python](https://www.python.org/) 2.7+ (recommended) or 3.5+
- [Project Malmo](https://github.com/Microsoft/malmo) - we recommend downloading the [Malmo-0.21.0 release](https://github.com/Microsoft/malmo/releases) and installing dependencies for [Windows](https://github.com/Microsoft/malmo/blob/master/doc/install_windows.md), [Linux](https://github.com/Microsoft/malmo/blob/master/doc/install_linux.md) or [MacOS](https://github.com/Microsoft/malmo/blob/master/doc/install_macosx.md). Test your Malmo installation by [launching Minecraft with Malmo](https://github.com/Microsoft/malmo#launching-minecraft-with-our-mod) and [launching an agent](https://github.com/Microsoft/malmo#launch-an-agent).

## Minimal installation

```
pip install -e git+https://github.com/Microsoft/malmo-challenge#egg=malmopy
```

or 

```
git clone https://github.com/Microsoft/malmo-challenge
cd malmo-challenge
pip install -e .
```

## Optional extensions

Some of the example code uses additional dependencies to provide 'extra' functionality. These can be installed using:

```
pip install -e '.[extra1, extra2]'
```
For example to install gym and chainer:

```
pip install -e '.[gym]'
```

Or to install all extras:

```
pip install -e '.[all]'
```

The following extras are available:
- `gym`: [OpenAI Gym](https://gym.openai.com/) is an interface to a wide range of reinforcement learning environments. Installing this extra enables the Atari example agents in [samples/atari](samples/atari) to train on the gym environments. *Note that OpenAI gym atari environments are currently not available on Windows.*
- `tensorflow`: [TensorFlow](https://www.tensorflow.org/) is a popular deep learning framework developed by Google. In our examples it enables visualizations through [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).


# Getting started

## Play the challenge task

The challenge task takes the form of a mini game, called Pig Chase. Learn about the game, and try playing it yourself on our [Pig Chase Challenge page](ai_challenge/pig_chase/README.md).

## Run your first experiment

See how to [run your first baseline experiment](ai_challenge/pig_chase/README.md#run-your-first-experiment) on the [Pig Chase Challenge page](ai_challenge/pig_chase/README.md).

# Next steps

## Run an experiment in Docker on Azure

Docker is a virtualization platform that makes it easy to deploy software with all its dependencies. 
We use docker to run experiments locally or in the cloud. Details on how to run an example experiment using docker are in the [docker README](docker/README.md).


## Resources

- [Malmo Platform Tutorial](https://github.com/Microsoft/malmo/blob/master/Malmo/samples/Python_examples/Tutorial.pdf)
- [Azure Portal](portal.azure.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Machine on Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/virtual-machines-linux-docker-machine)
- [CNTK Tutorials](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/tutorials/)
- [CNTK Documentation](https://github.com/Microsoft/CNTK/wiki)
- [Chainer Documentation](http://docs.chainer.org/en/stable/)
- [TensorBoard Documentation](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
