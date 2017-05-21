# -*- coding: utf-8 -*-
import time
from datetime import datetime
import torch
from torch.autograd import Variable

from pc_environment import Env
from pc_model import ActorCritic
from pc_utils import ACTION_SIZE, action_to_one_hot, extend_input, plot_line

def leaderboard_save(accumulators, experiment_name, filepath):
    """
    Save the evaluation results in a JSON file
    understandable by the leaderboard.

    Note: The leaderboard will not accept a submission if you already
    uploaded a file with the same experiment name.

    :param experiment_name: An identifier for the experiment
    :param filepath: Path where to store the results file
    :return:
    """

    assert experiment_name is not None, 'experiment_name cannot be None'

    from json import dump
    from os.path import exists, join, pardir, abspath
    from os import makedirs
    from numpy import mean, var

    # Compute metrics
    metrics = {key: {'mean': mean(buffer),
                     'var': var(buffer),
                     'count': len(buffer)}
               for key, buffer in accumulators.items()}

    metrics['experimentname'] = experiment_name

    try:
        filepath = abspath(filepath)
        parent = join(pardir, filepath)
        if not exists(parent):
            makedirs(parent)

        with open(filepath, 'w') as f_out:
            dump(metrics, f_out)

        print('==================================')
        print('Evaluation done, results written at %s' % filepath)

    except Exception as e:
        print('Unable to save the results: %s' % e)


def test(rank, args, T, shared_model):
  torch.manual_seed(args.seed + rank)

  env = Env(rank)
  model = ActorCritic(args.hidden_size)
  model.eval()

  can_test = True  # Test flag
  t_start = 1  # Test step counter to check against global counter
  rewards, steps = [], []  # Rewards and steps for plotting
  l = str(len(str(args.T_max)))  # Max num. of digits for logging steps
  done = True  # Start new episode

  accumulators = {'100k': [], '500k': []}

  while T.value() <= args.T_max:
    if can_test:
      t_start = T.value()  # Reset counter

      metrics_acc = None
      if args.eval_model == 100:
          metrics_acc = accumulators['100k']
      if args.eval_model == 500:
          metrics_acc = accumulators['500k']

      # Evaluate over several episodes and average results
      avg_rewards, avg_episode_lengths = [], []
      for _ in range(args.evaluation_episodes):
        while True:
          # Reset or pass on hidden state
          if done:
            # Sync with shared model every episode
            model.load_state_dict(shared_model.state_dict())
            hx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
            cx = Variable(torch.zeros(1, args.hidden_size), volatile=True)
            # Reset environment and done flag
            state = env.reset()
            action, reward, done, episode_length = 0, 0, False, 0
            reward_sum = 0

          # Optionally render validation states
          if args.render:
            env.render()

          # Calculate policy
          input = extend_input(state, action_to_one_hot(action, ACTION_SIZE), reward, episode_length)
          policy, _, _, (hx, cx) = model(Variable(input, volatile=True), (hx.detach(), cx.detach()))  # Break graph for memory efficiency

          # Choose action greedily
          action = policy.max(1)[1].data[0, 0]

          # Step
          state, reward, done, _ = env.step(action)
          reward_sum += reward
          done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
          episode_length += 1  # Increase episode counter

          if metrics_acc is not None:
              metrics_acc.append(reward)

          # Log and reset statistics at the end of every episode
          if done:
            avg_rewards.append(reward_sum)
            avg_episode_lengths.append(episode_length)
            break

      print(('[{}] Step: {:<' + l + '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            t_start,
            sum(avg_rewards) / args.evaluation_episodes,
            sum(avg_episode_lengths) / args.evaluation_episodes))
      rewards.append(avg_rewards)  # Keep all evaluations
      steps.append(t_start)
      plot_line(steps, rewards, 'rewards.html', 'Average Reward')  # Plot rewards
      torch.save(model.state_dict(), 'checkpoints/' + str(t_start) + '.pth')  # Checkpoint model params
      can_test = False  # Finish testing
      if args.evaluate:
        return
    else:
      if T.value() - t_start >= args.evaluation_interval:
        can_test = True

    time.sleep(0.001)  # Check if available to test every millisecond

  env.close()

  # if save
  leaderboard_save(accumulators,
    'Baseline_Experiment_'+ str(args.eval_model),
    './baseline'+str(args.eval_model)+'.json')
