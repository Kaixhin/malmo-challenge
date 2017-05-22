# -*- coding: utf-8 -*-
import math
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from pc_environment import Env
from pc_model import ActorCritic
from pc_utils import ACTION_SIZE, action_to_one_hot, extend_input


# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return k - 1


# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr


# Updates networks
def _update_networks(args, T, model, shared_model, loss, optimiser):
  # Zero shared and local grads
  optimiser.zero_grad()
  # Calculate gradients (not losses defined as negatives of normal update rules for gradient descent)
  loss.backward()
  # Gradient L1 norm clipping
  nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm, 1)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if args.lr_decay:
    # Linearly decay learning rate
    _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))


# Trains model
def _train(args, T, model, shared_model, optimiser, target_class, pred_class):
  class_loss = 0

  # Step backwards from the last state
  t = len(pred_class)
  for i in reversed(range(t)):
    # Train classification loss
    class_loss += F.binary_cross_entropy(pred_class[i], target_class)

  # Optionally normalise loss by number of time steps
  if not args.no_time_normalisation:
    class_loss /= t
  # Update networks
  _update_networks(args, T, model, shared_model, class_loss, optimiser)


# Acts and trains model
def train(rank, args, T, shared_model, shared_average_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = Env(rank)
  model = ActorCritic(args.hidden_size)
  model.train()

  t = 1  # Thread step counter
  done = True  # Start new episode

  while T.value() <= args.T_max:
    # On-policy episode loop
    while True:
      # Sync with shared model at least every t_max steps
      model.load_state_dict(shared_model.state_dict())
      # Get starting timestep
      t_start = t

      # Reset or pass on hidden state
      if done:
        hx, avg_hx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
        cx, avg_cx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
        # Reset environment and done flag
        symbols, state = env.reset()
        action, reward, done, episode_length = 0, 0, False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      classes = []
      
      while not done and t - t_start < args.t_max:
        # Get label from the environment
        cls_id = env.get_class_label()
        
        # Calculate policy and values
        input = extend_input(state, action_to_one_hot(action, ACTION_SIZE), reward, episode_length)
        action, meta_policy, (hx, cx) = model.act(symbols, Variable(input), reward, done, (hx, cx), is_training=True)

        # Step
        symbols, state, reward, done, _ = env.step(action)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        episode_length += 1  # Increase episode counter

        # Save outputs for online training
        classes.append(meta_policy)

        # Increment counters
        t += 1
        T.increment()

      # Train the network on-policy
      _train(args, T, model, shared_model, optimiser, Variable(torch.Tensor([[cls_id]])), classes)

  env.close()
