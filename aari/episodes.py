import time
from collections import deque
from itertools import chain
from aari.envs import make_vec_envs
import numpy as np
import torch
from src.pretrained_agents import get_ppo_rollouts, checkpointed_steps_full_sorted, get_ppo_representations
import wandb
from aari.label_preprocess import remove_duplicates, remove_low_entropy_labels


def get_random_agent_episodes(args, device, steps):
    envs = make_vec_envs(args, args.num_processes)
    obs = envs.reset()
    episode_rewards = deque(maxlen=10)
    print('-------Collecting samples----------')
    episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
    episode_labels = [[[]] for _ in range(args.num_processes)]
    for step in range(steps // args.num_processes):
        # Take action using a random policy
        action = torch.tensor(
            np.array([np.random.randint(1, envs.action_space.n) for _ in range(args.num_processes)])) \
            .unsqueeze(dim=1).to(device)
        obs, reward, done, infos = envs.step(action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            if done[i] != 1:
                episodes[i][-1].append(obs[i].clone())
                if "labels" in info.keys():
                    episode_labels[i][-1].append(info["labels"])
            else:
                episodes[i].append([obs[i].clone()])
                if "labels" in info.keys():
                    episode_labels[i].append([info["labels"]])

    # Convert to 2d list from 3d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 2d list from 3d list
    episode_labels = list(chain.from_iterable(episode_labels))
    envs.close()
    return episodes, episode_labels


def get_pretrained_rl_episodes(args, steps):
    checkpoint = checkpointed_steps_full_sorted[args.checkpoint_index]
    episodes, episode_labels, mean_reward, mean_action_entropy = get_ppo_rollouts(args, steps, checkpoint)
    wandb.log({'action_entropy': mean_action_entropy, 'mean_reward': mean_reward})
    return episodes, episode_labels


def get_pretrained_rl_representations(args):
    checkpoint = checkpointed_steps_full_sorted[args.checkpoint_index]
    episodes, episode_labels, mean_reward = get_ppo_representations(args, checkpoint)
    wandb.log({"reward": mean_reward, "checkpoint": checkpoint})
    return episodes, episode_labels


def get_episodes(args, device, collect_mode="random_agent", train_mode="probe", seed=None):
    seed = seed if seed else args.seed
    steps = args.probe_steps if train_mode == "probe" else args.pretraining_steps

    if collect_mode == "random_agent":
        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_random_agent_episodes(args, device, steps)

    elif collect_mode == "pretrained_ppo":
        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_pretrained_rl_episodes(args, steps)

    elif collect_mode == "pretrained_representations":
        # "episodes" are vectors from output of last layer of PPO agent
        episodes, episode_labels = get_pretrained_rl_representations(args, steps)

    ep_inds = [i for i in range(len(episodes)) if len(episodes[i]) > args.batch_size]
    episodes = [episodes[i] for i in ep_inds]
    inds = np.arange(len(episodes))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(inds)

    if train_mode == "train_encoder":
        split_ind = int(0.8 * len(inds))
        tr_eps, val_eps = episodes[:split_ind], episodes[split_ind:]
        return tr_eps, val_eps

    if train_mode == "probe":
        episode_labels = [episode_labels[i] for i in ep_inds]
        episode_labels = remove_low_entropy_labels(episode_labels, entropy_threshold=args.entropy_threshold)

        val_split_ind, te_split_ind = int(0.7 * len(inds)), int(0.8 * len(inds))
        tr_eps, val_eps, test_eps = episodes[:val_split_ind], episodes[val_split_ind:te_split_ind], episodes[
                                                                                                    te_split_ind:]
        tr_labels, val_labels, test_labels = episode_labels[:val_split_ind], episode_labels[
                                                                             val_split_ind:te_split_ind], episode_labels[
                                                                                                          te_split_ind:]
        test_eps, test_labels = remove_duplicates(tr_eps, val_eps, test_eps, test_labels)
        test_ep_inds = [i for i in range(len(test_eps)) if len(test_eps[i]) > 1]
        test_eps = [test_eps[i] for i in test_ep_inds]
        test_labels = [test_labels[i] for i in test_ep_inds]

        return tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels
