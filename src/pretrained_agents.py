import argparse
from collections import deque
from itertools import chain
import numpy as np
import torch
import wandb
import time
import os

from src.envs import make_vec_envs
from src.utils import get_argparser

checkpointed_steps = [10753536, 12289536, 13825536, 1536, 15361536, 1537536, 16897536, 18433536, 19969536, 21505536,
                      23041536, 24577536, 26113536, 27649536, 29185536, 30721536, 3073536, 32257536, 33793536, 35329536,
                      36865536, 38401536, 39937536, 41473536, 43009536, 44545536, 46081536, 4609536, 47617536, 49153536,
                      49999872, 6145536, 7681536, 9217536]

checkpointed_steps_sorted = [1536, 1537536, 3073536, 4609536, 6145536, 7681536, 9217536, 10753536, 12289536, 13825536,
                             15361536, 16897536, 18433536, 19969536, 21505536, 23041536, 24577536, 26113536, 27649536,
                             29185536, 30721536, 32257536, 33793536, 35329536, 36865536, 38401536, 39937536, 41473536,
                             43009536, 44545536, 46081536, 47617536, 49153536, 49999872]


def download_run(args, checkpoint_step):
    api = wandb.Api()
    runs = list(api.runs("curl-atari/pretrained-rl-agents", {"state": "finished",
                                                             "config.env_name": args.env_name}))
    run = runs[0]
    mean_reward = run.summary['Mean Rewards']
    index = checkpointed_steps.index(int(checkpoint_step))
    files = [file for file in run.files() if file.name.startswith(args.env_name)]
    files[index].download(root='../trained_models/', replace=True)
    return '../trained_models/' + args.env_name + '_' + str(checkpoint_step) + ".pt", mean_reward


def get_ppo_rollouts(args, checkpoint_step,  use_representations_instead_of_frames=False):
    filepath, mean_reward = download_run(args, checkpoint_step)
    while not os.path.exists(filepath):
        time.sleep(5)

    args.no_downsample = False
    args.num_frame_stack = 4
    envs = make_vec_envs(args, args.num_processes)

    actor_critic, ob_rms = \
        torch.load(filepath, map_location=lambda storage, loc: storage)


    episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
    episode_labels = [[[]] for _ in range(args.num_processes)]
    episode_rewards = deque(maxlen=10)
    episode_features = [[[]] for _ in range(args.num_processes)] 

    step = 0
    masks = torch.zeros(1, 1)
    obs = envs.reset()
    for step in range(args.probe_steps // args.num_processes):
        # Take action using a random policy
        _, action, _, _, actor_features = actor_critic.act(obs, None, masks, deterministic=False)
        action = torch.tensor(
                    np.array([np.random.randint(1, envs.action_space.n) for _ in range(args.num_processes)])) \
                    .unsqueeze(dim=1)
        obs, reward, done, infos = envs.step(action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            if done[i] != 1:
                episodes[i][-1].append(obs[i].clone())
                episode_features[i][-1].append(actor_features[i].clone())
                if "labels" in info.keys():
                    episode_labels[i][-1].append(info["labels"])
            else:
                episodes[i].append([obs[i].clone()])
                episode_features[i].append([actor_features[i].clone()])
                if "labels" in info.keys():
                    episode_labels[i].append([info["labels"]])

    # Put episode frames on the GPU.
    for p in range(args.num_processes):
        for e in range(len(episodes[p])):
            episodes[p][e] = torch.stack(episodes[p][e])
            episode_features[p][e] = torch.stack(episode_features[p][e])
            

    # Convert to 1d list from 2d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 1d list from 2d list
    episode_labels = list(chain.from_iterable(episode_labels))
    episode_features = list(chain.from_iterable(episode_features))
    if use_representations_instead_of_frames:
        return episode_features, episode_labels, mean_reward
    else:
        return episodes, episode_labels, mean_reward


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    episodes, episode_labels, mean_reward = get_ppo_rollouts(args, 1537536)