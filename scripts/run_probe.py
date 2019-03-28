from src.probe import ProbeTrainer
import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from src.envs import make_vec_envs
from src.utils import get_argparser, visualize_activation_maps
from src.encoders import NatureCNN, ImpalaCNN
from src.appo import AppoTrainer

import wandb
import sys
def main():
    parser = get_argparser()
    parser.set_defaults(env_name="PitfallNoFrameskip-v4")
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes)
    encoder = NatureCNN(envs.observation_space.shape[0])
    encoder.to(device)
    torch.set_num_threads(1)

    wandb.init(project="curl-atari", entity="curl-atari", tags=['probe-only'])
    config = {
        'pretraining_steps': args.pretraining_steps,
        'env_name': args.env_name,
        'method': args.method,
        'encoder': encoder.__class__.__name__,
        'obs_space': str(envs.observation_space.shape),
        'epochs': args.epochs,
        'lr': args.lr,
        'mini_batch_size': args.batch_size,
        'optimizer': 'Adam',
    }
    wandb.config.update(config)

    obs = envs.reset()
    episode_rewards = deque(maxlen=10)
    start = time.time()
    print('-------Collecting samples----------')
    episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
    episode_labels = [[[]] for _ in range(args.num_processes)]
    for step in range(args.pretraining_steps // args.num_processes):
        # Take action using a random policy
        action = torch.tensor(np.array([np.random.randint(1, envs.action_space.n) for _ in range(args.num_processes)])) \
            .unsqueeze(dim=1).to(device)
        obs, reward, done, infos = envs.step(action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])


            if done[i] != 1:
                episodes[i][-1].append(obs[i])
                if "labels" in info.keys():
                    episode_labels[i][-1].append(info["labels"])
            else:
                episodes[i].append([obs[i]])
                if "labels" in info.keys():
                    episode_labels[i].append([info["labels"]])

    # Put episode frames on the GPU.
    for p in range(args.num_processes):
        for e in range(len(episodes[p])):
            episodes[p][e] = torch.stack(episodes[p][e]).to(device)

    # Convert to 1d list from 2d list
    episodes = list(chain.from_iterable(episodes))
    episodes = [x for x in episodes if len(x) > 10]

    # Convert to 1d list from 2d list
    episode_labels = list(chain.from_iterable(episode_labels))
    episode_labels = [x for x in episode_labels if len(x) > 10]

    trainer = ProbeTrainer(encoder, wandb, info_dict=info["num_classes"])

    trainer.train(episodes, episode_labels)
    
if __name__ == "__main__":
    main()
