import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from src.envs import make_vec_envs
from src.utils import get_argparser, visualize_activation_maps
from src.encoders import NatureCNN, ImpalaCNN
from src.appo import AppoTrainer
from src.cpc import CPCTrainer

import wandb


def main():
    parser = get_argparser()
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, num_frame_stack=args.num_frame_stack)
    encoder = NatureCNN(envs.observation_space.shape[0])
    encoder.to(device)
    torch.set_num_threads(1)

    wandb.init(project="curl-atari", entity="curl-atari", tags=['pretraining-only'])
    config = {
        'encoder': encoder.__class__.__name__,
        'obs_space': str(envs.observation_space.shape),
        'optimizer': 'Adam',
    }
    config.update(vars(args))
    wandb.config.update(config)

    if args.method == 'appo':
        trainer = AppoTrainer(encoder, config, device=device, wandb=wandb)
    if args.method == 'cpc':
        trainer = CPCTrainer(encoder, config, device=device, wandb=wandb)

    obs = envs.reset()
    episode_rewards = deque(maxlen=10)
    start = time.time()
    print('-------Collecting samples----------')
    episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
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
            else:
                episodes[i].append([obs[i]])

    # Put episode frames on the GPU.
    for p in range(args.num_processes):
        for e in range(len(episodes[p])):
            episodes[p][e] = torch.stack(episodes[p][e])

    # Convert to 1d list from 2d list
    episodes = list(chain.from_iterable(episodes))
    episodes = [x for x in episodes if len(x) > 10]

    trainer.train(episodes)
    # TODO: Better data selection. This failed for a run on Pong.
    # frames = episodes[200][:60, :, :, :]
    # visualize_activation_maps(encoder, frames, wandb)


if __name__ == "__main__":
    main()
