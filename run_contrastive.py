import random
import time
from collections import deque
from itertools import chain

import gym
import numpy as np
import torch

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.envs import make_vec_envs, VecPyTorch
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from utils import preprocess, save_model, evaluate_policy, visualize_activation_maps
from encoders import NatureCNN, ImpalaCNN
from contrastive import ContrastiveTrainer

from tensorboardX import SummaryWriter
import wandb


def main():
    wandb.init(project="rl-representation-learning", tags=['atari'])
    args, writer, num_updates, eval_log_dir = preprocess()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device, False)
    encoder = NatureCNN(3)
    encoder.to(device)
    trainer = ContrastiveTrainer(encoder)

    obs = envs.reset()
    episode_rewards = deque(maxlen=10)
    start = time.time()
    episodes = [[[]] for _ in range(args.num_processes)]
    for step in range(args.num_steps):
        # Observe reward and next obs
        obs, reward, done, infos = envs.step(torch.tensor([envs.action_space.sample() for _ in range(args.num_processes)]).unsqueeze(dim=1))
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
            if done[i] != 1:
                episodes[i][-1].append(obs[i])
            else:
                episodes[i].append([obs[i]])

    trainer.train(episodes)
    # Sample 20 random frames
    frames = torch.stack(random.sample(list(chain.from_iterable(list(chain.from_iterable(episodes)))), 20))
    visualize_activation_maps(encoder, frames, wandb)
    writer.close()


if __name__ == "__main__":
    main()
