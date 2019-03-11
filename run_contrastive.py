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
    args, writer, num_updates, eval_log_dir = preprocess()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device, False)
    encoder = NatureCNN(envs.observation_space.shape[0])
    encoder.to(device)
    torch.set_num_threads(1)

    wandb.init(project="rl-representation-learning", tags=['pretraining-only'])
    config = {
        'pretraining_steps': args.num_steps,
        'env_name': args.env_name,
        'mode': 'pcl',
        'encoder': encoder.__class__.__name__,
        'obs_space': str(envs.observation_space.shape),
        'epochs': 50,
        'lr': 3e-4,
        'mini_batch_size': 64,
        'optimizer': 'Adam'
    }
    wandb.config.update(config)

    trainer = ContrastiveTrainer(encoder, mode=config['mode'], epochs=config['epochs'], lr=config['lr'],
                                 mini_batch_size=config['mini_batch_size'], device=device)

    obs = envs.reset()
    episode_rewards = deque(maxlen=10)
    start = time.time()
    print('-------Collecting samples----------')
    episodes = [[[]] for _ in range(args.num_processes)]
    for step in range(args.num_env_steps // args.num_processes):
        # Observe reward and next obs
        obs, reward, done, infos = envs.step(torch.tensor([envs.action_space.sample() for _ in range(args.num_processes)])
                                             .unsqueeze(dim=1))
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
            if done[i] != 1:
                episodes[i][-1].append(obs[i])
            else:
                episodes[i].append([obs[i]])
    end = time.time()
    print('Took {} seconds to collect samples'.format(end - start))
    print('-------Starting Contrastive Training----------')
    trainer.train(episodes, wandb)
    print('-------Contrastive Training Finished----------')
    end_training = time.time()
    print('Took {} seconds to train'.format(end_training - end))
    # Sample 20 random frames
    frames = torch.stack(random.sample(list(chain.from_iterable(list(chain.from_iterable(episodes)))), 20))
    visualize_activation_maps(encoder, frames, wandb)
    writer.close()


if __name__ == "__main__":
    main()
