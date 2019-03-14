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
from dimrl.utils import preprocess, save_model, evaluate_policy, visualize_activation_maps
from dimrl.encoders import NatureCNN, ImpalaCNN
from dimrl.contrastive import ContrastiveTrainer

import wandb


def main():
    args, writer, num_updates, eval_log_dir = preprocess()
    device = torch.device("cuda:"+str(args.cuda_id) if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, torch.device('cpu'), False)
    encoder = NatureCNN(envs.observation_space.shape[0])
    encoder.to(device)
    torch.set_num_threads(1)

    wandb.init(project="rl-representation-learning", tags=['pretraining-only'])
    config = {
        'pretraining_steps': args.pretraining_steps,
        'env_name': args.env_name,
        'mode': args.contrastive_mode,
        'encoder': encoder.__class__.__name__,
        'obs_space': str(envs.observation_space.shape),
        'epochs': args.contrastive_epochs,
        'lr': args.contrastive_lr,
        'mini_batch_size': args.contrastive_bs,
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
    for step in range(args.pretraining_steps // args.num_processes):
        # Observe reward and next obs
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

    for p in range(args.num_processes):
        for e in range(len(episodes[p])):
            episodes[p][e] = torch.stack(episodes[p][e]).to(device)

    end = time.time()
    print('Took {} seconds to collect samples'.format(end - start))
    print('-------Starting Contrastive Training----------')
    trainer.train(episodes, wandb)
    print('-------Contrastive Training Finished----------')
    end_training = time.time()
    print('Took {} seconds to train'.format(end_training - end))
    episodes = list(chain.from_iterable(episodes))
    frames = episodes[200][:60, :, :, :]
    visualize_activation_maps(encoder, frames, wandb)
    # writer.close()


if __name__ == "__main__":
    main()
