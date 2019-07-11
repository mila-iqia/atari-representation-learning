import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from aari.envs import make_vec_envs
from src.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from src.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from src.pretrained_agents import get_ppo_rollouts, checkpointed_steps_full_sorted
from src.spatio_temporal import SpatioTemporalTrainer
from src.utils import get_argparser, visualize_activation_maps
from src.encoders import NatureCNN, ImpalaCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
import wandb
import sys


def train_encoder(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args, args.num_processes)

    if args.encoder_type == "Nature":
        encoder = NatureCNN(envs.observation_space.shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(envs.observation_space.shape[0], args)
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = envs.observation_space.shape  # weird hack
    if args.method == 'cpc':
        trainer = CPCTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'spatial-appo':
        trainer = SpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'vae':
        trainer = VAETrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "naff":
        trainer = NaFFPredictorTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)


    if args.collect_mode == "random_agent":
        obs = envs.reset()
        episode_rewards = deque(maxlen=10)
        start = time.time()
        print('-------Collecting samples----------')
        episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
        for step in range(args.pretraining_steps // args.num_processes):
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
                else:
                    episodes[i].append([obs[i].clone()])

        episode_lens = []
        for i in range(args.num_processes):
            episode_lens += [len(episode) for episode in episodes[i]]
        print("Episode lengths: mean/std/min/max",
              np.mean(episode_lens), np.std(episode_lens),
              np.min(episode_lens), np.max(episode_lens))

        # Convert to 2d list from 3d list
        episodes = list(chain.from_iterable(episodes))
        episodes = [x for x in episodes if len(x) > args.batch_size]

    elif args.collect_mode == "pretrained_ppo":
        checkpoint = checkpointed_steps_full_sorted[args.checkpoint_index]
        episodes, episode_labels, mean_reward, mean_action_entropy = get_ppo_rollouts(args, args.pretraining_steps,
                                                                                      checkpoint)
        episodes = [x for x in episodes if len(x) > args.batch_size]

    inds = range(len(episodes))
    split_ind = int(0.8 * len(inds))

    tr_eps, val_eps = episodes[:split_ind], episodes[split_ind:]

    trainer.train(tr_eps, val_eps)
    envs.close()

    return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    wandb.init(project="curl-atari-2", entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    train_encoder(args)
