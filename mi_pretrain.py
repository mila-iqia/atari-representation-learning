import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_monitor import VecMonitor

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs, VecPyTorch
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule

from tensorboardX import SummaryWriter
#import wandb

from dim import MIEstimator
from env__util import CoinrunSubprocess, get_env_fun


def preprocessing():
    args = get_args()
    #wandb.init(project="dim-rl", tags=['Baseline'])
    at_config = {}
    # wandb.config.update(at_config)
    writer = SummaryWriter(comment='runs')

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    eval_log_dir = args.log_dir + "_eval"

    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    return args, writer, num_updates, eval_log_dir

def main():
    args, writer, num_updates, eval_log_dir = preprocessing()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = VecPyTorch(VecMonitor(CoinrunSubprocess(num_processes=args.num_processes, test=False), filename='monitor.csv'),
                      device=device)
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    mi_estimator = MIEstimator(encoder=actor_critic.base.main, device=device)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    # episodes will be of the shape (num_processes * episodes_in_process * episode_length)
    episodes = [[[]] for _ in range(args.num_processes)]
    start = time.time()
    for j in range(num_updates):
        episodes = [[[]] for _ in range(args.num_processes)]
        for step in range(args.num_steps):
            # Sample actions
            action = torch.IntTensor([envs.action_space.sample() for _ in range(args.num_processes)]).unsqueeze(dim=1)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                if done[i] != 1:
                    episodes[i][-1].append(obs[i])
                else:
                    episodes[i].append([obs[i]])

        mi_loss = mi_estimator.maximize_mi(episodes)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, Epoch Loss: {}\n".format(j, mi_loss))
            writer.add_scalar('data/mi_loss', mi_loss, j)

    writer.close()


if __name__ == "__main__":
    main()
