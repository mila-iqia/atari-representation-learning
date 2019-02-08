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
import wandb

from dim import MIEstimator
from env__util import CoinrunSubprocess, get_env_fun


def preprocessing():
    args = get_args()
    wandb.init(project="dim-rl", tags=['Baseline'])
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
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, recurrent_hidden_states, actor_features = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                if done[i] != 1:
                    episodes[i][-1].append(obs[i])
                else:
                    episodes[i].append([obs[i]])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        mi_estimator.maximize_mi(episodes)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = VecPyTorch(VecMonitor(CoinrunSubprocess(num_processes=args.num_processes, test=True), filename='monitor.csv'),
                       device=device)
            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            done_masks = [False] * args.num_processes
            while len(eval_episode_rewards) < args.num_processes:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states, actor_features = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for i, info in enumerate(infos):
                    if 'episode' in info.keys() and not done_masks[i]:
                        eval_episode_rewards.append(info['episode']['r'])
                    if done[i]:
                        done_masks[i] = True

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))
            writer.add_scalar('data/mean_eval_episode_rewards', np.mean(eval_episode_rewards), total_num_steps)
            wandb.log({'mean_eval_episode_rewards': np.mean(eval_episode_rewards)}, step=total_num_steps)

    writer.close()


if __name__ == "__main__":
    main()
