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
from dimrl.actor_critic import CNNBase, LatentPolicy
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
    torch.set_num_threads(1)

    actor_critic = LatentPolicy(envs.action_space, encoder=encoder)
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    wandb.init(project="rl-representation-learning", tags=['ppo-after-pretraining'])
    config = {
        'pretraining_steps': args.pretraining_steps,
        'env_name': args.env_name,
        'mode': 'pcl',
        'encoder': encoder.__class__.__name__,
        'obs_space': str(envs.observation_space.shape),
        'epochs': args.contrastive_epochs,
        'lr': args.contrastive_lr,
        'mini_batch_size': args.contrastive_bs,
        'optimizer': 'Adam',
        'policy_training_steps': args.num_env_steps
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

    # Send episodes to GPU
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

    # Visualize activation maps
    episodes = list(chain.from_iterable(episodes))
    frames = episodes[200][:60, :, :, :]
    visualize_activation_maps(encoder, frames, wandb)

    # Use GPUs if available for envs now
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep, device, False)
    start = time.time()
    episode_rewards = deque(maxlen=10)
    obs = envs.reset()
    # Collect samples for training policy
    print('-------Starting RL Training----------')
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        # Update policy
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
            wandb.log({'Mean Rewards': np.mean(episode_rewards)}, step=total_num_steps)

    # writer.close()


if __name__ == "__main__":
    main()
