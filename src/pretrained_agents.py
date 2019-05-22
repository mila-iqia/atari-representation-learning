import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs as mve


def evaluate(actor_critic, env_name, seed, num_processes, eval_log_dir,
             device, num_evals):
    eval_envs = mve(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, num_frame_stack=1)

    vec_norm = utils.get_vec_normalize(eval_envs)


    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_evals:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states, _,_ = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return np.mean(eval_episode_rewards)

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

checkpointed_steps_full = [10753536, 1076736, 11828736, 12903936, 13979136, 15054336, 1536, 16129536, 17204736,
                           18279936,
                           19355136, 20430336, 21505536, 2151936, 22580736, 23655936, 24731136, 25806336, 26881536,
                           27956736,
                           29031936, 30107136, 31182336, 32257536, 3227136, 33332736, 34407936, 35483136, 36558336,
                           37633536,
                           38708736, 39783936, 40859136, 41934336, 43009536, 4302336, 44084736, 45159936, 46235136,
                           47310336,
                           48385536, 49460736, 49999872, 5377536, 6452736, 7527936, 8603136, 9678336]

checkpointed_steps_full_sorted = [1536, 1076736, 2151936, 3227136, 4302336, 5377536, 6452736, 7527936, 8603136, 9678336,
                                  10753536, 11828736, 12903936, 13979136, 15054336, 16129536, 17204736, 18279936,
                                  19355136, 20430336, 21505536, 22580736, 23655936, 24731136, 25806336, 26881536,
                                  27956736, 29031936, 30107136, 31182336, 32257536, 33332736, 34407936, 35483136,
                                  36558336, 37633536, 38708736, 39783936, 40859136, 41934336, 43009536, 44084736,
                                  45159936, 46235136, 47310336, 48385536, 49460736, 49999872]


def download_run(args, checkpoint_step):
    api = wandb.Api()
    runs = list(api.runs("curl-atari/pretrained-rl-agents-2", {"state": "finished",
                                                               "config.env_name": args.env_name}))
    run = runs[0]
    filename = args.env_name + '_' + str(checkpoint_step) + '.pt'
    run.files(names=[filename])[0].download(root='./trained_models_full/', replace=True)
    print('Downloaded ' + filename)
    return './trained_models_full/' + filename


def get_ppo_rollouts(args, checkpoint_step):
    filepath = download_run(args, checkpoint_step)
    while not os.path.exists(filepath):
        time.sleep(5)

    envs = make_vec_envs(args, args.num_processes)

    actor_critic, ob_rms = \
        torch.load(filepath, map_location=lambda storage, loc: storage)

    episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
    episode_labels = [[[]] for _ in range(args.num_processes)]
    episode_rewards = deque(maxlen=10)

    step = 0
    masks = torch.zeros(1, 1)
    obs = envs.reset()
    entropies = []
    for step in range(args.probe_steps // args.num_processes):
        # Take action using a random policy
        obs, action, _, _, actor_features, dist_entropy = actor_critic.act(obs, None, masks, deterministic=False)
        entropies.append(dist_entropy.clone())
        obs, reward, done, infos = envs.step(action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            if done[i] != 1:
                episodes[i][-1].append(obs[i].clone())
                if "labels" in info.keys():
                    episode_labels[i][-1].append(info["labels"])
            else:
                episodes[i].append([obs[i].clone()])
                if "labels" in info.keys():
                    episode_labels[i].append([info["labels"]])

    # Put episode frames on the GPU.
    for p in range(args.num_processes):
        for e in range(len(episodes[p])):
            episodes[p][e] = torch.stack(episodes[p][e])

    # Convert to 1d list from 2d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 1d list from 2d list
    episode_labels = list(chain.from_iterable(episode_labels))
    mean_entropy = torch.stack(entropies).mean()
    return episodes, episode_labels, np.mean(episode_rewards), mean_entropy


def get_ppo_representations(args, checkpoint_step, rollout_checkpoint_step=None):
    # Gives PPO represnetations over data collected by a random agent
    filepath = download_run(args, checkpoint_step)
    while not os.path.exists(filepath):
        time.sleep(5)
    random_agent = True  
    if rollout_checkpoint_step:
        random_agent = False
        ro_filepath = download_run(args, rollout_checkpoint_step)
        while not os.path.exists(ro_filepath):
            time.sleep(5)
        ro_actor_critic, ob_rms = \
        torch.load(ro_filepath, map_location=lambda storage, loc: storage)
        

    # args.no_downsample = False
    # args.num_frame_stack = 4
    envs = make_vec_envs(args, args.num_processes)

    actor_critic, ob_rms = \
        torch.load(filepath, map_location=lambda storage, loc: storage)
    mean_reward = evaluate(actor_critic,
                           env_name=args.env_name,
                           seed=args.seed,
                           num_processes=args.num_processes,
                           eval_log_dir="./tmp",device="cpu",num_evals=args.num_rew_evals)
    print(mean_reward)
    episode_labels = [[[]] for _ in range(args.num_processes)]
    episode_rewards = deque(maxlen=10)
    episode_features = [[[]] for _ in range(args.num_processes)]

    step = 0
    masks = torch.zeros(1, 1)
    obs = envs.reset()
    for step in range(args.probe_steps // args.num_processes):
        # Take action using a random policy
        _, action, _, _, actor_features, _ = actor_critic.act(obs, None, masks, deterministic=False)
        
        if random_agent:
            action = torch.tensor(
                np.array([np.random.randint(1, envs.action_space.n) for _ in range(args.num_processes)])) \
                .unsqueeze(dim=1)
        else:
            _, action, _, _, actor_features, _ = ro_actor_critic.act(obs, None, masks, deterministic=False)
        
        obs, reward, done, infos = envs.step(action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            if done[i] != 1:
                episode_features[i][-1].append(actor_features[i].clone())
                if "labels" in info.keys():
                    episode_labels[i][-1].append(info["labels"])
            else:
                episode_features[i].append([actor_features[i].clone()])
                if "labels" in info.keys():
                    episode_labels[i].append([info["labels"]])

    # Put episode frames on the GPU.
    for p in range(args.num_processes):
        for e in range(len(episode_features[p])):
            episode_features[p][e] = torch.stack(episode_features[p][e])

    # Convert to 1d list from 2d list
    episode_labels = list(chain.from_iterable(episode_labels))
    episode_features = list(chain.from_iterable(episode_features))
    return episode_features, episode_labels, mean_reward


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    args.env_name = 'SeaquestNoFrameskip-v4'
    args.probe_steps = 2000
    episode_features, episode_labels, mean_reward = get_ppo_representations(args, 10753536)
