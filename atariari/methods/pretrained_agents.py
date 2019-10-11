from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs as mve
from collections import deque
from itertools import chain
import numpy as np
import torch
import wandb
import time
import os

from atariari.benchmark.envs import make_vec_envs
from atariari.benchmark.episodes import checkpointed_steps_full_sorted
from .utils import get_argparser
from atariari.benchmark.utils import download_run

# elif collect_mode == "pretrained_representations":
# # "episodes" are vectors from output of last layer of PPO agent
# episodes, episode_labels = get_pretrained_rl_representations(args, steps)

def get_pretrained_rl_representations(args, steps):
    checkpoint = checkpointed_steps_full_sorted[args.checkpoint_index]
    episodes, episode_labels, mean_reward = get_ppo_representations(args, steps, checkpoint)
    wandb.log({"reward": mean_reward, "checkpoint": checkpoint})
    return episodes, episode_labels


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
            _, action, _, eval_recurrent_hidden_states, _, _ = actor_critic.act(
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


def get_ppo_representations(args, steps, checkpoint_step):
    # Gives PPO represnetations over data collected by a random agent
    filepath = download_run(args, checkpoint_step)
    while not os.path.exists(filepath):
        time.sleep(5)

    envs = make_vec_envs(args, args.num_processes)

    actor_critic, ob_rms = \
        torch.load(filepath, map_location=lambda storage, loc: storage)
    mean_reward = evaluate(actor_critic,
                           env_name=args.env_name,
                           seed=args.seed,
                           num_processes=args.num_processes,
                           eval_log_dir="./tmp", device="cpu", num_evals=args.num_rew_evals)
    print(mean_reward)
    episode_labels = [[[]] for _ in range(args.num_processes)]
    episode_rewards = deque(maxlen=10)
    episode_features = [[[]] for _ in range(args.num_processes)]


    masks = torch.zeros(1, 1)
    obs = envs.reset()
    for step in range(steps // args.num_processes):
        # Take action using a random policy
        if args.probe_collect_mode == 'random_agent':
            action = torch.tensor(
                np.array([np.random.randint(1, envs.action_space.n) for _ in range(args.num_processes)])) \
                .unsqueeze(dim=1)
        else:
            with torch.no_grad():
                _, action, _, _, actor_features, _ = actor_critic.act(obs, None, masks, deterministic=False)
            action = torch.tensor([envs.action_space.sample() if np.random.uniform(0, 1) < 0.2 else action[i]
                                   for i in range(args.num_processes)]).unsqueeze(dim=1)

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

    # Convert to 2d list from 3d list
    episode_labels = list(chain.from_iterable(episode_labels))
    episode_features = list(chain.from_iterable(episode_features))
    return episode_features, episode_labels, mean_reward


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    args.env_name = 'SeaquestNoFrameskip-v4'
    args.probe_steps = 2000
    episode_features, episode_labels, mean_reward = get_ppo_representations(args, 10753536)
