from .label_preprocess import remove_duplicates, remove_low_entropy_labels
from collections import deque
from itertools import chain
import numpy as np
import torch
import time
import os
from .envs import make_vec_envs
from .utils import download_run
try:
    import wandb
except:
    pass


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


def get_random_agent_rollouts(env_name, steps, seed=42, num_processes=1, num_frame_stack=1, downsample=False, color=False):
    envs = make_vec_envs(env_name, seed,  num_processes, num_frame_stack, downsample, color)
    envs.reset();
    episode_rewards = deque(maxlen=10)
    print('-------Collecting samples----------')
    episodes = [[[]] for _ in range(num_processes)]  # (n_processes * n_episodes * episode_len)
    episode_labels = [[[]] for _ in range(num_processes)]
    for step in range(steps // num_processes):
        # Take action using a random policy
        action = torch.tensor(
            np.array([np.random.randint(1, envs.action_space.n) for _ in range(num_processes)])) \
            .unsqueeze(dim=1)
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

    # Convert to 2d list from 3d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 2d list from 3d list
    episode_labels = list(chain.from_iterable(episode_labels))
    envs.close()
    return episodes, episode_labels


def get_ppo_rollouts(env_name, steps, seed=42, num_processes=1,
                     num_frame_stack=1, downsample=False, color=False, checkpoint_index=-1):
    checkpoint_step = checkpointed_steps_full_sorted[checkpoint_index]
    filepath = download_run(env_name, checkpoint_step)
    while not os.path.exists(filepath):
        time.sleep(5)

    envs = make_vec_envs(env_name, seed,  num_processes, num_frame_stack, downsample, color)

    actor_critic, ob_rms = torch.load(filepath, map_location=lambda storage, loc: storage)

    episodes = [[[]] for _ in range(num_processes)]  # (n_processes * n_episodes * episode_len)
    episode_labels = [[[]] for _ in range(num_processes)]
    episode_rewards = deque(maxlen=10)

    masks = torch.zeros(1, 1)
    obs = envs.reset()
    entropies = []
    for step in range(steps // num_processes):
        # Take action using the PPO policy
        with torch.no_grad():
            _, action, _, _, actor_features, dist_entropy = actor_critic.act(obs, None, masks, deterministic=False)
        action = torch.tensor([envs.action_space.sample() if np.random.uniform(0, 1) < 0.2 else action[i]
                               for i in range(num_processes)]).unsqueeze(dim=1)
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

    # Convert to 2d list from 3d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 2d list from 3d list
    episode_labels = list(chain.from_iterable(episode_labels))
    mean_entropy = torch.stack(entropies).mean()
    mean_episode_reward = np.mean(episode_rewards)
    try:
        wandb.log({'action_entropy': mean_entropy, 'mean_reward': mean_episode_reward})
    except:
        pass

    return episodes, episode_labels


def get_episodes(env_name,
                 steps,
                 seed=42,
                 num_processes=1,
                 num_frame_stack=1,
                 downsample=False,
                 color=False,
                 entropy_threshold=0.6,
                 collect_mode="random_agent",
                 train_mode="probe",
                 checkpoint_index=-1,
                 min_episode_length=64):

    if collect_mode == "random_agent":
        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_random_agent_rollouts(env_name=env_name,
                                                             steps=steps,
                                                             seed=seed,
                                                             num_processes=num_processes,
                                                             num_frame_stack=num_frame_stack,
                                                             downsample=downsample, color=color)

    elif collect_mode == "pretrained_ppo":
        import wandb
        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_ppo_rollouts(env_name=env_name,
                                                   steps=steps,
                                                   seed=seed,
                                                   num_processes=num_processes,
                                                   num_frame_stack=num_frame_stack,
                                                   downsample=downsample,
                                                   color=color,
                                                   checkpoint_index=checkpoint_index)


    else:
        assert False, "Collect mode {} not recognized".format(collect_mode)

    ep_inds = [i for i in range(len(episodes)) if len(episodes[i]) > min_episode_length]
    episodes = [episodes[i] for i in ep_inds]
    episode_labels = [episode_labels[i] for i in ep_inds]
    episode_labels, entropy_dict = remove_low_entropy_labels(episode_labels, entropy_threshold=entropy_threshold)

    try:
        wandb.log(entropy_dict)
    except:
        pass

    inds = np.arange(len(episodes))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(inds)

    if train_mode == "train_encoder":
        assert len(inds) > 1, "Not enough episodes to split into train and val. You must specify enough steps to get at least two episodes"
        split_ind = int(0.8 * len(inds))
        tr_eps, val_eps = episodes[:split_ind], episodes[split_ind:]
        return tr_eps, val_eps

    if train_mode == "probe":
        val_split_ind, te_split_ind = int(0.7 * len(inds)), int(0.8 * len(inds))
        assert val_split_ind > 0 and te_split_ind > val_split_ind,\
            "Not enough episodes to split into train, val and test. You must specify more steps"
        tr_eps, val_eps, test_eps = episodes[:val_split_ind], episodes[val_split_ind:te_split_ind], episodes[
                                                                                                    te_split_ind:]
        tr_labels, val_labels, test_labels = episode_labels[:val_split_ind], \
                                             episode_labels[val_split_ind:te_split_ind], episode_labels[te_split_ind:]
        test_eps, test_labels = remove_duplicates(tr_eps, val_eps, test_eps, test_labels)
        test_ep_inds = [i for i in range(len(test_eps)) if len(test_eps[i]) > 1]
        test_eps = [test_eps[i] for i in test_ep_inds]
        test_labels = [test_labels[i] for i in test_ep_inds]
        return tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels

    if train_mode == "dry_run":
        return episodes, episode_labels




