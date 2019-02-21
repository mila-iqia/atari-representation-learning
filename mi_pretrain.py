import random
from itertools import chain
import time
from collections import deque

import numpy as np
import torch

from baselines.common.vec_env.vec_monitor import VecMonitor

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.envs import VecPyTorch
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from dim import MIEstimator
from encoders import ImpalaCNN
from env__util import CoinrunSubprocess
from utils import preprocess, visualize_activation_maps


def main():
    args, writer, num_updates, eval_log_dir = preprocess()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = VecPyTorch(
        VecMonitor(CoinrunSubprocess(num_processes=args.num_processes, test=False), filename='monitor.csv'),
        device=device)

    encoder = ImpalaCNN(3).to(device)
    mi_estimator = MIEstimator(encoder=encoder, device=device)
    obs = envs.reset()

    episode_rewards = deque(maxlen=10)
    # episodes will be of the shape (num_processes * episodes_in_process * episode_length)
    start = time.time()
    for j in range(num_updates):
        episodes = [[[]] for _ in range(args.num_processes)]
        for step in range(args.num_steps):
            # Sample actions
            action = torch.IntTensor([np.random.choice([1, 3, 4]) for _ in range(args.num_processes)]).unsqueeze(dim=1)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                if done[i] != 1:
                    episodes[i][-1].append(obs[i])
                else:
                    episodes[i].append([obs[i]])

        mi_loss, accuracy = mi_estimator.maximize_mi(episodes)
        # Sample 20 random frames
        frames = torch.stack(random.sample(list(chain.from_iterable(list(chain.from_iterable(episodes)))), 20))
        visualize_activation_maps(encoder, frames)

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
                           np.max(episode_rewards))
            )
            print("MI Loss:{}, Disc Accuracy: {}".format(mi_loss, accuracy))
            writer.add_scalar('data/mi_loss', mi_loss, j)

    writer.close()


if __name__ == "__main__":
    main()
