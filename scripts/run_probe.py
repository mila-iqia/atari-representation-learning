from src.probe import ProbeTrainer
import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from src.envs import make_vec_envs
from src.utils import get_argparser, visualize_activation_maps
from src.encoders import NatureCNN, ImpalaCNN
from src.appo import AppoTrainer

import wandb
import sys


def main():
    parser = get_argparser()
    parser.set_defaults(env_name="MontezumaRevengeNoFrameskip-v4")
    parser.add_argument("--weights-path", type=str, default="None")
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, num_frame_stack=args.num_frame_stack)
    if args.encoder_type == "Nature":
        encoder = NatureCNN(envs.observation_space.shape[0])
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(envs.observation_space.shape[0])

    if args.method == "random_cnn":
        print("Random CNN, so not loading in encoder weights!")
    elif args.method == "supervised":
        print("Fully supervised, so starting from random encoder weights!")
    else:
        if args.weights_path == "None":
            print("Probing without loading in encoder weights! Are sure you want to do that??")
        else:
            print("Print loading in encoder weights from probe of type {} from the following path: {}".format(args.method, args.weight_path))
            encoder.load_state_dict(torch.load(args.weights_path))
            encoder.eval()

    # encoder.to(device)
    torch.set_num_threads(1)

    wandb.init(project="curl-atari", entity="curl-atari", tags=['probe-only'])
    config = {
        'encoder_type': encoder.__class__.__name__,
        'obs_space': str(envs.observation_space.shape),
        'optimizer': 'Adam',
        'probe_lr': args.lr
    }
    config.update(vars(args))
    wandb.config.update(config)

    def collect_episodes(num_steps, val=False):
        obs = envs.reset()
        episode_rewards = deque(maxlen=10)
        start = time.time()
        print('-------Collecting samples----------')
        episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
        episode_labels = [[[]] for _ in range(args.num_processes)]
        for step in range(num_steps // args.num_processes):
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
        episodes = [x for x in episodes if len(x) > args.batch_size]

        # Convert to 1d list from 2d list
        episode_labels = list(chain.from_iterable(episode_labels))
        episode_labels = [x for x in episode_labels if len(x) > args.batch_size]

        if val:
            inds = range(len(episodes))
            split_ind = int(0.8 * len(inds))

            tr_eps, val_eps = episodes[:split_ind], episodes[split_ind:]
            tr_labels, val_labels = episode_labels[:split_ind], episode_labels[split_ind:]

            return tr_eps, val_eps, tr_labels, val_labels, info

        else:
            return episodes, episode_labels, info

    tr_eps, val_eps, tr_labels, val_labels, info = collect_episodes(args.probe_train_steps, val=True)
    trainer = ProbeTrainer(encoder, wandb, info_dict=info["num_classes"], epochs=args.epochs,
                           lr=args.lr, batch_size=args.batch_size, device=device, patience=args.patience)

    trainer.train(tr_eps, val_eps, tr_labels, val_labels)

    te_episodes, te_episode_labels, _ = collect_episodes(args.probe_test_steps)

    trainer.evaluate(te_episodes, te_episode_labels)



if __name__ == "__main__":
    main()
