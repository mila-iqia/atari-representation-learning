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
from src.atari_zoo import get_atari_zoo_episodes
import wandb
import sys


def main():
    parser = get_argparser()
#     sys.argv = []
#     parser.set_defaults(env_name="MontezumaRevengeNoFrameskip-v4")
    parser.add_argument("--weights-path", type=str, default="None")

    args = parser.parse_args()
#     args.probe_steps = 2000
#     args.collect_mode = "atari_zoo"
#     args.num_processes = 4
#     args.patience = 1
#     args.no_downsample = False
#     args.num_frame_stack = 4
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, num_frame_stack=args.num_frame_stack,
                         downsample=not args.no_downsample)
    if args.encoder_type == "Nature":
        encoder = NatureCNN(envs.observation_space.shape[0], downsample=not args.no_downsample)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(envs.observation_space.shape[0], downsample=not args.no_downsample)

    if args.method == "random_cnn":
        print("Random CNN, so not loading in encoder weights!")
    elif args.method == "supervised":
        print("Fully supervised, so starting from random encoder weights!")
    else:
        if args.weights_path == "None":
            sys.stderr.write("Probing without loading in encoder weights! Are sure you want to do that??")
        else:
            print("Print loading in encoder weights from probe of type {} from the following path: {}".format(args.method, args.weights_path))
            encoder.load_state_dict(torch.load(args.weights_path))
            encoder.eval()

    # encoder.to(device)
    torch.set_num_threads(1)
    tags=['probe-only']
    wandb.init(project="curl-atari", entity="curl-atari", tags=tags)
    config = {
        'encoder_type': encoder.__class__.__name__,
        'obs_space': str(envs.observation_space.shape),
        'optimizer': 'Adam',
        'probe_lr': args.lr
    }
    config.update(vars(args))
    wandb.config.update(config)


    if args.collect_mode == "random_agent":
        obs = envs.reset()
        episode_rewards = deque(maxlen=10)
        start = time.time()
        print('-------Collecting samples----------')
        episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
        episode_labels = [[[]] for _ in range(args.num_processes)]
        for step in range(args.probe_steps // args.num_processes):
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
        # Convert to 1d list from 2d list
        episode_labels = list(chain.from_iterable(episode_labels))    
    else:
        episodes, episode_labels = get_atari_zoo_episodes(args.env_name, tags=tags, num_frame_stack=args.num_frame_stack, downsample= not args.no_downsample)



    episodes = [torch.from_numpy(ep).permute(0,3,1,2).float() for ep in episodes]

    episodes = [x for x in episodes if len(x) > args.batch_size]
    episode_labels = [x for x in episode_labels if len(x) > args.batch_size]


    inds = range(len(episodes))
    val_split_ind, te_split_ind = int(0.6 * len(inds)), int(0.8 * len(inds))

    tr_eps, val_eps, test_eps = episodes[:val_split_ind], episodes[val_split_ind:te_split_ind], episodes[te_split_ind:]
    tr_labels, val_labels, test_labels = episode_labels[:val_split_ind], episode_labels[val_split_ind:te_split_ind], episode_labels[te_split_ind:]

    trainer = ProbeTrainer(encoder, wandb, epochs=args.epochs, sample_label=tr_labels[0][0],
                           lr=args.lr, batch_size=args.batch_size, device=device, patience=args.patience)

    trainer.train(tr_eps, val_eps, tr_labels, val_labels)
    trainer.evaluate(test_eps, test_labels)


if __name__ == "__main__":
    main()

