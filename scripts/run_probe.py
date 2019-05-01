from scripts.run_contrastive import train_encoder
from src.probe import ProbeTrainer
import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import gym

from src.envs import make_vec_envs
from src.utils import get_argparser, visualize_activation_maps, appendabledict
from src.encoders import NatureCNN, ImpalaCNN
from src.appo import AppoTrainer
from src.atari_zoo import get_atari_zoo_episodes
import wandb
import sys


def main():
    parser = get_argparser()
    parser.add_argument("--weights-path", type=str, default="None")
    parser.add_argument("--train-encoder", action='store_true')
    parser.add_argument('--probe-lr', type=float, default=5e-2)
    parser.add_argument("--probe-collect-mode", type=str, choices=["random_agent", "atari_zoo"], default="random_agent")
    parser.add_argument('--zoo-algos', nargs='+', default=["a2c"])
    parser.add_argument('--zoo-tags', nargs='+', default=["10HR"])
    parser.add_argument('--num-runs', type=int, default=1)
    args = parser.parse_args()
    # dummy env
    env = make_vec_envs(args.env_name, args.seed, 1, num_frame_stack=args.num_frame_stack,
                        downsample=not args.no_downsample)
    wandb.config.update(vars(args))

    if args.train_encoder:
        assert (args.method in ['appo', 'spatial-appo', 'cpc'])
        print("Training encoder from scratch")
        encoder = train_encoder(args)
        encoder.probing = True
        encoder.eval()

    else:
        if args.encoder_type == "Nature":
            encoder = NatureCNN(env.observation_space.shape[0], args)
        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(env.observation_space.shape[0], args)

        if args.method == "random_cnn":
            print("Random CNN, so not loading in encoder weights!")
        elif args.method == "supervised":
            print("Fully supervised, so starting from random encoder weights!")
        elif args.method == "pretrained-rl-agent":
            print("Representation from pretrained rl agent, so we don't need an encoder!")
        elif args.method == "flat-pixels":
            print("Just using flattened pixels, so no need for encoder or weights for that matter!")
        else:
            if args.weights_path == "None":
                sys.stderr.write("Probing without loading in encoder weights! Are sure you want to do that??")
            else:
                print("Print loading in encoder weights from probe of type {} from the following path: {}"
                      .format(args.method, args.weights_path))
                encoder.load_state_dict(torch.load(args.weights_path))
                encoder.eval()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

    # encoder.to(device)
    torch.set_num_threads(1)

    all_runs_test_acc = appendabledict()
    for i, seed in enumerate(range(args.seed, args.seed + args.num_runs)):
        print("Run number {} of {}".format(i + 1, args.num_runs))
        test_acc = run_probe(encoder, args, device, seed)
        all_runs_test_acc.append_update(test_acc)

    mean_acc_dict = {"mean_" + k: np.mean(v) for k, v in all_runs_test_acc.items()}
    stderr_acc_dict = {"stderr_" + k: np.std(v) / np.sqrt(len(v)) for k, v in all_runs_test_acc.items()}
    print(mean_acc_dict)
    print(stderr_acc_dict)
    wandb.log(mean_acc_dict)
    wandb.log(stderr_acc_dict)


def get_random_agent_episodes(args, device, seed):
    envs = make_vec_envs(args.env_name, seed, args.num_processes, num_frame_stack=args.num_frame_stack,
                         downsample=not args.no_downsample)
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
    return episodes, episode_labels


def run_probe(encoder, args, device, seed):
    if args.probe_collect_mode == "random_agent":
        episodes, episode_labels = get_random_agent_episodes(args, device, seed)

    else:
        episodes, episode_labels = get_atari_zoo_episodes(args.env_name,
                                                          num_frame_stack=args.num_frame_stack,
                                                          downsample=not args.no_downsample,
                                                          algos=args.zoo_algos,
                                                          tags=args.zoo_tags,
                                                          use_representations_instead_of_frames=(
                                                                      "pretrained-rl-agent" in args.method))

        episodes = [torch.from_numpy(ep).float() for ep in episodes]

        if len(episodes[0].shape) > 2:
            episodes = [ep.permute(0, 3, 1, 2) for ep in episodes]

    episodes = [x for x in episodes if len(x) > args.batch_size]
    episode_labels = [x for x in episode_labels if len(x) > args.batch_size]

    inds = np.arange(len(episodes))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(inds)
    val_split_ind, te_split_ind = int(0.7 * len(inds)), int(0.85 * len(inds))

    tr_eps, val_eps, test_eps = episodes[:val_split_ind], episodes[val_split_ind:te_split_ind], episodes[te_split_ind:]
    tr_labels, val_labels, test_labels = episode_labels[:val_split_ind], episode_labels[
                                                                         val_split_ind:te_split_ind], episode_labels[
                                                                                                      te_split_ind:]

    feature_size = np.prod(tr_eps[0][0].shape[1:]) if args.method == "flat-pixels" else None
    trainer = ProbeTrainer(encoder,
                           wandb,
                           epochs=args.epochs,
                           sample_label=tr_labels[0][0],
                           lr=args.lr,
                           batch_size=args.batch_size,
                           device=device,
                           patience=args.patience,
                           log=False,
                           feature_size=feature_size)

    trainer.train(tr_eps, val_eps, tr_labels, val_labels)
    _, test_acc = trainer.evaluate(test_eps, test_labels)
    return test_acc


if __name__ == "__main__":
    tags = ['probe']
    wandb.init(project="curl-atari-2", entity="curl-atari", tags=tags)
    main()
