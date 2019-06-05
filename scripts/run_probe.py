from scripts.run_contrastive import train_encoder
from src.probe import ProbeTrainer
import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import gym

from src.envs import make_vec_envs
from src.utils import get_argparser, visualize_activation_maps, appendabledict, calculate_multiclass_accuracy, \
    calculate_multiclass_f1_score
from src.encoders import NatureCNN, ImpalaCNN
from src.pretrained_agents import get_ppo_rollouts, checkpointed_steps_full_sorted, get_ppo_representations
import wandb
import sys


def remove_duplicates(tr_eps, val_eps, test_eps, test_labels):
    """
    Remove any items in test_eps (&test_labels) which are present in tr/val_eps
    """
    flat_tr = list(chain.from_iterable(tr_eps))
    flat_val = list(chain.from_iterable(val_eps))
    tr_val_set = set([x.numpy().tostring() for x in flat_tr] + [x.numpy().tostring() for x in flat_val])
    flat_test = list(chain.from_iterable(test_eps))

    for i, episode in enumerate(test_eps[:]):
        test_labels[i] = [label for obs, label in zip(test_eps[i], test_labels[i]) if obs.numpy().tostring() not in tr_val_set]
        test_eps[i] = [obs for obs in episode if obs.numpy().tostring() not in tr_val_set]
    test_len = len(list(chain.from_iterable(test_eps)))
    dups = len(flat_test) - test_len
    print('Duplicates: {}, Test Len: {}'.format(dups, test_len))
    wandb.log({'Duplicates': dups, 'Test Len': test_len})
    return test_eps, test_labels


def remove_low_entropy_labels(episode_labels, entropy_threshold=0.3):
    flat_label_list = list(chain.from_iterable(episode_labels))
    counts = {}

    for label_dict in flat_label_list:
        for k in label_dict:
            counts[k] = counts.get(k, {})
            v = label_dict[k]
            counts[k][v] = counts[k].get(v, 0) + 1
    low_entropy_labels = []

    for k in counts:
        entropy = torch.distributions.Categorical(
            torch.tensor([x / len(flat_label_list) for x in counts[k].values()])).entropy()
        wandb.log({'entropy_' + k: entropy})
        if entropy < entropy_threshold:
            print("Deleting {} for being too low in entropy! Sorry, dood!".format(k))
            low_entropy_labels.append(k)

    for e in episode_labels:
        for obs in e:
            for key in low_entropy_labels:
                del obs[key]

    return episode_labels


def majority_baseline(tr_labels, test_labels, wandb):
    tr_labels = list(chain.from_iterable(tr_labels))
    test_labels = list(chain.from_iterable(test_labels))
    counts, maj_dict, test_counts = {}, {}, {}

    for label_dict in tr_labels:
        for k in label_dict:
            counts[k] = counts.get(k, {})
            v = label_dict[k]
            counts[k][v] = counts[k].get(v, 0) + 1

    # Get keys with maximum value
    for label in counts:
        maj_dict[label] = max(counts[label], key=counts[label].get)

    accuracy_dict = {}
    f1_score_dict = {}
    for k in test_labels[0]:
        labels = torch.tensor([label_d[k] for label_d in test_labels]).long()
        preds = torch.zeros(len(test_labels), 256)
        preds[:, maj_dict[k]] = 1
        accuracy = calculate_multiclass_accuracy(preds, labels)
        f1score = calculate_multiclass_f1_score(preds, labels)
        accuracy_dict[k + "_test_acc"] = accuracy
        f1_score_dict[k + "_f1score"] = f1score

    accuracy_dict['mean_test_acc'] = np.mean(list(accuracy_dict.values()))
    f1_score_dict["mean_f1score"] = np.mean(list(f1_score_dict.values()))
    wandb.log(accuracy_dict)
    wandb.log(f1_score_dict)
    return accuracy_dict, f1_score_dict


def get_random_agent_episodes(args, device):
    envs = make_vec_envs(args, args.num_processes)
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

    # Convert to 2d list from 3d list
    episodes = list(chain.from_iterable(episodes))
    # Convert to 2d list from 3d list
    episode_labels = list(chain.from_iterable(episode_labels))
    return episodes, episode_labels


def run_probe(encoder, args, device, seed):
    if args.method != "pretrained-rl-agent":
        if args.probe_collect_mode == "random_agent":
            episodes, episode_labels = get_random_agent_episodes(args, device)
        elif args.probe_collect_mode == 'pretrained_ppo':
            checkpoint = checkpointed_steps_full_sorted[args.checkpoint_index]
            episodes, episode_labels, mean_reward, mean_action_entropy = get_ppo_rollouts(args, args.probe_steps, checkpoint)
            wandb.log({'action_entropy': mean_action_entropy, 'mean_reward': mean_reward})

    elif args.method == 'pretrained-rl-agent':
        checkpoint = checkpointed_steps_full_sorted[args.checkpoint_index]
        episodes, episode_labels, mean_reward = get_ppo_representations(args, checkpoint)
        wandb.log({"reward": mean_reward, "checkpoint": checkpoint})

    print("got episodes!")
    ep_inds = [i for i in range(len(episodes)) if len(episodes[i]) > args.batch_size]
    episodes = [episodes[i] for i in ep_inds]
    episode_labels = [episode_labels[i] for i in ep_inds]
    episode_labels = remove_low_entropy_labels(episode_labels, entropy_threshold=args.entropy_threshold)

    inds = np.arange(len(episodes))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(inds)
    val_split_ind, te_split_ind = int(0.7 * len(inds)), int(0.8 * len(inds))

    tr_eps, val_eps, test_eps = episodes[:val_split_ind], episodes[val_split_ind:te_split_ind], episodes[te_split_ind:]
    tr_labels, val_labels, test_labels = episode_labels[:val_split_ind], episode_labels[
                                                                         val_split_ind:te_split_ind], episode_labels[
                                                                                                      te_split_ind:]
    test_eps, test_labels = remove_duplicates(tr_eps, val_eps, test_eps, test_labels)
    for e in range(len(episodes)):
        episodes[e] = torch.stack(episodes[e])

    if args.method == 'majority':
        return majority_baseline(tr_labels, test_labels, wandb)

    trainer = ProbeTrainer(encoder,
                           wandb,
                           epochs=args.epochs,
                           sample_label=tr_labels[0][0],
                           lr=args.lr,
                           batch_size=args.batch_size,
                           device=device,
                           patience=args.patience,
                           log=False)

    trainer.train(tr_eps, val_eps, tr_labels, val_labels)
    test_acc, test_f1score = trainer.test(test_eps, test_labels)

    return test_acc, test_f1score


def main(args):
    # dummy env
    env = make_vec_envs(args, 1)
    wandb.config.update(vars(args))

    if args.train_encoder and args.method in ['appo', 'spatial-appo', 'cpc', 'vae', 'bert', 'ms-dim', 'pixel_predictor',
                                              "naff", "infonce-stdim", "global-infonce-stdim", "global-local-infonce-stdim"]:
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
        if args.method == "majority":
            print("Majority baseline!")
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
    env.close()

    # encoder.to(device)
    torch.set_num_threads(1)

    all_runs_test_f1 = appendabledict()
    all_runs_test_acc = appendabledict()
    for i, seed in enumerate(range(args.seed, args.seed + args.num_runs)):
        print("Run number {} of {}".format(i + 1, args.num_runs))
        test_acc, f1score = run_probe(encoder, args, device, seed=1)
        all_runs_test_f1.append_update(f1score)
        all_runs_test_acc.append_update(test_acc)

    mean_acc_dict = {"mean_" + k: np.mean(v) for k, v in all_runs_test_acc.items()}
    var_acc_dict = {"var_" + k: np.var(v) for k, v in all_runs_test_acc.items()}
    mean_f1_dict = {"mean_" + k: np.mean(v) for k, v in all_runs_test_f1.items()}
    var_f1_dict = {"var_" + k: np.var(v) for k, v in all_runs_test_f1.items()}
    print(mean_acc_dict)
    print(var_acc_dict)
    wandb.log(mean_acc_dict)
    wandb.log(var_acc_dict)
    print(mean_f1_dict)
    print(var_f1_dict)
    wandb.log(mean_f1_dict)
    wandb.log(var_f1_dict)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['probe']
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    main(args)
