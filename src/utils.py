import argparse
import copy
import os
import subprocess

import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize
from collections import defaultdict


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4',
                        help='environment to train on (default: MontezumaRevengeNoFrameskip-v4)')
    parser.add_argument('--num-frame-stack', type=int, default=4,
                        help='Number of frames to stack for a state')
    parser.add_argument('--pretraining-steps', type=int, default=100000,
                        help='Number of steps to pretrain representations (default: 100000)')
    parser.add_argument('--probe-train-steps', type=int, default=30000,
                        help='Number of steps to train probes (default: 30000 )')
    parser.add_argument('--probe-test-steps', type=int, default=15000,
                        help='Number of steps to train probes (default: 15000 )')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--method', type=str, default='appo', choices=["appo", "cpc", "supervised", "random", "nonlinear"],
                        help='Method to use for training representations (default: appo)')
    parser.add_argument('--mode', type=str, default='pcl',
                        help='Mode to use when using the Appo estimator [pcl | tcl | both] (default: pcl)')
    parser.add_argument('--linear', action='store_true', default=True,
                        help='Whether to use a linear classifier')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning Rate foe learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Mini-Batch Size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for  (default: 100)')
    parser.add_argument('--cuda-id', type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to use')
    parser.add_argument('--encoder-type', type=str, default="Nature", choices=["Impala", "Nature"],
                        help='Encoder type (Impala or Nature)')
    parser.add_argument("--patience", type=int, default=10)

    # CPC-specific arguments
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='Sequence length.')
    parser.add_argument('--steps_start', type=int, default=0,
                        help='Number of immediate future steps to ignore.')
    parser.add_argument('--steps_end', type=int, default=99,
                        help='Number of future steps to predict.')
    parser.add_argument('--steps_step', type=int, default=4,
                        help='Number of future steps to predict.')
    parser.add_argument('--gru_size', type=int, default=512,
                        help='Hidden size of the GRU layers.')
    parser.add_argument('--gru_layers', type=int, default=2,
                        help='Number of GRU layers.')
    return parser


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calculate_accuracy(preds, y):
    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()
    return acc


def calculate_multiclass_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    acc = float(torch.sum(torch.eq(labels, preds)).data) / labels.size(0)
    return acc


def save_model(model, envs, save_dir, model_name, use_cuda):
    save_path = os.path.join(save_dir)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    # A really ugly way to save a model to CPU
    save_model = model
    if use_cuda:
        save_model = copy.deepcopy(model).cpu()

    save_model = [save_model,
                  getattr(get_vec_normalize(envs), 'ob_rms', None)]

    torch.save(save_model, os.path.join(save_path, model_name + ".pt"))


def evaluate_policy(actor_critic, envs, args, eval_log_dir, device):
    eval_envs = make_vec_envs(
        args.env_name, args.seed + args.num_processes, args.num_processes,
        args.gamma, eval_log_dir, args.add_timestep, device, True)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                               actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                        for done_ in done])
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".
          format(len(eval_episode_rewards),
                 np.mean(eval_episode_rewards)))
    eval_envs.close()
    return eval_episode_rewards


def visualize_activation_maps(encoder, input_obs_batch, wandb):
    scaled_images = input_obs_batch / 255.
    if encoder.__class__.__name__ == 'ImpalaCNN':
        fmap = F.relu(encoder.layer3(encoder.layer2(encoder.layer1(scaled_images)))).detach()
    elif encoder.__class__.__name__ == 'NatureCNN':
        fmap = F.relu(encoder.main[4](F.relu(encoder.main[2](F.relu(encoder.main[0](input_obs_batch)))))).detach()
    out_channels = fmap.shape[1]
    # upsample and add a dummy channel dimension
    fmap_upsampled = F.interpolate(fmap, size=input_obs_batch.shape[-2:], mode='bilinear').unsqueeze(dim=2)
    for i in range(input_obs_batch.shape[0]):
        fmap_grid = make_grid(fmap_upsampled[i], normalize=True)
        img_grid = make_grid([scaled_images[i]] * out_channels)
        plt.imshow(img_grid.cpu().numpy().transpose([1, 2, 0]))
        plt.imshow(fmap_grid.cpu().numpy().transpose([1, 2, 0]), cmap='jet', alpha=0.5)
        # plt.savefig('act_maps/' + 'file%02d.png' % i)
        wandb.log({'actmap': wandb.Image(plt, caption='Activation Map')})
    # generate_video()


def generate_video():
    os.chdir("act_maps")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])


class appendabledict(defaultdict):
    def __init__(self, type_=list, *args, **kwargs):
        self.type_ = type_
        super().__init__(type_, *args, **kwargs)

    #     def map_(self, func):
    #         for k, v in self.items():
    #             self.__setitem__(k, func(v))

    def subslice(self, slice_):
        """indexes every value in the dict according to a specified slice

        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.


        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}

         """
        sliced_dict = {}
        for k, v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict

    def append_update(self, other_dict):
        """appends current dict's values with values from other_dict

        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to append to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

         """
        for k, v in other_dict.items():
            self.__getitem__(k).append(v)


# Thanks Bjarten! (https://github.com/Bjarten/early-stopping-pytorch)
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, wandb=None, name=""):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.
        self.name = name
        self.wandb = wandb

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping for {self.name} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'{self.name} has stopped')

        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation accuracy increased for {self.name}  ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')

        save_dir = self.wandb.run.dir
        torch.save(model.state_dict(), save_dir + "/" + self.name + ".pt")
        self.val_acc_max = val_acc


def bucket_coord(coord, num_buckets, min_coord=0, max_coord=255, stride=1):
    # stride is how much a variable is incremented by (usually 1)
    try:
        assert (coord <= max_coord and coord >= min_coord)
    except:
        print("coord: %i, max: %i, min: %i, num_buckets: %i" % (coord, max_coord, min_coord, num_buckets))
        assert False, coord
    coord_range = (max_coord - min_coord) + 1

    # thresh is how many units in raw_coord space correspond to one bucket
    if coord_range < num_buckets:  # we never want to upsample from the original coord
        thresh = stride
    else:
        thresh = coord_range / num_buckets
    bucketed_coord = np.floor((coord - min_coord) / thresh)
    return bucketed_coord


def bucket_discrete(coord, possible_values):
    # possible values is a list of all values a coord can take on
    inds = range(len(possible_values))
    hash_table = dict(zip(possible_values, inds))
    return hash_table[coord]
