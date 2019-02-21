import copy
import glob
import os
import torch
from a2c_ppo_acktr.arguments import get_args
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


def preprocess():
    args = get_args()
    writer = SummaryWriter(comment='runs')

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    eval_log_dir = args.log_dir + "_eval"

    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    return args, writer, num_updates, eval_log_dir


def calculate_accuracy(preds, y):
    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()
    return acc


def save_model(model, save_dir, model_name):
    pass


def visualize_activation_maps(encoder, input_obs_batch):
    scaled_images = input_obs_batch / 255.
    fmap = F.relu(encoder.layer3(encoder.layer2(encoder.layer1(scaled_images)))).detach()
    out_channels = fmap.shape[1]
    # upsample and add a dummy channel dimension
    fmap_upsampled = F.interpolate(fmap, size=input_obs_batch.shape[-2:], mode='bilinear').unsqueeze(dim=2)
    for i in range(input_obs_batch.shape[0]):
        fmap_grid = make_grid(fmap_upsampled[i], normalize=True)
        img_grid = make_grid([scaled_images[i]] * out_channels)
        plt.imshow(img_grid.cpu().numpy().transpose([1, 2, 0]))
        plt.imshow(fmap_grid.cpu().numpy().transpose([1, 2, 0]), cmap='jet', alpha=0.7)
        plt.savefig('act_map' + str(i) + '.png')
