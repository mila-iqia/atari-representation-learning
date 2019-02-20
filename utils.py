import copy
import glob
import os
import torch
from a2c_ppo_acktr.arguments import get_args
from tensorboardX import SummaryWriter
import torch.nn.functional as F


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


def visualize_activation_maps(model, input_obs):
    fmap = F.relu(model[4](F.relu(model[2](F.relu(model[0](input_obs))))))