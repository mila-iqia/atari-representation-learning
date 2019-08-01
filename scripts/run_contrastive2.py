import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from src.dim_baseline import DIMTrainer
from src.static_dim import StaticDIMTrainer
from src.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from src.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from src.infonce_spatio_temporal_new import InfoNCESpatioTemporalTrainer3
from src.spatio_temporal import SpatioTemporalTrainer
from src.static_dim2 import StaticDIMTrainer2
from src.utils import get_argparser
from src.encoders import NatureCNN, ImpalaCNN
from src.cpc import CPCTrainer
from src.vae import VAETrainer
from src.no_action_feedforward_predictor import NaFFPredictorTrainer
from src.infonce_spatio_temporal import InfoNCESpatioTemporalTrainer
import wandb
from aari.episodes import get_episodes


def train_encoder(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    envs = ['AsteroidsNoFrameskip-v4', 'BerzerkNoFrameskip-v4', 'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4',
            'BreakoutNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4',
            'HeroNoFrameskip-v4', 'MontezumaRevengeNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'PitfallNoFrameskip-v4',
            'PongNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'QbertNoFrameskip-v4', 'RiverraidNoFrameskip-v4',
            'SeaquestNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'TennisNoFrameskip-v4', 'VentureNoFrameskip-v4',
            'VideoPinballNoFrameskip-v4', 'YarsRevengeNoFrameskip-v4']

    all_tr_episodes, all_val_episodes = [], []
    pos_tr_eps, pos_val_eps = [], []
    for env in envs:
        steps = 5000
        if env == args.env_name:
            steps = 50000
        tr_eps, val_eps = get_episodes(steps=steps,
                                       env_name=env,
                                       seed=args.seed,
                                       num_processes=args.num_processes,
                                       num_frame_stack=args.num_frame_stack,
                                       downsample=not args.no_downsample,
                                       color=args.color,
                                       entropy_threshold=args.entropy_threshold,
                                       collect_mode=args.probe_collect_mode,
                                       train_mode="train_encoder",
                                       checkpoint_index=args.checkpoint_index,
                                       min_episode_length=args.batch_size)
        if env == args.env_name:
            pos_tr_eps = tr_eps
            pos_val_eps = val_eps
        else:
            all_tr_episodes.append(tr_eps)
            all_val_episodes.append(val_eps)

    all_tr_episodes = list(chain.from_iterable(all_tr_episodes))
    all_val_episodes = list(chain.from_iterable(all_val_episodes))
    observation_shape = pos_tr_eps[0][0].shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    if args.method == 'cpc':
        trainer = CPCTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'spatial-appo':
        trainer = SpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'vae':
        trainer = VAETrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "naff":
        trainer = NaFFPredictorTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "static-dim":
        trainer = StaticDIMTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "stdim-3":
        trainer = InfoNCESpatioTemporalTrainer3(encoder, config, device=device, wandb=wandb)
    elif args.method == "static-dim-2":
        trainer = StaticDIMTrainer2(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    trainer.train(pos_tr_eps, pos_val_eps, all_tr_episodes, all_val_episodes)

    return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    train_encoder(args)
