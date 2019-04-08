import torch
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from a2c_ppo_acktr.envs import TimeLimitMask, MaskGoal, TransposeObs, TransposeImage, VecPyTorch, VecNormalize, \
    VecPyTorchFrameStack
from pathlib import Path
import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from src.atari import AtariWrapper


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
            env = AtariWrapper(env)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma=0.99, log_dir='./tmp/',
                  device=torch.device('cpu'), num_frame_stack=1, allow_early_resets=False):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    envs = [make_env(env_name, seed, i, log_dir, allow_early_resets)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack > 1:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

    return envs
