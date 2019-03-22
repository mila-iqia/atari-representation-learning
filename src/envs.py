import torch
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from a2c_ppo_acktr.envs import make_env, VecNormalize, VecPyTorch


def make_vec_envs(env_name, seed, num_processes, gamma=0.99, log_dir='/tmp/',
                  device=torch.device('cpu'), allow_early_resets=False):
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

    # if num_frame_stack is not None:
    #     envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    # elif len(envs.observation_space.shape) == 3:
    #     envs = VecPyTorchFrameStack(envs, 4, device)

    return envs