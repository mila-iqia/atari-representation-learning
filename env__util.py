from multiprocessing import Pool
import torch

from a2c_ppo_acktr.envs import VecPyTorch
from baselines.common.vec_env import CloudpickleWrapper
from baselines.common.vec_env.vec_monitor import VecMonitor
from coinrun import setup_utils
import coinrun.main_utils as utils


def get_subprocess_envs(num_processes=8, test=False, device=torch.device('cpu')):
    if test:
        _ = setup_utils.setup_and_load(test_eval=True, set_seed=42)
    else:
        setup_utils.setup_and_load(num_levels=500)
    envs = utils.make_general_env(num_processes)
    envs = VecPyTorch(VecMonitor(envs, filename='monitor.csv'), device)
    return envs

def get_train_test_envs(num_processes=8, device=torch.device('cpu')):
    p = Pool(processes=2)
    envs = p.starmap(get_subprocess_envs, [(num_processes, False, device), (num_processes, True, device)])
    return envs[0], envs[1]


