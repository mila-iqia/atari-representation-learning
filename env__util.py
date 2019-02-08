from multiprocessing import Process, Pipe
from baselines.common.vec_env import CloudpickleWrapper, VecEnvWrapper, VecEnv
from coinrun import setup_utils
import coinrun.main_utils as utils


def get_env_fun(num_processes=8, test=False):
    def _thunk():
        if test:
            _ = setup_utils.setup_and_load(test_eval=True, set_seed=42)
        else:
            setup_utils.setup_and_load(num_levels=500)
        envs = utils.make_general_env(num_processes)
        return envs
    return _thunk

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

class CoinrunSubprocess(VecEnv):
    """
    Run a CoinrunVecEnv inside a subprocess.
    A particular use-case of this is to encapsulate train and test envs inside separate subprocesses,
    to ensure they are thread safe and don't interfere with each other's global state.
    """
    def __init__(self, num_processes=32, test=False, spaces=None):
        self.test = test
        self.num_envs = num_processes
        env_fn = get_env_fun(num_processes=num_processes, test=self.test)
        self.waiting = False
        self.closed = False
        self.remote, self.work_remote = Pipe()
        self.p = Process(target=worker, args=(self.work_remote, self.remote, CloudpickleWrapper(env_fn)))
        self.p.daemon = True  # if the main process crashes, we should not cause things to hang
        self.p.start()
        self.work_remote.close()
        self.remote.send(('get_spaces', None))
        self.observation_space, self.action_space = self.remote.recv()
        self.viewer = None
        VecEnv.__init__(self, self.num_envs, self.observation_space, self.action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        self.remote.send(('step', actions))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = self.remote.recv()
        self.waiting = False
        obs, rews, dones, infos = results
        return obs, rews, dones, infos

    def reset(self):
        self._assert_not_closed()
        self.remote.send(('reset', None))
        return self.remote.recv()

    def close_extras(self):
        self.closed = True
        if self.waiting:
            self.remote.recv()
        self.remote.send(('close', None))
        self.p.join()

    def get_images(self):
        self._assert_not_closed()
        self.remote.send(('render', None))
        imgs = self.remote.recv()
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
