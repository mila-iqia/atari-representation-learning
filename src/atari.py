from gym.envs.registration import register
import gym
from gym import spaces
from .atari_ram_annotations import atari_dict

class InfoWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, self.info(info)

    def reset(self,**kwargs):
        return self.env.reset(**kwargs)


    def info(self):
        raise NotImplementedError

    def labels(self):
        raise NotImplementedError


class AtariWrapper(InfoWrapper):
    def __init__(self, env):
        super().__init__(env)
        env_name = self.env.spec.id
        self.env_name = env_name.split("-")[0].split("No")[0].lower()
        # print(list(atari_dict.keys()))
        # print(self.env_name)

        if self.env_name in atari_dict:

            self.ram_dict = atari_dict[self.env_name]
            self.nclasses_dict = {k: 256 for k in self.ram_dict.keys()}

    def info(self, info):
        if self.env_name in atari_dict:

            info["num_classes"] = self.nclasses_dict
            label_dict = self.labels()
            info["labels"] = label_dict
        return info

    def labels(self):
        ram = self.env.unwrapped.ale.getRAM()
        label_dict = {k: ram[ind] for k, ind in self.ram_dict.items()}
        return label_dict
