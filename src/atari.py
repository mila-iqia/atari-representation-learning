from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np
import os

atari_dict = {"Pitfall": {"ram": dict(agent_x=97,
                                      agent_y=105),
                                 "num_classes": {}},
                  
                  
                  
              "PrivateEye": {"ram": dict(agent_x=63,
                                           agent_y=86),
                                    "num_classes": {}},
              
              "MontezumaRevenge": {"ram": dict(room_number=3,
                                              agent_facing_direction=52, # 72 if facing left, 128 if facing right
                                              agent_x=42,
                                              agent_y=43,
                                              skull_x=47,
                                              skull_y=46),
                                    "num_classes": {} },
              
              "Pong": {"ram":dict(player_y=51,
                                           enemy=50,
                                           ball_x=49,
                                           ball_y=54,
                                           enemy_score=13,
                                           player_score=14),
                                "num_classes":{}}
             }
														

class InfoWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, self.info(info)


    def info(self):
        raise NotImplementedError
        
    def labels(self):
        raise NotImplementedError
        
class AtariWrapper(InfoWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        env_name = self.env.spec.id
        self.env_name = env_name.split("-")[0].split("No")[0]
        
        if self.env_name in atari_dict:
            self.ram_dict = atari_dict[self.env_name]["ram"]
            ncd = atari_dict[self.env_name]["num_classes"] 
            if len(ncd) > 0:
                self.nclasses_dict = ncd
            else:
                self.nclasses_dict = {k:256 for k in self.ram_dict.keys()}
        
        
        
           
    def info(self, info):
        if self.env_name  in atari_dict:
            info["num_classes"] = self.nclasses_dict
            label_dict = self.labels()
            info["labels"] = label_dict
        return info
    
    def labels(self):
        ram = self.env.env.ale.getRAM()
        label_dict = {k:ram[ind] for k,ind in self.ram_dict.items()}
        return label_dict