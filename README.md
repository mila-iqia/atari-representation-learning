# Unsupervised State Representation Learning in Atari

Ankesh Anand*, Evan Racah*, Sherjil Ozair*, Yoshua Bengio, Marc-Alexandre Côté, R Devon Hjelm

https://arxiv.org/abs/1906.08226

State representation learning, or the ability to capture latent generative factors of an environment, is crucial for building intelligent agents that can perform a wide variety of tasks. Learning such representations without supervision from rewards is a challenging open problem. We introduce a method that learns state representations by maximizing mutual information across spatially and temporally distinct features of a neural encoder of the observations. We also introduce a new benchmark based on Atari 2600 games where we evaluate representations based on how well they capture the ground truth state variables. We believe this new framework for evaluating representation learning models will be crucial for future representation learning research. Finally, we compare our technique with other state-of-the-art generative and contrastive representation learning methods.


## Installation
#### Dependencies:
* PyTorch 
* OpenAI Baselines (for vectorized environments and Atari wrappers)
* pytorch-a2c-ppo-acktr (for actor-critic algorithms)
* wandb (logging tool)
* gym[atari]
* opencv-python

To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c pytorch

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# pytorch-a2c-ppo-acktr for RL utils
git clone https://github.com/ankeshanand/pytorch-a2c-ppo-acktr-gail
cd pytorch-a2c-ppo-acktr-gail
pip install -e .

# Other requirements
pip install -r requirements.txt
```

To install the project: 

```bash
git clone https://github.com/ankeshanand/atari-representation-learning.git
cd atari-representation-learning
pip install -e .
```

## Usage 
### Atari Annotated RAM Interface (AARI): 

![AARI](aari/aari.png?raw=true)


AARI exposes the ground truth labels for different state variables for each observation. We have made AARI available as a Gym wrapper, to use it simply wrap an Atari gym env with `AARIWrapper`. 
```python
import gym
from aari.wrapper import AARIWrapper
env = AARIWrapper(gym.make('MontezumaRevengeNoFrameskip-v4'))
obs = env.reset()
obs, reward, done, info = env.step(1)
```

Now, `info` is a dictionary of the form:

```python
{'ale.lives': 6, 
'labels': {
  'room_number': 1, 
  'player_x': 77, 
  'player_y': 235, 
  'player_direction': 8, 
  'enemy_skull_x': 57, 
  'enemy_skull_y': 240, 
  'key_monster_x': 14, 
  'key_monster_y': 254, 
  'level': 0, 
  'num_lives': 5, 
  'items_in_inventory_count': 0, 
  'room_state': 8, 
  'score_0': 0, 
  'score_1': 0, 
  'score_2': 0}
  }
```

**Note:** In our experiments, we use additional preprocessing for Atari environments mainly following Minh et. al, 2014. See [aari/envs.py](aari/envs.py) for more info! 

If you want the raw RAM annotations (which parts of ram correspond to each state variable), check out [aari/ram_annotations.py](aari/ram_annotations.py)


### Probing
We provide an interface for the included probing tasks.

First, get episodes for train, val and, test:
```python
from aari.episodes import get_episodes

tr_episodes, val_episodes,\
tr_labels, val_labels,\
test_episodes, test_labels = get_episodes(env_name="PitfallNoFrameskip-v4", 
                                     steps=50000, 
                                     collect_mode="random_agent")
```

Then probe them using ProbeTrainer and your encoder (`my_encoder`):

```python
from aari.probe import ProbeTrainer

probe_trainer = ProbeTrainer(my_encoder, representation_len=my_encoder.feature_size)
probe_trainer.train(tr_episodes, val_episodes,
                     tr_labels, val_labels,)
final_accuracies, final_f1_scores = probe_trainer.test(test_episodes, test_labels)
```

To see how we use ProbeTrainer, check out [scripts/run_probe.py](scripts/run_probe.py)

Here is an example of `my_encoder`:
```python 
# get your encoder
import torch.nn as nn
import torch
class MyEncoder(nn.Module):
    def __init__(self, input_channels, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.input_channels = input_channels
        self.final_conv_size = 64 * 9 * 6
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.final_conv_size, self.feature_size)

    def forward(self, inputs):
        x = self.cnn(inputs)
        x = x.view(x.size(0), -1)
        return self.fc(x)
        

my_encoder = MyEncoder(input_channels=1,feature_size=256)
# load in weights
my_encoder.load_state_dict(torch.load(open("path/to/my/weights.pt", "rb")))
```

### Spatio-Temporal DeepInfoMax:
`src/` contains implementations of several representation learning methods, along with `ST-DIM`. Here's a sample usage: 

```bash
python -m scripts.run_probe --method infonce-stdim --env-name {env_name}
```
