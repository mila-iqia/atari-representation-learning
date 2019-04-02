# rl-reprsentation-learning
Representation Learning for RL

Dependencies: 
* PyTorch 
* OpenAI Baselines (for vectorized environments and Atari wrappers)
* pytorch-a2c-ppo-acktr (for actor-critic algorithms)
* wandb (logging tool)
* gym

To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c pytorch

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# pytorch-a2c-ppo-acktr for RL utils
git clone https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.git
cd pytorch-a2c-ppo-acktr-gail
pip install -e .

# Other requirements
pip install -r requirements.txt
```

### Usage: 
```bash
python -m scripts.run_contrastive --method appo
```
#### Probing
```bash
python -m scripts.run_probe --weights_path ./path/where/I/store/my/models/model.pt
```
### Notes

Every estimator inherits from `Trainer`:
```python
class Trainer():
    def __init__(self, encoder, wandb, device=torch.device('cpu')):
        self.encoder = encoder
        self.wandb = wandb
        self.device = device

    def generate_batch(self, episodes):
        raise NotImplementedError

    def train(self, episodes):
        raise NotImplementedError

    def log_results(self, epoch_idx, epoch_loss, accuracy):
        raise NotImplementedError
```
Look into `appo.py` for an example implementation.

### Directory structure

    .
    ├── scripts                 #  "main" files to run experiments
    │   ├── run_contrastive.py  # Pretrain representations 
    │   └── ppo_with_reps.py    # Run PPO after pre-training representations (Unstable)        
    │
    ├── src                     # Source files
    │   ├── actor_critic.py     # Util file that allows to easily swith encoders for a policy 
    │   ├── appo.py             # Implements contrastive estimators described in Appo's paper
    │   ├── encoders.py         # NatureCNN and ImpalaCNN encoders (maps image to a flat feature space) 
    │   ├── trainer.py          # Abstract class that describes how an estimator should be implemented
    │   └── utils.py            # Functions for parsing arguments, visualizing act_maps etc.
    │
    ├── exp.sh                  # Sample bash script for running jobs
    └── README.md
