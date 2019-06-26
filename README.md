# rl-representation-learning
Representation Learning for RL

Dependencies: 
* PyTorch 
* OpenAI Baselines (for vectorized environments and Atari wrappers)
* pytorch-a2c-ppo-acktr (for actor-critic algorithms)
* wandb (logging tool)
* gym
* wget

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

### Usage: 
```bash
python -m scripts.probe --method infonce-stdim --env-name {env_name}
```
