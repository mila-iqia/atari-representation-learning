# dim-rl
Representation Learning for RL using DeepInfomax

Dependencies: 
* PyTorch 
* OpenAI Baselines (for vectorized environments and Atari wrappers)
* pytorch-a2c-ppo-acktr (for actor-critic algorithms)
* wandb (logging tool)
* gym

To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c soumith

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

Usage: 
* python -m scripts.run_contrastive --method appo
