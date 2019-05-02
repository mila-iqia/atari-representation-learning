FROM phillyregistry.azurecr.io/philly/jobs/custom/generic-docker:py36

# Install Pytorch
RUN conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch \
 && conda clean -ya

# Install Tensorflow for baselines
RUN conda install -y tensorflow \
 && conda clean -ya

# Baselines for Atari preprocessing
RUN pip install https://github.com/openai/baselines/archive/master.zip

# pytorch-a2c-ppo-acktr for RL utils
RUN pip install https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/archive/master.zip

# Install additional requirements
RUN pip install 'gym[atari]' matplotlib wandb wget

# Login to wandb
ARG wandb_key
ENV LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8
RUN wandb login $wandb_key