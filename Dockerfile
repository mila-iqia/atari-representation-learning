FROM phillyregistry.azurecr.io/philly/jobs/custom/generic-docker:py36

ENV PATH /opt/conda/bin:$PATH

RUN sudo chmod -R 777 /opt/conda && \
    pip install tensorflow-gpu opencv-python 'gym[atari]' matplotlib wandb wget && \
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch && \
    pip install https://github.com/openai/baselines/archive/master.zip && \
    pip install https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/archive/master.zip

# Login to wandb
ARG wandb_key
ENV LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8
RUN wandb login $wandb_key