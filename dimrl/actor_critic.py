import torch.nn as nn
import torch
import numpy as np

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.model import Policy, NNBase
from a2c_ppo_acktr.utils import init


class LatentPolicy(Policy):
    def __init__(self, action_space, encoder):
        super(Policy, self).__init__()
        self.base = CNNBase(encoder)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    def forward(self, inputs, rnn_hxs, masks):
        return self.base(inputs, rnn_hxs, masks)


class CNNBase(NNBase):
    def __init__(self, encoder, hidden_size=256):
        super().__init__(False, encoder.hidden_size, encoder.hidden_size)
        self.encoder = encoder
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               np.sqrt(2))

        # self.actor = nn.Sequential(
        #     init_(nn.Linear(self.encoder.hidden_size, hidden_size)),
        #     nn.Tanh(),
        #     init_(nn.Linear(hidden_size, hidden_size)),
        #     nn.Tanh()
        # )
        #
        # self.critic = nn.Sequential(
        #     init_(nn.Linear(self.encoder.hidden_size, hidden_size)),
        #     nn.Tanh(),
        #     init_(nn.Linear(hidden_size, hidden_size)),
        #     nn.Tanh()
        # )

        self.critic_linear = init_(nn.Linear(self.encoder.hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        with torch.no_grad():
            features = self.encoder(inputs)
        # hidden_actor = self.actor(features)
        # hidden_critic = self.critic(features)
        return self.critic_linear(features), features, rnn_hxs
