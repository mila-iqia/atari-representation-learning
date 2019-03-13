import torch.nn as nn

from a2c_ppo_acktr.model import Policy, NNBase


class CNNBase(NNBase):
    def __init__(self, encoder):
        super().__init__(False, encoder.hidden_size, encoder.hidden_size)
        self.encoder = encoder
        self.critic_linear = nn.Linear(encoder.hidden_size, 1)

    def forward(self, inputs, rnn_hxs, masks):
        out = self.encoder(inputs)
        return self.critic_linear(out), out, rnn_hxs
