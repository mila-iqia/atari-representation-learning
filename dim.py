import torch
import torch.nn as nn
import numpy as np
from a2c_ppo_acktr.utils import init
from torch.utils.data import BatchSampler, RandomSampler


class MIEstimator():
    def __init__(self, encoder, feature_size=512, global_span=8, lr=5e-4, epochs=10):
        super(MIEstimator, self).__init__()

        self.encoder = encoder
        self.feature_size = feature_size
        self.discriminator = Discriminator(self.feature_size*2)
        self.optimizer = torch.optim.Adam([self.encoder.parameters(), self.discriminator.parameters()], lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.global_span = global_span
        self.epochs = epochs
        self.mini_batch_size = 128

    def data_generator(self, episodes):
        """
        Iteratively yield batches of data
        :return:
        """
        # Only consider episodes with len >= global_span
        episodes = [x for x in episodes if len(x) >= self.global_span]
        total_steps = sum([len(e) for e in episodes])
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps // self.global_span),
                               self.mini_batch_size, drop_last=False)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            pos_obs_batch, neg_obs_batch = [], []
            for episode in episodes_batch:
                start_idx, start_idx_neg = np.random.choice(len(episode)), np.random.choice(len(episode))
                pos_obs_batch += episode[start_idx:start_idx+self.global_span] # Append
                neg_obs_batch += episode[start_idx_neg:start_idx_neg+self.global_span]

            yield pos_obs_batch, neg_obs_batch

    def maximize_mi(self, episodes):
        """
        JSD based maximization of MI for `self.epochs` number of epochs
        Equivalent to minimizing Binary Cross-Entropy
        """
        for e in range(self.epochs):
            data_generator = self.data_generator(episodes)
            for batch in data_generator:
                pos_obs_batch, neg_obs_batch = batch

                pos_features_batch = self.encoder(pos_obs_batch)
                global_features = torch.mean(pos_features_batch).unsqueeze(0).expand(self.mini_batch_size, -1)
                pos_samples = torch.cat((pos_features_batch, global_features), dim=-1)
                neg_features_batch = self.encoder(neg_obs_batch)
                neg_samples = torch.cat((neg_features_batch, global_features), dim=-1)
                target = torch.cat((torch.ones(self.mini_batch_size, 1), torch.zeros(self.mini_batch_size, 1)), dim=0)
                samples = torch.cat((pos_samples, neg_samples), dim=0)

                self.optimizer.zero_grad()
                loss = self.loss_fn(self.discriminator(samples, target))
                loss.backward()
                self.optimizer.step()


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(Discriminator, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               np.sqrt(2))

        self.network = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)),
        )

    def forward(self, x):
        return self.network(x)

