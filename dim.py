import torch
import torch.nn as nn
import numpy as np
from a2c_ppo_acktr.utils import init
from torch.utils.data import BatchSampler, RandomSampler
import itertools
import random

from utils import calculate_accuracy


class MIEstimator():
    def __init__(self, encoder, feature_size=512, global_span=8, lr=5e-4, epochs=10, device=torch.device('cpu')):
        super(MIEstimator, self).__init__()

        self.device = device
        self.encoder = encoder
        self.feature_size = feature_size
        self.discriminator = Discriminator(self.feature_size * 2).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.discriminator.parameters()),
                                          lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.global_span = global_span
        self.epochs = epochs
        self.mini_batch_size = 128

    def data_generator(self, episodes):
        """
        Iteratively yield batches of data
        :return:
        """
        # Convert to 2d list from 3d list
        episodes = list(itertools.chain.from_iterable(episodes))
        # Only consider episodes with len >= global_span
        episodes = [x for x in episodes if len(x) >= self.global_span]
        total_steps = sum([len(e) for e in episodes])
        # Episode sampler
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps // self.global_span),
                               self.mini_batch_size, drop_last=True)
        # Iteratively yield batches of data
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            pos_obs_batch, neg_obs_batch = [], []
            for episode in episodes_batch:
                start_idx, start_idx_neg = np.random.choice(len(episode) - self.global_span + 1), \
                                           np.random.choice(len(episode) - self.global_span + 1)
                pos_obs_batch.append(torch.stack(episode[start_idx:start_idx + self.global_span]))  # Append
                neg_obs_batch.append(torch.stack(random.sample(episode, self.global_span)))

            # shape: mini_batch_size * global_span * obs_shape, normalize inputs
            yield torch.stack(pos_obs_batch) / 255., torch.stack(neg_obs_batch) / 255.

    def maximize_mi(self, episodes):
        """
        JSD based maximization of MI for `self.epochs` number of epochs
        Equivalent to minimizing Binary Cross-Entropy
        """
        epoch_loss, accuracy, steps = 0., 0., 0
        for e in range(self.epochs):
            data_generator = self.data_generator(episodes)
            for batch in data_generator:
                pos_obs_batch, neg_obs_batch = batch
                # pass to encoder as (-1, obs_shape)
                pos_features_batch = self.encoder(pos_obs_batch.view(-1, *pos_obs_batch.shape[2:]))
                # reshape to mini_batch_size * global_span * feature_size to compute global features
                pos_features_batch = pos_features_batch.view(self.mini_batch_size, self.global_span, -1)

                # compute global features by taking mean around the global_span dimension,
                # then expand as features_batch to concat with features_batch
                global_features = torch.mean(pos_features_batch, dim=1, keepdim=True).expand_as(pos_features_batch)
                pos_samples = torch.cat((pos_features_batch, global_features), dim=-1)
                # reshape to (-1, feature_size*2)
                pos_samples = pos_samples.view(-1, pos_samples.shape[-1])

                # Do the same to get neg_samples
                neg_features_batch = self.encoder(neg_obs_batch.view(-1, *neg_obs_batch.shape[2:]))
                neg_features_batch = neg_features_batch.view(self.mini_batch_size, self.global_span, -1)
                neg_samples = torch.cat((neg_features_batch, global_features), dim=-1)
                neg_samples = neg_samples.view(-1, neg_samples.shape[-1])

                # Create target tensor
                target = torch.cat((torch.ones(self.mini_batch_size * self.global_span, 1),
                                    torch.zeros(self.mini_batch_size * self.global_span, 1)), dim=0).to(self.device)
                # Concatenate positive and negative samples along 0th dimension to get data to feed to the disc
                samples = torch.cat((pos_samples, neg_samples), dim=0)

                self.optimizer.zero_grad()
                loss = self.loss_fn(self.discriminator(samples), target)
                loss.backward()
                epoch_loss += loss.detach().item()
                self.optimizer.step()
                preds = torch.sigmoid(self.discriminator(samples))
                accuracy += calculate_accuracy(preds, target)
                steps += 1
        return epoch_loss / steps, accuracy / steps


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=256):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)
