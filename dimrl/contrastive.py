import itertools

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy


class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)


class ContrastiveTrainer():
    def __init__(self, encoder, mode='pcl', epochs=100, mini_batch_size=64, lr=3e-4, device=torch.device('cpu')):
        self.encoder = encoder
        self.mode = mode
        self.classifier = Classifier(self.encoder.hidden_size * 2).to(device)
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier.parameters()) + list(self.encoder.parameters()),
                                          lr=lr, eps=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.mini_batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.mini_batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                while abs(t_hat - t) < 5:
                    t, t_hat = np.random.randint(1, len(episode)), np.random.randint(1, len(episode))
                x_t.append(episode[t])
                x_tprev.append(episode[t - 1])
                if self.mode == 'both':
                    x_that.append(episode[t_hat-1])
                else:
                    x_that.append(episode[t_hat])
                ts.append([t])
                thats.append([t_hat])
            yield torch.stack(x_t) / 255., torch.stack(x_tprev) / 255., torch.stack(x_that) / 255., \
                  torch.Tensor(ts), torch.Tensor(thats)

    def train(self, episodes, wandb):
        # Convert to 2d list from 3d list
        episodes = list(itertools.chain.from_iterable(episodes))
        for e in range(self.epochs):
            epoch_loss, accuracy, steps = 0., 0., 0
            data_generator = self.generate_batch(episodes)
            for x_t, x_tprev, x_that, ts, thats in data_generator:
                f_t, f_t_prev = self.encoder(x_t), self.encoder(x_tprev)
                f_t_2, f_t_hat = self.encoder(x_t), self.encoder(x_that)

                if self.mode == 'pcl':
                    f_pos, f_neg = torch.cat((f_t, f_t_prev), dim=-1), torch.cat((f_t_2, f_t_hat), dim=-1)
                elif self.mode == 'tcl':
                    f_pos, f_neg = torch.cat((f_t, ts), dim=-1), torch.cat((f_t_2, thats), dim=-1)
                elif self.mode == 'both':
                    f_pos, f_neg = torch.cat((f_t, f_t_prev, ts), dim=-1), torch.cat((f_t_2, f_t_hat, thats), dim=-1)

                samples = torch.cat((f_pos, f_neg), dim=0)
                target = torch.cat((torch.ones(self.mini_batch_size, 1),
                                    torch.zeros(self.mini_batch_size, 1)), dim=0).to(self.device)

                self.optimizer.zero_grad()
                loss = self.loss_fn(self.classifier(samples), target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.detach().item()
                preds = torch.sigmoid(self.classifier(samples))
                accuracy += calculate_accuracy(preds, target)
                steps += 1
            print('Epoch: {}, Loss: {}, Accuracy: {}'.format(e, epoch_loss / steps, accuracy / steps))
            wandb.log({'Accuracy': accuracy / steps}, step=e)


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=1024):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.network(x)
