import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy
from .trainer import Trainer


class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_size=256, linear=False):
        super().__init__()
        if not linear:
            self.network = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        else:
            self.network = nn.Bilinear(num_inputs, num_inputs, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class AppoTrainer(Trainer):
    # TODO: Make it work for all modes, right now only it defaults to pcl.
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.mode = config['mode']
        self.feature_sizes = {
            'pcl': self.encoder.hidden_size * 2,
            'tcl': self.encoder.hidden_size + 1,
            'both': self.encoder.hidden_size * 2 + 1
        }
        self.classifier = Classifier(self.encoder.hidden_size, linear=config['linear']).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier.parameters()) + list(self.encoder.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
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
                    x_that.append(episode[t_hat - 1])
                else:
                    x_that.append(episode[t_hat])
                ts.append([t])
                thats.append([t_hat])
            yield torch.stack(x_t).to(self.device) / 255., torch.stack(x_tprev).to(self.device) / 255., torch.stack(x_that).to(self.device) / 255., \
                  torch.Tensor(ts).to(self.device), torch.Tensor(thats).to(self.device)

            
    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.classifier.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        data_generator = self.generate_batch(episodes)
        for x_t, x_tprev, x_that, ts, thats in data_generator:
            f_t, f_t_prev = self.encoder(x_t), self.encoder(x_tprev)
            f_t_2, f_t_hat = self.encoder(x_t), self.encoder(x_that)
            target = torch.cat((torch.ones(self.batch_size, 1),
                                torch.zeros(self.batch_size, 1)), dim=0).to(self.device)

            x1, x2 = torch.cat([f_t, f_t_2]), torch.cat([f_t_prev, f_t_hat])
            shuffled_idxs = torch.randperm(len(target))
            x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.classifier(x1, x2), target)
            
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            preds = torch.sigmoid(self.classifier(x1, x2))
            accuracy += calculate_accuracy(preds, target)
            steps += 1
        self.log_results(epoch, epoch_loss / steps, accuracy / steps, prefix=mode )
        
    def train(self, tr_eps, val_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(), self.classifier.train()
            self.do_one_epoch(e, tr_eps)
            
            self.encoder.eval(), self.classifier.eval()
            self.do_one_epoch(e, val_eps)
            
            
            
            
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir,  self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss, accuracy, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {} Accuracy: {}".format(prefix.capitalize(), epoch_idx, epoch_loss, prefix.capitalize(), accuracy))
        self.wandb.log({prefix + '_loss': epoch_loss, prefix + '_accuracy': accuracy})
