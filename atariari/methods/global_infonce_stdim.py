import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from .utils import EarlyStopping
from torchvision import transforms
import torchvision.transforms.functional as TF


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class GlobalInfoNCESpatioTemporalTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.classifier1 = nn.Linear(self.encoder.hidden_size, 256).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier1.parameters()) + list(self.encoder.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

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
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
                x_t.append(episode[t])

                # Apply the same transform to x_{t-1} and x_{t_hat}
                # https://github.com/pytorch/vision/issues/9#issuecomment-383110707
                # Use numpy's random seed because Cutout uses np
                # seed = random.randint(0, 2 ** 32)
                # np.random.seed(seed)
                x_tprev.append(episode[t - 1])
                # np.random.seed(seed)
                #x_that.append(episode[t_hat])

                ts.append([t])
                #thats.append([t_hat])
            yield torch.stack(x_t).float().to(self.device) / 255., torch.stack(x_tprev).float().to(self.device) / 255.

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2 = 0., 0.
        epoch_loss1, epoch_loss2 = 0., 0.
        data_generator = self.generate_batch(episodes)
        for x_t, x_tprev in data_generator:
            f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True), self.encoder(x_tprev, fmaps=True)

            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['out']

            N = f_t.size(0)
            predictions = self.classifier1(f_t)
            logits = torch.matmul(predictions, f_t_prev.t())
            step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
            loss = step_loss

            self.optimizer.zero_grad()
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            #preds1 = torch.sigmoid(self.classifier1(x1, x2).squeeze())
            #accuracy1 += calculate_accuracy(preds1, target)
            #preds2 = torch.sigmoid(self.classifier2(x1_p, x2_p).squeeze())
            #accuracy2 += calculate_accuracy(preds2, target)
            steps += 1
        self.log_results(epoch, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(), self.classifier1.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval(), self.classifier1.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss}, step=epoch_idx)
