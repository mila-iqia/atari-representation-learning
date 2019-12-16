import random

import torch
import os
import torch
import torch.nn as nn
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


class SpatioTemporalTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.classifier1 = Classifier(self.encoder.hidden_size, self.encoder.local_layer_depth).to(device)
        self.classifier2 = Classifier(self.encoder.local_layer_depth, self.encoder.local_layer_depth).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier1.parameters()) + list(self.encoder.parameters()) +
                                          list(self.classifier2.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()
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
                x_that.append(episode[t_hat])

                ts.append([t])
                thats.append([t_hat])
            yield torch.stack(x_t).float().to(self.device) / 255., torch.stack(x_tprev).float().to(self.device) / 255., \
                  torch.stack(x_that).float().to(self.device) / 255., torch.Tensor(ts).to(self.device), \
                  torch.Tensor(thats).to(self.device)

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2 = 0., 0.
        epoch_loss1, epoch_loss2 = 0., 0.
        data_generator = self.generate_batch(episodes)
        for x_t, x_tprev, x_that, ts, thats in data_generator:
            f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True), self.encoder(x_tprev, fmaps=True)
            f_t_hat_maps = self.encoder(x_that, fmaps=True)

            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
            f_t_hat = f_t_hat_maps['f5']
            f_t = f_t.unsqueeze(1).unsqueeze(1).expand(-1, f_t_prev.size(1), f_t_prev.size(2), self.encoder.hidden_size)

            target = torch.cat((torch.ones_like(f_t[:, :, :, 0]),
                                torch.zeros_like(f_t[:, :, :, 0])), dim=0).to(self.device)

            x1, x2 = torch.cat([f_t, f_t], dim=0), torch.cat([f_t_prev, f_t_hat], dim=0)
            shuffled_idxs = torch.randperm(len(target))
            x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]
            self.optimizer.zero_grad()
            loss1 = self.loss_fn(self.classifier1(x1, x2).squeeze(), target)

            # Loss 2: f5 patches at time t, with f5 patches at time t-1
            f_t = f_t_maps['f5']
            x1_p, x2_p = torch.cat([f_t, f_t], dim=0), torch.cat([f_t_prev, f_t_hat], dim=0)
            x1_p, x2_p = x1_p[shuffled_idxs], x2_p[shuffled_idxs]
            loss2 = self.loss_fn(self.classifier2(x1_p, x2_p).squeeze(), target)

            loss = loss1 + loss2
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_loss1 += loss1.detach().item()
            epoch_loss2 += loss2.detach().item()
            preds1 = torch.sigmoid(self.classifier1(x1, x2).squeeze())
            accuracy1 += calculate_accuracy(preds1, target)
            preds2 = torch.sigmoid(self.classifier2(x1_p, x2_p).squeeze())
            accuracy2 += calculate_accuracy(preds2, target)
            steps += 1
        self.log_results(epoch, epoch_loss1 / steps, epoch_loss2 / steps, epoch_loss / steps,
                         accuracy1 / steps, accuracy2 / steps, (accuracy1 + accuracy2) / steps, prefix=mode)
        if mode == "val":
            self.early_stopper((accuracy1 + accuracy2) / steps, self.encoder)

    def train(self, tr_eps, val_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(), self.classifier1.train(), self.classifier2.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval(), self.classifier1.eval(), self.classifier2.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss2, epoch_loss, accuracy1, accuracy2, accuracy, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {} Accuracy: {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize(), accuracy))
        self.wandb.log({prefix + '_loss': epoch_loss, prefix + '_accuracy': accuracy,
                        prefix + '_loss1': epoch_loss1, prefix + '_accuracy1': accuracy1,
                        prefix + '_loss2': epoch_loss2, prefix + '_accuracy2': accuracy2}, step=epoch_idx)
