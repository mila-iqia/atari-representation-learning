import random

import torch
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms
import torchvision.transforms.functional as TF


class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class MultiStepSTDIM(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        for k, v in config.items():
            setattr(self, k, v)
        self.steps_gen = range(self.steps_start+1, self.steps_end+1, self.steps_step)
        self.classifiers_gl = {i: Classifier(self.encoder.hidden_size, 128).to(device) for i in self.steps_gen}
        self.classifiers_ll = {i: Classifier(128, 128).to(device) for i in self.steps_gen}
        params = list(self.encoder.parameters())
        for c in list(self.classifiers_gl.values()) + list(self.classifiers_ll.values()):
            params += c.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config['lr'], eps=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.device = device

    def generate_batch(self, episodes):
        episodes = [episode for episode in episodes if len(episode) >= self.sequence_length]
        total_steps = sum([len(e) for e in episodes])
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_t, x_tpos, x_that = {i: [] for i in self.steps_gen}, {i: [] for i in self.steps_gen},\
                                   {i: [] for i in self.steps_gen}
            for episode in episodes_batch:
                for i in self.steps_gen:
                    t, t_hat = np.random.randint(0, len(episode) - i), np.random.randint(0, len(episode))
                    x_t[i].append(episode[t])
                    x_tpos[i].append(episode[t + i])
                    x_that[i].append(episode[t_hat])

            for i in self.steps_gen:
                x_t[i] = torch.stack(x_t[i]).to(self.device) / 255.
                x_tpos[i] = torch.stack(x_tpos[i]).to(self.device) / 255.
                x_that[i] = torch.stack(x_that[i]).to(self.device) / 255.

            yield x_t, x_tpos, x_that

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training else "val"
        step_losses = {i: [] for i in self.steps_gen}
        step_accuracies = {i: [] for i in self.steps_gen}
        data_generator = self.generate_batch(episodes)
        for x_t_dict, x_tpos_dict, x_that_dict in data_generator:
            for i in self.steps_gen:
                x_t, x_tpos, x_that = x_t_dict[i], x_tpos_dict[i], x_that_dict[i]
                f_t_maps, f_t_pos_maps = self.encoder(x_t, fmaps=True), self.encoder(x_tpos, fmaps=True)
                f_t_hat_maps = self.encoder(x_that, fmaps=True)

                # Loss 1: Global at time t, f5 patches at time t-1
                f_t, f_t_pos = f_t_maps['out'], f_t_pos_maps['f5']
                f_t_hat = f_t_hat_maps['f5']
                f_t = f_t.unsqueeze(1).unsqueeze(1).expand(-1, f_t_pos.size(1), f_t_pos.size(2),
                                                           self.encoder.hidden_size)

                target = torch.cat((torch.ones_like(f_t[:, :, :, 0]),
                                    torch.zeros_like(f_t[:, :, :, 0])), dim=0).to(self.device)

                x1, x2 = torch.cat([f_t, f_t], dim=0), torch.cat([f_t_pos, f_t_hat], dim=0)
                shuffled_idxs = torch.randperm(len(target))
                x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]
                self.optimizer.zero_grad()
                loss1 = self.loss_fn(self.classifiers_gl[i](x1, x2).squeeze(), target)

                # Loss 2: f5 patches at time t, with f5 patches at time t-1
                f_t = f_t_maps['f5']
                x1_p, x2_p = torch.cat([f_t, f_t], dim=0), torch.cat([f_t_pos, f_t_hat], dim=0)
                x1_p, x2_p = x1_p[shuffled_idxs], x2_p[shuffled_idxs]
                loss2 = self.loss_fn(self.classifiers_ll[i](x1_p, x2_p).squeeze(), target)

                loss = loss1 + loss2
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                step_losses[i].append(loss.detach().item())
                preds1 = torch.sigmoid(self.classifiers_gl[i](x1, x2).squeeze())
                accuracy1 = calculate_accuracy(preds1, target)
                preds2 = torch.sigmoid(self.classifiers_ll[i](x1_p, x2_p).squeeze())
                accuracy2 = calculate_accuracy(preds2, target)
                accuracy = (accuracy1 + accuracy2) / 2.
                step_accuracies[i].append(accuracy.detach().item())

        epoch_losses = {i: np.mean(step_losses[i]) for i in step_losses}
        epoch_accuracies = {i: np.mean(step_accuracies[i]) for i in step_accuracies}
        self.log_results(epoch, epoch_losses, epoch_accuracies, prefix=mode)
        if mode == "val":
            self.early_stopper(np.mean(list(epoch_accuracies.values())), self.encoder)

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.encoder.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_losses, epoch_accuracies, prefix=""):
        print("Epoch: {}".format(epoch_idx))
        print("Step Losses[{}: {}: {}]: {}".format(self.steps_start+1, self.steps_end+1, self.steps_step,
                                                   ", ".join(map(str, epoch_losses.values()))))
        print("Step Accuracies[{}: {}: {}]: {}".format(self.steps_start+1, self.steps_end+1, self.steps_step,
                                                       ", ".join(map(str, epoch_accuracies.values()))))
        log_results = {}
        for i in self.steps_gen:
            log_results[prefix + '_step_loss_{}'.format(i + 1)] = epoch_losses[i]
            log_results[prefix + '_step_accuracy_{}'.format(i)] = epoch_accuracies[i]
        self.wandb.log(log_results, step=epoch_idx)
