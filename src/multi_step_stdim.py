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
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=len(episodes) * self.sequence_length),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            sequences = []
            for episode in episodes_batch:
                start_index = np.random.randint(0, len(episode) - self.sequence_length + 1)
                seq = episode[start_index: start_index + self.sequence_length]
                sequences.append(seq)
            yield torch.stack(sequences).to(self.device) / 255.

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2 = 0., 0.
        step_accuracies = {i: [] for i in self.steps_gen}
        data_generator = self.generate_batch(episodes)
        for sequence_batch in data_generator:
            w, h = self.config['obs_space'][-2], self.config['obs_space'][-1]
            flat_sequence = sequence_batch.view(-1, self.num_frame_stack, w, h)
            flat_latents = self.encoder(flat_sequence, fmaps=True)
            flat_latents_g, flat_latents_f5 = flat_latents['out'], flat_latents['f5']
            w_f5, h_f5 = flat_latents_f5[0].size(-3), flat_latents_f5[0].size(-2),
            latents_g = flat_latents_g.view(self.batch_size, self.sequence_length, self.encoder.hidden_size)
            latents_f5 = flat_latents_f5.view(self.batch_size, self.sequence_length, w_f5, h_f5, 128)
            loss = 0.
            for i in self.steps_gen:
                anchor_idx, neg_idx = 0, i
                while neg_idx == anchor_idx + i:
                    anchor_idx, neg_idx = np.random.randint(0, self.sequence_length - i), np.random.randint(0, self.sequence_length)
                pos_idx = anchor_idx + i

                # Loss 1: Global at time t, f5 patches at time t+i
                f_t = latents_g[:, anchor_idx, :]
                f_t_i, f_t_hat = latents_f5[:, pos_idx, :], latents_f5[:, neg_idx, :]
                f_t = f_t.unsqueeze(1).unsqueeze(1).expand(-1, f_t_i.size(1), f_t_i.size(2),
                                                           self.encoder.hidden_size)
                x1, x2 = torch.cat([f_t, f_t], dim=0), torch.cat([f_t_i, f_t_hat], dim=0)
                target = torch.cat((torch.ones_like(f_t_i[:, :, :, 0]),
                                    torch.zeros_like(f_t_i[:, :, :, 0])), dim=0).to(self.device)
                loss1 = self.loss_fn(self.classifiers_gl[i](x1, x2).squeeze(), target)

                # Loss 2: f5 patches at time t, with f5 patches at time t+i
                f_t = latents_f5[:, anchor_idx, :]
                f_t_i, f_t_hat = latents_f5[:, pos_idx, :], latents_f5[:, neg_idx, :]
                x1_p, x2_p = torch.cat([f_t, f_t], dim=0), torch.cat([f_t_i, f_t_hat], dim=0)
                loss2 = self.loss_fn(self.classifiers_ll[i](x1_p, x2_p).squeeze(), target)

                self.optimizer.zero_grad()
                loss += loss1 + loss2

                epoch_loss += (loss1 + loss2).detach().item()
                preds1 = torch.sigmoid(self.classifiers_gl[i](x1, x2).squeeze())
                accuracy1 += calculate_accuracy(preds1, target)
                preds2 = torch.sigmoid(self.classifiers_ll[i](x1_p, x2_p).squeeze())
                accuracy2 += calculate_accuracy(preds2, target)
                accuracy = (accuracy1 + accuracy2) / 2.
                step_accuracies[i].append(accuracy.detach().item())
                steps += 1

            if mode == "train":
                loss.backward()
                self.optimizer.step()
        self.log_results(epoch, step_accuracies, prefix=mode)
        if mode == "val":
            self.early_stopper(accuracy / steps, self.encoder)

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.encoder.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_accuracies, prefix=""):
        print("Epoch: {}".format(epoch_idx))
        print("Step Accuracies[{}: {}: {}]: {}".format(self.steps_start, self.steps_end, self.steps_step,
                                                       ", ".join(map(str, epoch_accuracies.values()))))
        log_results = {}
        for i in self.steps_gen:
            log_results[prefix + '_step_accuracy_{}'.format(i + 1)] = epoch_accuracies[i]
        self.wandb.log(log_results)
