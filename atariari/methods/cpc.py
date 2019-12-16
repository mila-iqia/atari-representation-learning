import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy
from .trainer import Trainer
from .utils import EarlyStopping


class CPCTrainer(Trainer):
    # TODO: Make it work for all modes, right now only it defaults to pcl.
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        for k, v in config.items():
            setattr(self, k, v)

        self.device = device
        self.steps_gen = lambda: range(self.steps_start, self.steps_end, self.steps_step)
        self.discriminators = {i: nn.Linear(self.gru_size, self.encoder.hidden_size).to(device) for i in self.steps_gen()}
        self.gru = nn.GRU(input_size=self.encoder.hidden_size, hidden_size=self.gru_size, num_layers=self.gru_layers, batch_first=True).to(device)
        self.labels = {i: torch.arange(self.batch_size * (self.sequence_length - i - 1)).to(device) for i in self.steps_gen()}
        params = list(self.encoder.parameters()) + list(self.gru.parameters())
        for disc in self.discriminators.values():
          params += disc.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config['lr'])
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")

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
              start_index = np.random.randint(0, len(episode) - self.sequence_length+1)
              seq = episode[start_index: start_index + self.sequence_length]
              sequences.append(torch.stack(seq))
            yield torch.stack(sequences).float()

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.gru.training else "val"
        steps = 0
        step_losses = {i: [] for i in self.steps_gen()}
        step_accuracies = {i: [] for i in self.steps_gen()}

        data_generator = self.generate_batch(episodes)
        for sequence in data_generator:
            with torch.set_grad_enabled(mode == 'train'):
                sequence = sequence.to(self.device)
                sequence = sequence / 255.
                channels, w, h = self.config['obs_space'][-3:]
                flat_sequence = sequence.view(-1, channels, w, h)
                flat_latents = self.encoder(flat_sequence)
                latents = flat_latents.view(
                    self.batch_size, self.sequence_length, self.encoder.hidden_size)
                contexts, _ = self.gru(latents)
                loss = 0.
                for i in self.steps_gen():
                  predictions = self.discriminators[i](contexts[:, :-(i+1), :]).contiguous().view(-1, self.encoder.hidden_size)
                  targets = latents[:, i+1:, :].contiguous().view(-1, self.encoder.hidden_size)
                  logits = torch.matmul(predictions, targets.t())
                  step_loss = F.cross_entropy(logits, self.labels[i])
                  step_losses[i].append(step_loss.detach().item())
                  loss += step_loss

                  preds = torch.argmax(logits, dim=1)
                  step_accuracy = preds.eq(self.labels[i]).sum().float() / self.labels[i].numel()
                  step_accuracies[i].append(step_accuracy.detach().item())

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            steps += 1
        epoch_losses = {i: np.mean(step_losses[i]) for i in step_losses}
        epoch_accuracies = {i: np.mean(step_accuracies[i]) for i in step_accuracies}
        self.log_results(epoch, epoch_losses, epoch_accuracies, prefix=mode)
        if mode == "val":
            self.early_stopper(np.mean(list(epoch_accuracies.values())), self.encoder)

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.encoder.train(), self.gru.train()
            for k, disc in self.discriminators.items():
                disc.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval(), self.gru.eval()
            for k, disc in self.discriminators.items():
                disc.eval()
            self.do_one_epoch(e, val_eps)
            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir,  self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_losses, epoch_accuracies, prefix=""):
        print("Epoch: {}".format(epoch_idx))
        print("Step Losses[{}: {}: {}]: {}".format(self.steps_start, self.steps_end, self.steps_step, ", ".join(map(str, epoch_losses.values()))))
        print("Step Accuracies[{}: {}: {}]: {}".format(self.steps_start, self.steps_end, self.steps_step, ", ".join(map(str, epoch_accuracies.values()))))
        log_results = {}
        for i in self.steps_gen():
          log_results[prefix + '_step_loss_{}'.format(i+1)] = epoch_losses[i]
          log_results[prefix + '_step_accuracy_{}'.format(i+1)] = epoch_accuracies[i]
        self.wandb.log(log_results)
