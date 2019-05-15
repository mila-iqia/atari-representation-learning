import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy
from .trainer import Trainer
from src.utils import EarlyStopping
from src.vae import Decoder

class PixelPredictorTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        for k, v in config.items():
            setattr(self, k, v)

        self.feature_size = self.encoder.feature_size
        self.final_conv_size = self.encoder.final_conv_size
        self.final_conv_shape = self.encoder.final_conv_shape
        self.input_channels = self.encoder.input_channels

        self.decoder = Decoder(feature_size=self.feature_size,
                               final_conv_size=self.final_conv_size,
                               final_conv_shape=self.final_conv_shape,
                               num_input_channels=self.input_channels)

        self.decoder = self.decoder.to(device)

        self.device = device
        self.steps_gen = lambda: range(self.steps_start, self.steps_end, self.steps_step)
        self.discriminators = {i: nn.Linear(self.gru_size, self.encoder.hidden_size).to(device) for i in self.steps_gen()}
        self.gru = nn.GRU(input_size=self.encoder.hidden_size, hidden_size=self.gru_size, num_layers=self.gru_layers, batch_first=True).to(device)
        self.labels = {i: torch.arange(self.batch_size * (self.sequence_length - i - 1)).to(device) for i in self.steps_gen()}
        params = list(self.encoder.parameters())
        for disc in self.discriminators.values():
          params += disc.parameters()
        params += self.gru.parameters()
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
              sequences.append(seq)
            yield torch.stack(sequences)

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.gru.training else "val"
        steps = 0
        step_losses = {i: [] for i in self.steps_gen()}
        step_accuracies = {i: [] for i in self.steps_gen()}

        data_generator = self.generate_batch(episodes)
        for sequence in data_generator:
            sequence = sequence.to(self.device)
            sequence = sequence / 255.
            w, h = self.config['obs_space'][-2], self.config['obs_space'][-1]
            flat_sequence = sequence.view(-1, self.num_frame_stack, w, h)
            flat_latents = self.encoder(flat_sequence)
            latents = flat_latents.view(
                self.batch_size, self.sequence_length, self.encoder.hidden_size)
            contexts, _ = self.gru(latents)
            loss = 0.
            for i in self.steps_gen():
              latent_predictions = self.discriminators[i](contexts[:, :-(i+1)])
              pixel_predictions = self.decoder(latent_predictions)

              flat_pixel_targets = sequence[:, i+1:].contiguous().view(-1, self.num_frame_stack, w, h)
              # print(flat_pixel_targets.size())
              # print(pixel_predictions.size())

              step_loss = F.mse_loss(pixel_predictions, flat_pixel_targets)
              step_losses[i].append(step_loss.detach().item())
              loss += step_loss

              # preds = torch.argmax(logits, dim=1)
              # step_accuracy = preds.eq(self.labels[i]).sum().float() / self.labels[i].numel()
              # step_accuracies[i].append(step_accuracy.detach().item())

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            steps += 1
        epoch_losses = {i: np.mean(step_losses[i]) for i in step_losses}
        # epoch_accuracies = {i: np.mean(step_accuracies[i]) for i in step_accuracies}
        self.log_results(epoch, epoch_losses, prefix=mode)
        if mode == "val":
            self.early_stopper(-np.mean(list(epoch_losses.values())), self.encoder)

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

    def log_results(self, epoch_idx, epoch_losses, prefix=""):
        print("Epoch: {}".format(epoch_idx))
        print("Step Losses[{}: {}: {}]: {}".format(self.steps_start, self.steps_end, self.steps_step, ", ".join(map(str, epoch_losses.values()))))
        log_results = {}
        for i in self.steps_gen():
          log_results[prefix + '_step_loss_{}'.format(i+1)] = epoch_losses[i]
        self.wandb.log(log_results)
