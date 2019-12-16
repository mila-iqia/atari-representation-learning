import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init
import numpy as np
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy
from .trainer import Trainer
from .vae import Decoder
from .utils import EarlyStopping
import sys


class NaFFPredictor(nn.Module):
    def __init__(self, encoder, fc_size=2048):
        super().__init__()
        self.encoder = encoder
        self.fc_size = fc_size
        self.feature_size = self.encoder.feature_size
        self.final_conv_size = self.encoder.final_conv_size
        self.final_conv_shape = self.encoder.final_conv_shape
        self.input_channels = self.encoder.input_channels

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=self.feature_size,
                      out_features=self.feature_size))

        self.decoder = Decoder(feature_size=self.feature_size,
                               final_conv_size=self.final_conv_size,
                               final_conv_shape=self.final_conv_shape,
                               num_input_channels=self.input_channels)

    def forward(self, x):
        feature_vector = self.encoder(x)
        z = self.fc_layers(feature_vector)
        x_hat = self.decoder(z)
        return x_hat


class NaFFPredictorTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.fc_size = self.config["naff_fc_size"]
        self.pred_offset = self.config["pred_offset"]
        self.naff = NaFFPredictor(encoder, self.fc_size).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.naff.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.loss_fn = nn.MSELoss()
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")

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
            x_t, x_tn = [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t = np.random.randint(0, len(episode) - self.pred_offset)
                t_n = t + self.pred_offset

                x_t.append(episode[t])
                x_tn.append(episode[t_n])
            yield torch.stack(x_t).float().to(self.device) / 255., \
                  torch.stack(x_tn).float().to(self.device) / 255.

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.naff.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        data_generator = self.generate_batch(episodes)
        for x_t, x_tn in data_generator:
            with torch.set_grad_enabled(mode == 'train'):
                x_tn_hat = self.naff(x_t)
                loss = self.loss_fn(x_tn_hat, x_tn)

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            steps += 1
        self.log_results(epoch, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.naff.train()
            self.do_one_epoch(e, tr_eps)

            self.naff.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}".format(prefix.capitalize(), epoch_idx, epoch_loss))
        self.wandb.log({prefix + '_loss': epoch_loss})
