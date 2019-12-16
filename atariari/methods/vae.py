import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init

import os
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .trainer import Trainer
from .utils import EarlyStopping


class Unflatten(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x_uf = x.view(-1, *self.new_shape)
        return x_uf


class Decoder(nn.Module):
    def __init__(self, feature_size, final_conv_size, final_conv_shape, num_input_channels, encoder_type="Nature"):
        super().__init__()
        self.feature_size = feature_size
        self.final_conv_size = final_conv_size
        self.final_conv_shape = final_conv_shape
        self.num_input_channels = num_input_channels
        # self.fc =
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        if encoder_type == "Nature":
            self.main = nn.Sequential(
                nn.Linear(in_features=self.feature_size,
                          out_features=self.final_conv_size),
                nn.ReLU(),
                Unflatten(self.final_conv_shape),

                init_(nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0,
                                         output_padding=1)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=32, out_channels=num_input_channels,
                                         kernel_size=8, stride=4, output_padding=(2, 0))),
                nn.Sigmoid()
            )

    def forward(self, f):
        im = self.main(f)
        return im


class VAE(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.feature_size = self.encoder.feature_size
        self.final_conv_size = self.encoder.final_conv_size
        self.final_conv_shape = self.encoder.final_conv_shape
        self.input_channels = self.encoder.input_channels

#         self.mu_fc = nn.Linear(in_features=self.feature_size,
#                                    out_features=self.feature_size)
        
        self.logvar_fc = nn.Linear(in_features=self.final_conv_size,
                                   out_features=self.feature_size)

        self.decoder = Decoder(feature_size=self.feature_size,
                               final_conv_size=self.final_conv_size,
                               final_conv_shape=self.final_conv_shape,
                               num_input_channels=self.input_channels)

    def reparametrize(self, mu, logvar):
        if self.training:
            eps = torch.randn(*logvar.size()).to(mu.device)
            std = torch.exp(0.5 * logvar)
            z = mu + eps * std
        else:
            z = mu
        return z

    def forward(self, x):
        mu = self.encoder(x)
        logvar = self.logvar_fc(self.encoder.main[:-1](x))
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class VAELoss(object):
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, x, x_hat, mu, logvar):
        kldiv = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))
        rec = F.mse_loss(x_hat, x, reduction='sum')
        loss = rec + self.beta * kldiv
        return loss


class VAETrainer(Trainer):
    # TODO: Make it work for all modes, right now only it defaults to pcl.
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.VAE = VAE(encoder).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.VAE.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.loss_fn = VAELoss(beta=self.config["beta"])
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
            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
                x_t.append(episode[t])
            yield torch.stack(x_t).float().to(self.device) / 255.

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.VAE.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        data_generator = self.generate_batch(episodes)
        for x_t in data_generator:
            with torch.set_grad_enabled(mode == 'train'):
                x_hat, mu, logvar = self.VAE(x_t)
                loss = self.loss_fn(x_t, x_hat, mu, logvar)

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            steps += 1
        self.log_results(epoch, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    #             xim = x_hat.detach().cpu().numpy()[0].transpose(1,2,0)
    #             self.wandb.log({"example_reconstruction": [self.wandb.Image(xim, caption="")]})

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.VAE.train()
            self.do_one_epoch(e, tr_eps)

            self.VAE.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}".format(prefix.capitalize(), epoch_idx, epoch_loss))
        self.wandb.log({prefix + '_loss': epoch_loss})
