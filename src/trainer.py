import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy


class Trainer():
    def __init__(self, encoder, wandb, device=torch.device('cpu')):
        self.encoder = encoder
        self.wandb = wandb
        self.device = device

    def generate_batch(self, episodes):
        raise NotImplementedError

    def train(self, episodes):
        raise NotImplementedError

    def log_results(self, epoch_idx, epoch_loss, accuracy):
        raise NotImplementedError

