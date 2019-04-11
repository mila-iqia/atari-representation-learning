import torch
from src.utils import EarlyStopping
from torch import nn
from src.trainer import Trainer
from src.utils import appendabledict
from src.utils import calculate_multiclass_accuracy
from copy import deepcopy
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_classes)

    def forward(self, feature_vectors):
        return self.model(feature_vectors)


class FullySupervisedLinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=255):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.probe = LinearProbe(input_dim=self.encoder.hidden_size,
                                 num_classes=num_classes)

    def forward(self, x):
        feature_vec = self.encoder(x)
        return self.probe(feature_vec)


class ProbeTrainer(Trainer):
    def __init__(self, encoder, wandb, info_dict,
                 device=torch.device('cpu'),
                 epochs=100, lr=5e-4, batch_size=64, patience=15):
        super().__init__(encoder, wandb, device)
        self.info_dict = info_dict
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = wandb.config["method"]
        self.info_dict = info_dict

        if self.method == "supervised":
            self.probes = {k: FullySupervisedLinearProbe(encoder=self.encoder,
                                                         num_classes=info_dict[k]).to(device) for k in info_dict.keys()}
        else:
            # info_dict should have {label_name: number_of_classes_for_that_label}
            self.probes = {k: LinearProbe(input_dim=encoder.hidden_size,
                                          num_classes=info_dict[k]).to(device) for k in info_dict.keys()}

        self.early_stoppers = {k: EarlyStopping(patience=patience, verbose=False, wandb=self.wandb, name=k + "_probe")
                               for k in
                               info_dict.keys()}

        self.optimizers = {k: torch.optim.Adam(list(self.probes[k].parameters()),
                                               eps=1e-5, lr=self.lr) for k in info_dict.keys()}
        self.schedulers = {k: torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers[k], patience=5, factor=0.2, verbose=True, mode='max') for k in info_dict.keys()}
        self.loss_fn = nn.CrossEntropyLoss()

    def generate_batch(self, episodes, episode_labels):
        total_steps = sum([len(e) for e in episodes])
        # print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            episode_labels_batch = [episode_labels[x] for x in indices]
            xs, labels = [], appendabledict()
            for ep_ind, episode in enumerate(episodes_batch):
                # Get one sample from this episode
                t = np.random.randint(len(episode))
                xs.append(episode[t])
                labels.append_update(episode_labels_batch[ep_ind][t])
            yield torch.stack(xs).to(self.device) / 255., labels

    def probe(self, batch, k):
        [probe.to("cpu") for probe in self.probes.values()]
        probe = self.probes[k]
        probe.to(self.device)
        if self.method == "supervised":
            preds = probe(batch)
        else:
            with torch.no_grad():
                self.encoder.to(self.device)
                f = self.encoder(batch).detach()
            preds = probe(f)
        return preds

    def do_one_epoch(self, episodes, label_dicts):
        epoch_loss, accuracy = {k + "_loss": [] for k in self.info_dict.keys() if
                                not self.early_stoppers[k].early_stop}, \
                               {k + "_acc": [] for k in self.info_dict.keys() if not self.early_stoppers[k].early_stop}

        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                if self.early_stoppers[k].early_stop:
                    continue
                optim = self.optimizers[k]
                optim.zero_grad()

                label = torch.tensor(label).long().to(self.device)
                preds = self.probe(x, k)
                loss = self.loss_fn(preds, label)

                epoch_loss[k + "_loss"].append(loss.detach().item())
                accuracy[k + "_acc"].append(calculate_multiclass_accuracy(preds, label))
                if self.probes[k].training:
                    loss.backward()
                    optim.step()

        epoch_loss = {k: np.mean(loss) for k, loss in epoch_loss.items()}
        accuracy = {k: np.mean(acc) for k, acc in accuracy.items()}

        return epoch_loss, accuracy

    def train(self, episodes, label_dicts):
        inds = range(len(episodes))
        split_ind = int(0.8 * len(inds))
        tr_eps, val_eps = episodes[:split_ind], episodes[split_ind:]
        tr_labels, val_labels = label_dicts[:split_ind], label_dicts[split_ind:]
        for e in range(self.epochs):
            epoch_loss, accuracy = self.do_one_epoch(tr_eps, tr_labels)
            self.log_results(e, epoch_loss, accuracy)

            val_loss, val_accuracy = self.evaluate(val_eps, val_labels, epoch=e, prefix="val_")

            # update all early stoppers
            for k in self.info_dict.keys():
                if not self.early_stoppers[k].early_stop:
                    self.early_stoppers[k](val_loss["val_" + k + "_loss"], self.probes[k])

            # if all probes are done
            if np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()]):
                print("All probes early stopped!")
                break

            for k, scheduler in self.schedulers.items():
                scheduler.step(val_accuracy['val_' + k + '_acc'])

    def evaluate(self, test_episodes, test_label_dicts, epoch=None, prefix="test_"):
        if prefix == "test_":
            for k in self.early_stoppers.keys():
                self.early_stoppers[k].early_stop = False

        for k, probe in self.probes.items():
            probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(test_episodes, test_label_dicts)
        epoch_loss = {prefix + k: v for k, v in epoch_loss.items()}
        accuracy = {prefix + k: v for k, v in accuracy.items()}
        self.log_results(epoch, epoch_loss, accuracy)
        for k, probe in self.probes.items():
            probe.train()
        return epoch_loss, accuracy

    def log_results(self, epoch_idx, loss_dict, acc_dict):
        print("Epoch: {}".format(epoch_idx))
        for k in loss_dict.keys():
            print("\t {}: {:7.4f}".format(k, loss_dict[k]))
        print("\t --")
        for k in acc_dict.keys():
            print("\t {}: {:8.4f}%".format(k, 100 * acc_dict[k]))
        if self.wandb:
            self.wandb.log(loss_dict, step=epoch_idx)
            self.wandb.log(acc_dict, step=epoch_idx)
