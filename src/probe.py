import torch
from src.utils import EarlyStopping
from torch import nn
from src.trainer import Trainer
from src.utils import appendabledict
from src.utils import calculate_multiclass_accuracy, calculate_multiclass_f1_score
from copy import deepcopy
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from src.encoders import Flatten


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_classes)

    def forward(self, feature_vectors):
        return self.model(feature_vectors)


class NonLinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        )

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
    def __init__(self, encoder, wandb, sample_label,
                 device=torch.device('cpu'),
                 epochs=100, lr=5e-4,
                 batch_size=64,
                 patience=15, log=True, feature_size=None):
        super().__init__(encoder, wandb, device)
        self.sample_label = sample_label
        self.num_classes = 256
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = wandb.config["method"]
        if feature_size is None:
            self.feature_size = 512 if self.method == "pretrained-rl-agent" else encoder.hidden_size
        else:
            self.feature_size = feature_size
        self.log = log
        if self.method == "supervised":
            self.probes = {k: FullySupervisedLinearProbe(encoder=self.encoder,
                                                         num_classes=self.num_classes).to(device) for k in
                           sample_label.keys()}
        elif self.method == 'nonlinear':
            self.probes = {k: NonLinearProbe(input_dim=self.feature_size,
                                             num_classes=self.num_classes).to(device) for k in sample_label.keys()}
        else:
            # sample_label should have {label_name: number_of_classes_for_that_label}
            self.probes = {k: LinearProbe(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(device) for k in sample_label.keys()}

        self.early_stoppers = {k: EarlyStopping(patience=patience, verbose=False, wandb=self.wandb, name=k + "_probe")
                               for k in
                               sample_label.keys()}

        self.optimizers = {k: torch.optim.Adam(list(self.probes[k].parameters()),
                                               eps=1e-5, lr=self.lr) for k in sample_label.keys()}
        self.schedulers = {k: torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers[k], patience=5, factor=0.2, verbose=True, mode='max', min_lr=1e-5) for k in
            sample_label.keys()}
        self.loss_fn = nn.CrossEntropyLoss()

    def generate_batch(self, episodes, episode_labels):
        total_steps = sum([len(e) for e in episodes])
        assert total_steps > self.batch_size
        print('Total Steps: {}'.format(total_steps))
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
        if self.method == "flat-pixels":
            batch = Flatten()(batch)
        if self.method in ["supervised", "pretrained-rl-agent", "flat-pixels"]:
            '''#if method is supervised batch is a batch of frames and probe is a full encoder + linear or nonlinear probe
                if method is pretrained-rl-agent, then batch is a batch of feature vectors and probe is just a linear or nonlinear probe
                if method is flat-pixel, then batch is a batch of flattened raw images and the linear probe is a really wide linear probe'''
            preds = probe(batch)

        else:
            with torch.no_grad():
                self.encoder.to(self.device)
                f = self.encoder(batch).detach()
            preds = probe(f)
        return preds

    def do_one_epoch(self, episodes, label_dicts):
        epoch_loss, accuracy = {k + "_loss": [] for k in self.sample_label.keys() if
                                not self.early_stoppers[k].early_stop}, \
                               {k + "_acc": [] for k in self.sample_label.keys() if
                                not self.early_stoppers[k].early_stop}

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
    
    
    def do_test_epoch(self, episodes, label_dicts):
        accuracy_dict, f1_score_dict = {},{} 
        pred_dict, all_label_dict = {k: [] for k in self.sample_label.keys()},\
                                 {k: [] for k in self.sample_label.keys()},\
                                

        # collect all predicitons first
        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                label = torch.tensor(label).long().cpu()
                all_label_dict[k].append(label)
                preds = self.probe(x, k).detach().cpu()
                pred_dict[k].append(preds)
        
                

        for k in all_label_dict.keys():
            preds, labels = torch.cat(pred_dict[k]), torch.cat(all_label_dict[k])
            
            accuracy = calculate_multiclass_accuracy(preds, labels)
            f1score = calculate_multiclass_f1_score(preds, labels)
            accuracy_dict[k + "_test_acc"] = accuracy
            f1_score_dict[k + "_f1score"] = f1score
            
        return accuracy_dict, f1_score_dict

    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        e = 0
        all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        while (not all_probes_stopped) and e < 100:
            epoch_loss, accuracy = self.do_one_epoch(tr_eps, tr_labels)
            self.log_results(e, epoch_loss, accuracy)

            val_loss, val_accuracy = self.evaluate(val_eps, val_labels, epoch=e)
            # update all early stoppers
            for k in self.sample_label.keys():
                if not self.early_stoppers[k].early_stop:
                    self.early_stoppers[k](val_accuracy["val_" + k + "_acc"], self.probes[k])

            for k, scheduler in self.schedulers.items():
                if not self.early_stoppers[k].early_stop:
                    scheduler.step(val_accuracy['val_' + k + '_acc'])
            e += 1
            all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        print("All probes early stopped!")

    def evaluate(self, val_episodes, val_label_dicts, epoch=None):
        for k, probe in self.probes.items():
            probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(val_episodes, val_label_dicts)
        epoch_loss = {"val_" + k: v for k, v in epoch_loss.items()}
        accuracy = {"val_" + k: v for k, v in accuracy.items()}
        self.log_results(epoch, epoch_loss, accuracy)
        for k, probe in self.probes.items():
            probe.train()
        return epoch_loss, accuracy

    def test(self, test_episodes, test_label_dicts, epoch=None):
        for k in self.early_stoppers.keys():
            self.early_stoppers[k].early_stop = False
        for k, probe in self.probes.items():
            probe.eval()
        accuracy_dict, f1_score_dict = self.do_test_epoch(test_episodes, test_label_dicts)
        accuracy_dict['mean_test_acc'] = np.mean(list(accuracy_dict.values()))      
        f1_score_dict["mean_f1score"] = np.mean(list(f1_score_dict.values()))     
        self.wandb.log(f1_score_dict)
        self.wandb.log(accuracy_dict)
        print("F1 scores")
        for k in f1_score_dict.keys():
            print("\t  {}: {:8.4f}".format(k, f1_score_dict[k]))
        print("\t --")
        print("Accuracy")
        for k in accuracy_dict.keys():
            print("\t {}: {:8.4f}%".format(k, 100 * accuracy_dict[k]))
        return accuracy_dict, f1_score_dict
            
            
    def log_results(self, epoch_idx, loss_dict, acc_dict):
        print("Epoch: {}".format(epoch_idx))
        for k in loss_dict.keys():
            print("\t {}: {:7.4f}".format(k, loss_dict[k]))
        print("\t --")
        for k in acc_dict.keys():
            print("\t {}: {:8.4f}%".format(k, 100 * acc_dict[k]))

