import torch
from torch import nn
from .utils import EarlyStopping, appendabledict, \
    calculate_multiclass_accuracy, calculate_multiclass_f1_score,\
    append_suffix, compute_dict_average, calculate_multiple_accuracies, calculate_multiple_f1_scores

from copy import deepcopy
import numpy as np
import sys
from torch.utils.data import RandomSampler, BatchSampler
from .categorization import summary_key_dict
from torch.utils.data import DataLoader, TensorDataset



class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_classes)

    def forward(self, feature_vectors):
        return self.model(feature_vectors)


class FullySupervisedProbe(nn.Module):
    def __init__(self, encoder, num_classes=255):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.probe = LinearProbe(input_dim=self.encoder.hidden_size,
                                 num_classes=num_classes)

    def forward(self, x):
        feature_vec = self.encoder(x)
        return self.probe(feature_vec)


def train_all_probes(encoder, tr_eps, val_eps, test_eps, tr_labels, val_labels, test_labels,lr, representation_len, args, save_dir):
    trainer = ProbeTrainer(encoder=encoder,
                           epochs=args.epochs,
                           lr=lr,
                           batch_size=args.batch_size,
                           num_state_variables=len(tr_labels.keys()),
                           patience=args.patience,
                           fully_supervised=(args.method == "supervised"),
                           save_dir=save_dir,
                           representation_len=representation_len)

    trainer.train(tr_eps, val_eps, tr_labels, val_labels)
    test_acc, test_f1score = trainer.test(test_eps, test_labels)

    return test_acc, test_f1score



class ProbeTrainer(object):
    def __init__(self,
                 encoder = None,
                 method_name = "my_method",
                 patience = 15,
                 num_classes = 256,
                 num_state_variables=8,
                 fully_supervised = False,
                 save_dir = ".models",
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr = 5e-4,
                 epochs = 100,
                 batch_size = 64,
                 representation_len = 256):

        self.encoder = encoder
        self.num_state_variables = num_state_variables
        self.device = device
        self.fully_supervised = fully_supervised
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = method_name
        self.representation_len = representation_len
        if self.encoder == None:
            self.vector_input = True
        else:
            self.vector_input = False

        self.probe = nn.Linear(self.representation_len, self.num_classes * self.num_state_variables).to(self.device)

        # self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, save_dir=self.save_dir)
        self.optimizer = torch.optim.Adam(list(self.probe.parameters()),
                                               eps=1e-5, lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.2, verbose=True, mode='max', min_lr=1e-5)



    def generate_batch(self, frames, labels_dict, batch_size):
        labels = torch.tensor(list(labels_dict.values())).long()
        labels_tensor = labels.transpose(1, 0)
        ds = TensorDataset(frames, labels_tensor)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for x, y in dl:
            yield x.float().to(self.device) / 255., y.to(self.device)

    def do_probe(self, x):
        if self.vector_input:
            vectors = x
        elif self.fully_supervised:
            vectors = self.encoder(x)
        else:
            vectors = self.encoder(x).detach()

        batch_size, *rest = vectors.shape
        preds = self.probe(vectors)
        preds = preds.reshape(batch_size, -1, self.num_state_variables, self.num_classes)
        preds = preds.transpose(1, 3)

        return preds.squeeze()


    def do_one_epoch(self, episodes, labels_dict):
        losses = []
        all_preds = []
        all_labels = []

        data_generator = self.generate_batch(episodes, labels_dict, batch_size=self.batch_size)

        for x, labels in data_generator:
            preds = self.do_probe(x)
            loss = nn.CrossEntropyLoss()(preds, labels)
            if self.probe.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            preds = preds.cpu().detach().numpy()
            preds = np.argmax(preds, axis=1)
            all_preds.append(preds)
            labels = labels.cpu().detach().numpy()
            all_labels.append(labels)
            losses.append(loss.detach().item())

        epoch_loss = np.mean(losses)

        alab = np.concatenate(all_labels)
        ap = np.concatenate(all_preds)
        accuracies = calculate_multiple_accuracies(alab, ap)
        f1_scores = calculate_multiple_f1_scores(alab, ap)

        return epoch_loss, accuracies, f1_scores



    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        epoch = 0
        while epoch < self.epochs:
            self.probe.train()
            epoch_loss, accuracy, _ = self.do_one_epoch(tr_eps, tr_labels)
            self.probe.eval()
            val_loss, val_accuracy, _ = self.do_one_epoch(val_eps, val_labels)
            #val_accuracy = val_accuracy
            #tr_accuracy = accuracy,
            self.log_results(epoch, tr_loss=epoch_loss, val_loss=val_loss)

            epoch += 1
        sys.stderr.write("Probe done!\n")

    def test(self, test_episodes, test_label_dicts):
        self.probe.eval()
        _, acc, f1 = self.do_one_epoch(test_episodes, test_label_dicts)
        return acc, f1

    def log_results(self, epoch_idx, **kwargs):
        sys.stderr.write("Epoch: {}\n".format(epoch_idx))
        for k, v in kwargs.items():
            if isinstance(v, dict):
                sys.stderr.write("\t {}:\n".format(k))
                for kk, vv in v.items():
                    sys.stderr.write("\t\t {}: {:8.4f}\n".format(kk, vv))
                sys.stderr.write("\t ------\n")

            else:
                sys.stderr.write("\t {}: {:8.4f}\n".format(k, v))



def postprocess_raw_metrics(acc_dict, f1_dict):
    acc_overall_avg, f1_overall_avg = compute_dict_average(acc_dict), \
                                      compute_dict_average(f1_dict)
    acc_category_avgs_dict, f1_category_avgs_dict = compute_category_avgs(acc_dict), \
                                                    compute_category_avgs(f1_dict)
    acc_avg_across_categories, f1_avg_across_categories = compute_dict_average(acc_category_avgs_dict), \
                                                          compute_dict_average(f1_category_avgs_dict)
    acc_dict.update(acc_category_avgs_dict)
    f1_dict.update(f1_category_avgs_dict)

    acc_dict["overall_avg"], f1_dict["overall_avg"] = acc_overall_avg, f1_overall_avg
    acc_dict["across_categories_avg"], f1_dict["across_categories_avg"] = [acc_avg_across_categories,
                                                                           f1_avg_across_categories]

    acc_dict = append_suffix(acc_dict, "_acc")
    f1_dict = append_suffix(f1_dict, "_f1")

    return acc_dict, f1_dict


def compute_category_avgs(metric_dict):
    category_dict = {}
    for category_name, category_keys in summary_key_dict.items():
        category_values = [v for k, v in metric_dict.items() if k in category_keys]
        if len(category_values) < 1:
            continue
        category_mean = np.mean(category_values)
        category_dict[category_name + "_avg"] = category_mean
    return category_dict



