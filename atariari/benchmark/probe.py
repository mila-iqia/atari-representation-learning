import torch
from torch import nn
from .utils import append_suffix, compute_dict_average, calculate_multiple_accuracies, calculate_multiple_f1_scores

from copy import deepcopy
import numpy as np
import sys
from .categorization import summary_key_dict

class ProbeTrainer(object):
    def __init__(self,
                 encoder = None,
                 method_name = "my_method",
                 patience = 15,
                 num_classes = 256,
                 num_state_variables=8,
                 fully_supervised = False,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr = 5e-4,
                 epochs = 100,
                 batch_size = 64,
                 representation_len = 256):

        self.encoder = encoder
        self.num_state_variables = num_state_variables
        self.device = device
        self.fully_supervised = fully_supervised
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

        return preds


    def do_one_epoch(self, dataloader):
        losses = []
        all_preds = []
        all_labels = []
        for x,y in dataloader:
            frames = x.float().to(self.device) / 255.
            labels = y.long().to(self.device)
            preds = self.do_probe(frames)
            lbls = labels[:, :, None].repeat(1, 1, preds.shape[3]).squeeze() # for if preds has another dimension for slot encoders for example
            loss = nn.CrossEntropyLoss()(preds.squeeze(), lbls)
            if self.probe.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            preds = preds.cpu().detach().numpy().squeeze()
            preds = np.argmax(preds, axis=1).reshape(preds.shape[0], -1)
            all_preds.append(preds)
            lbls = lbls.cpu().detach().numpy()
            all_labels.append(lbls.reshape(lbls.shape[0], -1))
            losses.append(loss.detach().item())

        epoch_loss = np.mean(losses)

        labels_tensor = np.concatenate(all_labels)
        preds_tensor = np.concatenate(all_preds)
        accuracies = calculate_multiple_accuracies(preds_tensor, labels_tensor)
        f1_scores = calculate_multiple_f1_scores(preds_tensor, labels_tensor)

        return epoch_loss, accuracies, f1_scores



    def train(self, tr_dl, val_dl):
        epoch = 0
        while epoch < self.epochs:
            self.probe.train()
            epoch_loss, accuracy, _ = self.do_one_epoch(tr_dl)
            self.probe.eval()
            val_loss, val_accuracy, _ = self.do_one_epoch(val_dl)
            self.log_results(epoch, tr_loss=epoch_loss, val_loss=val_loss)

            epoch += 1
        sys.stderr.write("Probe done!\n")

    def test(self, test_dl):
        self.probe.eval()
        _, acc, f1 = self.do_one_epoch(test_dl)
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



