import torch
from torch import nn
from .utils import EarlyStopping, appendabledict, \
    calculate_multiclass_accuracy, calculate_multiclass_f1_score,\
    append_suffix, compute_dict_average

from copy import deepcopy
import numpy as np
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


def train_all_probes(encoder, tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels, args, wandb):
    acc_dict, f1_dict = {}, {}
    for label_name in tr_labels.keys():
        trainer = ProbeTrainer(encoder=encoder,
                   epochs=args.epochs,
                   method_name=args.method,
                   lr=args.probe_lr,
                   batch_size=args.batch_size,
                   patience=args.patience,
                   wandb=wandb,
                   fully_supervised=(args.method == "supervised"),
                   save_dir=wandb.run.dir)

        trainer.train(tr_eps, val_eps, tr_labels[label_name], val_labels[label_name])
        test_acc, test_f1score = trainer.test(test_eps, test_labels[label_name])
        acc_dict[label_name] = test_acc
        f1_dict[label_name] = test_f1score




    acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

    print("""In our paper, we report F1 scores and accuracies averaged across each category. 
          That is, we take a mean across all state variables in a category to get the average score for that category.
          Then we average all the category averages to get the final score that we report per game for each method. 
          These scores are called \'across_categories_avg_acc\' and \'across_categories_avg_f1\' respectively
          We do this to prevent categories with large number of state variables dominating the mean F1 score.
          """)
    return acc_dict, f1_dict



class ProbeTrainer(object):
    def __init__(self,
                 encoder = None,
                 method_name = "my_method",
                 wandb = None,
                 patience = 15,
                 num_classes = 256,
                 fully_supervised = False,
                 save_dir = ".models",
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr = 5e-4,
                 epochs = 100,
                 batch_size = 64,
                 representation_len = 256) -> object:

        self.encoder = encoder
        self.wandb = wandb
        self.device = device
        self.fully_supervised = fully_supervised
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = method_name
        self.feature_size = representation_len
        self.loss_fn = nn.CrossEntropyLoss()
        if self.encoder == None:
            self.vector_input = True
        else:
            self.vector_input = False

        if self.fully_supervised:
            self.probe = FullySupervisedProbe(self.encoder)
        else:
            self.probe = LinearProbe(input_dim=self.feature_size, num_classes=self.num_classes)

        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, save_dir=self.save_dir)
        self.optimizer = torch.optim.Adam(list(self.probe.parameters()),
                                               eps=1e-5, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.2, verbose=True, mode='max', min_lr=1e-5)



    def generate_batch(self, frames, labels, batch_size):
        labels_tensor = torch.tensor(labels).long()
        ds = TensorDataset(frames, labels_tensor)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for x, y in dl:
            yield x.float().to(self.device) / 255., y.to(self.device)


    def do_probe(self, x):
        if self.encoder is None:
            # x is a set vectors
            preds = self.probe(x)
        else:
            preds = self.probe(self.encoder(x).detach())
        return preds


    def do_one_epoch(self, episodes, labels):
        losses, accuracies = [], []

        data_generator = self.generate_batch(episodes, labels, batch_size=self.batch_size)

        for step, (x, label) in enumerate(data_generator):
            self.optimizer.zero_grad()
            preds = self.do_probe(x)
            loss = self.loss_fn(preds, label)

            losses.append(loss.detach().item())
            preds = preds.cpu().detach().numpy()
            preds = np.argmax(preds, axis=1)
            label = label.cpu().detach().numpy()
            accuracies.append(calculate_multiclass_accuracy(preds, label))
            if self.probe.training:
                loss.backward()
                self.optimizer.step()

        epoch_loss = np.mean(losses)
        accuracy = np.mean(accuracies)



        return epoch_loss, accuracy

    def do_test_epoch(self, episodes, labels):

        x, y = next(self.generate_batch(episodes, labels, batch_size=episodes.shape[0]))
        preds = self.do_probe(x)
        y = y.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        accuracy = calculate_multiclass_accuracy(preds, y)
        f1score = calculate_multiclass_f1_score(preds, y)

        return accuracy, f1score


    def evaluate(self, val_episodes, val_label_dicts):
        self.probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(val_episodes, val_label_dicts)
        self.probe.train()
        return epoch_loss, accuracy

    def train(self, tr_eps, val_eps, tr_labels, val_labels):
        epoch = 0
        while (not self.early_stopper.early_stop) and epoch < self.epochs:
            self.probe.train()
            epoch_loss, accuracy = self.do_one_epoch(tr_eps, tr_labels)
            val_loss, val_accuracy = self.evaluate(val_eps, val_labels)
            self.log_results(epoch, dict(tr_loss=epoch_loss, tr_accuracy=accuracy, val_loss=val_loss, val_accuracy=val_accuracy))

            # update all early stoppers
            if not self.early_stopper.early_stop:
                self.early_stopper(val_accuracy, self.probe)
                self.scheduler.step(val_accuracy)
            epoch += 1
        print("Probe early stopped!")

    def test(self, test_episodes, test_label_dicts, epoch=None):
        self.probe.eval()
        acc, f1 = self.do_test_epoch(test_episodes, test_label_dicts)
        return acc, f1

    def log_results(self, epoch_idx, *dictionaries):
        print("Epoch: {}".format(epoch_idx))
        for dictionary in dictionaries:
            for k, v in dictionary.items():
                print("\t {}: {:8.4f}".format(k, v))
            print("\t --")


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



