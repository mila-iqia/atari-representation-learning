from collections import defaultdict
import torch
from sklearn.metrics import f1_score as compute_f1_score
from pathlib import Path
import numpy as np


def compute_dict_average(metric_dict):
    return np.mean(list(metric_dict.values()))


def append_suffix(dictionary, suffix):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k + suffix] = v
    return new_dict


def download_run(env_name, checkpoint_step):
    import wandb
    api = wandb.Api()
    runs = list(api.runs("curl-atari/pretrained-rl-agents-2", {"state": "finished",
                                                               "config.env_name": env_name}))
    run = runs[0]
    filename = env_name + '_' + str(checkpoint_step) + '.pt'
    run.files(names=[filename])[0].download(root='./trained_models_full/', replace=True)
    print('Downloaded ' + filename)
    return './trained_models_full/' + filename


class appendabledict(defaultdict):
    def __init__(self, type_=list, *args, **kwargs):
        self.type_ = type_
        super().__init__(type_, *args, **kwargs)

    #     def map_(self, func):
    #         for k, v in self.items():
    #             self.__setitem__(k, func(v))

    def subslice(self, slice_):
        """indexes every value in the dict according to a specified slice

        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.


        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}

         """
        sliced_dict = {}
        for k, v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict

    def append_update(self, other_dict):
        """appends current dict's values with values from other_dict

        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to append to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

         """
        for k, v in other_dict.items():
            self.__getitem__(k).append(v)


# Thanks Bjarten! (https://github.com/Bjarten/early-stopping-pytorch)
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, name="", checkpoint=False, save_dir=".models"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.
        self.name = name
        self.checkpoint = checkpoint
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            if self.checkpoint:
                self.save_checkpoint(val_acc, model)
            self.val_acc_max = val_acc
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping for {self.name} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'{self.name} has stopped')

        else:
            self.best_score = score
            if self.checkpoint:
                self.save_checkpoint(val_acc, model)
            self.val_acc_max = val_acc
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation accuracy increased for {self.name}  ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')

        torch.save(model.state_dict(), str(self.save_dir) + "/" + self.name + ".pt")


def calculate_accuracy(preds, y):
    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()
    return acc


def calculate_multiclass_f1_score(preds, labels):
    f1score = compute_f1_score(labels, preds, average="weighted")
    return f1score


def calculate_multiclass_accuracy(preds, labels):
    acc = float(np.sum((preds == labels).astype(int)) / len(labels))
    return acc