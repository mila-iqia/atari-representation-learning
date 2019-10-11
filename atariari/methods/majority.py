from itertools import chain
from .utils import calculate_multiclass_accuracy, calculate_multiclass_f1_score
import numpy as np
import torch

def majority_baseline(tr_labels, test_labels, wandb):
    tr_labels = list(chain.from_iterable(tr_labels))
    test_labels = list(chain.from_iterable(test_labels))
    counts, maj_dict, test_counts = {}, {}, {}

    for label_dict in tr_labels:
        for k in label_dict:
            counts[k] = counts.get(k, {})
            v = label_dict[k]
            counts[k][v] = counts[k].get(v, 0) + 1

    # Get keys with maximum value
    for label in counts:
        maj_dict[label] = max(counts[label], key=counts[label].get)

    accuracy_dict = {}
    f1_score_dict = {}
    for k in test_labels[0]:
        labels = torch.tensor([label_d[k] for label_d in test_labels]).long()
        preds = torch.zeros(len(test_labels), 256)
        preds[:, maj_dict[k]] = 1
        accuracy = calculate_multiclass_accuracy(preds, labels)
        f1score = calculate_multiclass_f1_score(preds, labels)
        accuracy_dict[k + "_test_acc"] = accuracy
        f1_score_dict[k + "_f1score"] = f1score

    accuracy_dict['mean_test_acc'] = np.mean(list(accuracy_dict.values()))
    f1_score_dict["mean_f1score"] = np.mean(list(f1_score_dict.values()))
    wandb.log(accuracy_dict)
    wandb.log(f1_score_dict)
    return accuracy_dict, f1_score_dict