from itertools import chain
import torch
import wandb


def remove_duplicates(tr_eps, val_eps, test_eps, test_labels):
    """
    Remove any items in test_eps (&test_labels) which are present in tr/val_eps
    """
    flat_tr = list(chain.from_iterable(tr_eps))
    flat_val = list(chain.from_iterable(val_eps))
    tr_val_set = set([x.numpy().tostring() for x in flat_tr] + [x.numpy().tostring() for x in flat_val])
    flat_test = list(chain.from_iterable(test_eps))

    for i, episode in enumerate(test_eps[:]):
        test_labels[i] = [label for obs, label in zip(test_eps[i], test_labels[i]) if obs.numpy().tostring() not in tr_val_set]
        test_eps[i] = [obs for obs in episode if obs.numpy().tostring() not in tr_val_set]
    test_len = len(list(chain.from_iterable(test_eps)))
    dups = len(flat_test) - test_len
    print('Duplicates: {}, Test Len: {}'.format(dups, test_len))
    #wandb.log({'Duplicates': dups, 'Test Len': test_len})
    return test_eps, test_labels


def remove_low_entropy_labels(episode_labels, entropy_threshold=0.6):
    flat_label_list = list(chain.from_iterable(episode_labels))
    counts = {}

    for label_dict in flat_label_list:
        for k in label_dict:
            counts[k] = counts.get(k, {})
            v = label_dict[k]
            counts[k][v] = counts[k].get(v, 0) + 1
    low_entropy_labels = []

    entropy_dict = {}
    for k in counts:
        entropy = torch.distributions.Categorical(
            torch.tensor([x / len(flat_label_list) for x in counts[k].values()])).entropy()
        entropy_dict['entropy_' + k] = entropy
        if entropy < entropy_threshold:
            print("Deleting {} for being too low in entropy! Sorry, dood!".format(k))
            low_entropy_labels.append(k)

    for e in episode_labels:
        for obs in e:
            for key in low_entropy_labels:
                del obs[key]
    # wandb.log(entropy_dict)
    return episode_labels, entropy_dict
