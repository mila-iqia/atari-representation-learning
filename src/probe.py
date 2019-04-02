import torch

from torch import nn
from src.trainer import Trainer
from src.utils import appendabledict
from src.utils import calculate_multiclass_accuracy

import numpy as np
from torch.utils.data import RandomSampler, BatchSampler

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_classes)
    
    def forward(self, feature_vectors):
        return self.model(feature_vectors)

class ProbeTrainer(Trainer):
    def __init__(self,encoder, wandb, info_dict, 
                 device=torch.device('cpu'), 
                 epochs=100, lr=5e-4, mini_batch_size=64):
        super().__init__(encoder, wandb, device)
        self.info_dict = info_dict
        self.epochs = epochs
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        # info_dict should have {label_name: number_of_classes_for_that_label}
        self.probes = {k: LinearProbe(input_dim=encoder.hidden_size,
                                      num_classes=info_dict[k]) for k in info_dict.keys()}
        self.optimizers = {k: torch.optim.Adam(list(self.probes[k].parameters()), 
                                               eps=1e-5, lr=self.lr ) for k in info_dict.keys()}
        self.loss_fn = nn.CrossEntropyLoss()
        
    def generate_batch(self, episodes, episode_labels):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.mini_batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.mini_batch_size, drop_last=True)  
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            episode_labels_batch = [episode_labels[x] for x in indices]
            xs, labels = [], appendabledict()
            for ep_ind, episode in enumerate(episodes_batch):
                # Get one sample from this episode
                t = np.random.randint(len(episode))
                xs.append(episode[t])
                labels.append_update(episode_labels_batch[ep_ind][t])
            yield torch.stack(xs) / 255., labels

    
    
    def do_one_epoch(self, episodes, label_dicts, ):
        epoch_loss, accuracy = {k + "_loss":0 for k in self.info_dict.keys()},\
                              {k + "_acc" :0 for k in self.info_dict.keys()}
        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            f = self.encoder(x).detach()
            for k, label in labels_batch.items():
                probe = self.probes[k]
                optim = self.optimizers[k]
                optim.zero_grad()
                
                label = torch.tensor(label).long().to(self.device)
                preds = probe(f)
                try:
                    loss = self.loss_fn(preds, label)
                except:
                    print(label)
                    
    
                epoch_loss[k + "_loss"] += loss.detach().item()   
                accuracy[k + "_acc"] += calculate_multiclass_accuracy(preds, label)          
                if probe.training:
                    loss.backward()
                    optim.step()
        epoch_loss = {k: loss / (step + 1) for k, loss in epoch_loss.items()}
        accuracy = {k: acc / (step + 1) for k, acc in accuracy.items()}
        return epoch_loss, accuracy
        
    def train(self, episodes, label_dicts):
        for e in range(self.epochs):
            epoch_loss, accuracy = self.do_one_epoch(episodes, label_dicts)
            self.log_results(e,epoch_loss,accuracy)
            
    def test(self, test_episodes, test_label_dicts):
        for k,probe in self.probes.items():
            probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(test_episodes, test_label_dicts)
        epoch_loss = {"test_" + k:v for k,v in epoch_loss.items()}
        accuracy = {"test_" + k:v for k,v in accuracy.items()}
        self.log_results(0,epoch_loss,accuracy)
        
           
    def log_results(self, epoch_idx, loss_dict, acc_dict):
        print("Epoch: {}".format(epoch_idx))
        for k in loss_dict.keys():
            print("\t {}: {}".format(k, loss_dict[k]))
        print("\t --")
        for k in acc_dict.keys():
            print("\t {}: {}%".format(k, 100*acc_dict[k]))
                  
        self.wandb.log(loss_dict)
        self.wandb.log(acc_dict)
        
        