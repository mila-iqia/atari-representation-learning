# code from http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
#with some modificaitons

import torch
import numpy as np
import torch.nn as nn
import copy
import random
import torch
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from src.utils import calculate_accuracy
from src.trainer import Trainer
from src.utils import EarlyStopping
from torchvision import transforms

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    # mask can just be size of second to last dimension of key and it will be broadcasted
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / np.sqrt(d_k)


    if mask is not None:
        # where mask is 0, puts a really tiny number in the corresponding position in score
        # score is unnormalized prob dist over last dimension, which will be normalized by softmax
        scores = scores.masked_fill(mask == 0, -1e9)
    #last dim of p_attn sums to 1
    p_attn = torch.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)        

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.fc_q, self.fc_k, self.fc_v, self.fc_final = [nn.Linear(d_model, d_model) for i in range(4)]
        self.qkv_linears = [self.fc_q, self.fc_k, self.fc_v]
        self.attn = None
        self.dropout = None #nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # mask should be (batch_size,seq_length,seq_length)
        # and last dimension should encode which index of seq to mask out
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.fc_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.fc_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.fc_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        
        
        
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.fc_final(x)

class TransformerUnit(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, h, d_model, dropout):
        super().__init__()
        self.mha = MultiHeadedAttention(h,d_model)
        self.fc = nn.Linear(d_model,d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        v = self.norm(x + self.dropout(self.mha(x,x,x,mask)))
        output = self.norm(v + self.dropout(self.fc(v)))
        return output

class Transformer(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self,N=1, h=8, d_model=256, seq_len=10, dropout=0.1):
        super().__init__()
        tu =  TransformerUnit(h, d_model, dropout)
        self.layers = clones(tu, N)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask):
        """Mask should be batch_size, seq_len"""
        batch_size, seq_len = x.shape[0], x.shape[1]

        mask = mask.repeat((1,seq_len)).reshape(batch_size,seq_len,seq_len)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        
        #converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size] by simply taking the hidden state corresponding
        # to the first token.
        f = x[:,0,:]
        #f = self.fc(f)
        return f
        
        
class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)
    


class BERTTrainer(Trainer):
    # TODO: Make it work for all modes, right now only it defaults to pcl.
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.mode = config['mode']
        self.patience = self.config["patience"]
        self.seq_len = self.config["seq_len"]
        self.hidden_size = self.encoder.hidden_size
        self.N =  self.config["num_transformer_layers"]
        self.h = self.config["num_lin_projections"]
        self.dropout=self.config["dropout"]

        self.transformer = Transformer(N=self.N,h=self.h,
                                       d_model=self.hidden_size,
                                       seq_len=self.seq_len,
                                       dropout=self.dropout)

      
        self.classifier = Classifier(self.hidden_size, self.hidden_size ).to(device)  # x1 = global, x2=patch, n_channels = 32
        #self.classifier2 = Classifier(128, 128).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.time_window = config["time_window"]
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier.parameters())\
                                          + list(self.encoder.parameters())\
                                          + list(self.transformer.parameters()),\
                                         lr=config['lr'], eps=1e-5)
                                          #+list(self.classifier2.parameters()),

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_seq, x_neg, x_pos, masks = [], [], [], []
            tw = np.asarray(self.time_window)
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_neg = np.random.randint(0, len(episode)-self.seq_len), np.random.randint(0, len(episode))
                t_pos = np.random.randint(t,t+self.seq_len,size=(1,1))
                x_seq.append(episode[t:t+self.seq_len])
                x_neg.append(episode[t_neg])
                x_pos.append(episode[int(t_pos)])
                mask_ind = torch.from_numpy(t_pos % self.seq_len)
                base_mask = torch.zeros((1, self.seq_len))
                mask = base_mask.scatter_(1,mask_ind,1).squeeze()
                masks.append(mask)
             

            yield torch.stack(x_seq).to(self.device) / 255.,\
                  torch.stack(x_neg).to(self.device) / 255.,\
                  torch.stack(x_pos).to(self.device) / 255.,\
                  torch.stack(masks).to(self.device)

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.classifier.training and self.transformer.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        data_generator = self.generate_batch(episodes)
        for x_seq, x_neg, x_pos, masks in data_generator:
            # flatten [batch_size,seq_len,n_ch,x,y] into [batch_size*seq_len,n_ch,x,y]
            # so we it fits nicely into encoder, then reshape it back to sequence
            x_seq_batch = x_seq.reshape(self.batch_size * self.seq_len, *x_pos.shape[1:])
            f_seq = self.encoder(x_seq_batch).reshape(self.batch_size, self.seq_len, -1)
            f_pos, f_neg = self.encoder(x_pos), self.encoder(x_neg)
            print(masks.shape)
            f_tf = self.transformer(f_seq, masks)
            
            target = torch.cat((torch.ones(self.batch_size, 1),
                                torch.zeros(self.batch_size, 1)), dim=0).to(self.device)

            x1, x2 = torch.cat([f_tf,f_tf], dim=0), torch.cat([f_pos, f_neg], dim=0)
            shuffled_idxs = torch.randperm(len(target))
            x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.classifier(x1, x2), target)
            
            if mode == "train":
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            preds = torch.sigmoid(self.classifier(x1, x2))
            accuracy += calculate_accuracy(preds, target)
            steps += 1
        self.log_results(epoch, epoch_loss / steps, accuracy / steps, prefix=mode )
        if mode == "val":
            self.early_stopper(accuracy, self.encoder)

    def train(self, tr_eps, val_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(), self.classifier.train(), self.transformer.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval(), self.classifier.eval(), self.transformer.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss, accuracy, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {} Accuracy: {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize(), accuracy))
        self.wandb.log({prefix + '_loss': epoch_loss, prefix + '_accuracy': accuracy})
