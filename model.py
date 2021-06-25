import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
import pandas as pd
import glob
import seaborn as sns
import warnings
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from pathlib import Path
from tqdm.notebook import tqdm
import shutil
from typing import Tuple
import os
import wandb
import pytorch_lightning as pl
from sincnet import *

# This class contains base cnn implementation
# and SincNet implementation from original paper
class SincNet(pl.LightningModule):
    
    # Calculate shape after conv+pooling
    def get_shape(self, current_size, n_filters, filter_size, max_pool_size):
        return n_filters, int((current_size-filter_size+1)/max_pool_size)

    def __init__(self, config):
        super().__init__()
        
        # Basic params
        self.config = config
        n_class = config['n_class']
        input_shape = config['input_shape']
        sample_rate = config['sample_rate']
        self.use_wandb = config['use_wandb']
        
        # Use it for metrics accumulation
        if self.use_wandb:
            self.steps_num_val = 0
            self.steps_num_train = 0
            
            self.train_loss = 0
            self.val_loss = 0
            
            self.train_fer = 0
            self.val_fer = 0
        
        # LayerNorm shapes
        s1 = self.get_shape(input_shape[-1], 80, 251, 3)
        s2 = self.get_shape(s1[-1], 60, 5, 3)
        s3 = self.get_shape(s2[-1], 60, 5, 3)
        
        if config['first_conv'] == 'sinc':
            first_conv = SincConv_fast(80, 251, sample_rate)
        elif config['first_conv'] == 'base':
            first_conv = nn.Conv1d(1, 80, 251)
        else:
            raise NotImplementedError()
        
        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.LayerNorm(input_shape, eps=1e-6),
            first_conv,
            nn.MaxPool1d(3),
            nn.LayerNorm(s1, eps=1e-6),
            nn.ReLU(),
            # ----
            nn.Conv1d(80, 60, 5),
            nn.MaxPool1d(3),
            nn.LayerNorm(s2, eps=1e-6),
            nn.ReLU(),
            # ---
            nn.Conv1d(60, 60, 5),
            nn.MaxPool1d(3),
            nn.LayerNorm(s3, eps=1e-6),
            nn.ReLU(),
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(s3[0]*s3[1], 2048, bias=False),
            nn.BatchNorm1d(2048, momentum=0.05),
            nn.LeakyReLU(0.2),
            
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, momentum=0.05),

            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, momentum=0.05),
            nn.LeakyReLU(0.2),
            
            nn.Linear(2048, n_class)
        )

        # Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data):
        batch_size = data.shape[0]
        y = self.cnn(data)
        y = y.view(batch_size, -1)
        y = self.mlp(y)
        return y
    
    def training_step(self, batch, batch_idx):
        wf = batch['audios']
        target = batch['labels']
        preds = self(wf)
        loss = self.loss_fn(preds, target)
        
        # FER(%)
        fer = 100.0-torch.mean(1.0*(F.softmax(preds).argmax(axis=1).view(-1) == target))*100
        self.log('train_fer(%)', fer, prog_bar=True)
        
        # Do wandb logging if need
        if self.use_wandb:
            wandb.log({'train_loss/step': loss, 'train_fer(%)/step': fer})
            self.steps_num_train += 1
            self.train_loss += loss
            self.train_fer += fer
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        wf = batch['audios']
        target = batch['labels']
        preds = self(wf)
        loss = self.loss_fn(preds, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # FER(%)
        fer = 100.0-torch.mean(1.0*(F.softmax(preds).argmax(axis=1).view(-1) == target))*100
        self.log('val_fer(%)', fer, prog_bar=True)
        
        # Do wandb logging if need
        if self.use_wandb:
            wandb.log({'val_loss/step': loss, 'val_fer(%)/step': fer})
            self.steps_num_val += 1
            self.val_loss += loss
            self.val_fer += fer
            
        return {
            'loss':loss,
            'targets': target, 
            'preds': F.softmax(preds), 
            'sample_ids': batch['sample_ids']
            }
    
    # We need this for wandb every training epoch logging
    def training_epoch_end(self, training_step_outputs):
        if self.use_wandb:
            wandb.log({'train_loss/epoch': self.train_loss/self.steps_num_train, 
                       'train_fer(%)/epoch': self.train_fer/self.steps_num_train})
            self.steps_num_train = 0
            self.train_loss = 0
            self.train_fer = 0
    
    # We need this for CER calculation 
    # and wandb every validation epoch logging
    # Input: validation_step output
    def validation_epoch_end(self, validation_step_outputs):
        preds = None
        sample_ids = None
        target = None
        
        # Collect preds, sample_ids and targets
        # from all validation steps
        for out in validation_step_outputs:
            if preds is None:
                preds = out['preds']
                sample_ids = out['sample_ids']
                target = out['targets']
            else:
                preds = torch.cat((preds, out['preds']))
                sample_ids = torch.cat((sample_ids, out['sample_ids']))
                target = torch.cat((target, out['targets']))
        
        # Unique sample ids 
        unq_sample_ids = np.unique(sample_ids.cpu().numpy())
        
        # For each sample_id calculate
        # mean prediction vector over chunks
        # and calculate CER(%)
        cer = 0
        for id in unq_sample_ids:
            mask = sample_ids==id
            cer += int(preds[mask].mean(axis=0).argmax() == target[mask][0])
        cer = 100.0-(cer/len(unq_sample_ids))*100
        self.log('val_cer', cer, prog_bar=True)
        
        if self.use_wandb:
            wandb.log({'val_loss/epoch': self.val_loss/self.steps_num_val, 
                       'val_fer(%)/epoch': self.val_fer/self.steps_num_val, 
                       'val_cer': cer})
            self.steps_num_val = 0
            self.val_loss = 0
            self.val_fer = 0
    
    def configure_optimizers(self):
        if self.config['optimizer'] == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3, 
                                            alpha=0.95, eps=1e-7)
        elif self.config['optimizer'] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        elif self.config['optimizer'] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            raise NotImplementedError()
            
        return optimizer