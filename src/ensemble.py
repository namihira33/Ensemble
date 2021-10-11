import os
import sys
import time
from datetime import datetime
import random

from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
import pickle
import tensorboardX as tbx

from torchvision import models,transforms
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader 
from sklearn.model_selection import *

from utils  import *
import config
from network import *
from Dataset import load_dataloader

import csv
import matplotlib.pyplot as plt
import seaborn as sns

c = {
    'n_epoch': 1,'seed': [0], 'bs': [64], 'lr': [1e-5]
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

args = len(sys.argv)
if args >= 2:
    c['cv'] = int(sys.argv[1].split('=')[1])
    c['evaluate'] = int(sys.argv[2].split('=')[1])

class Ensemble():
    def __init__(self,c):
        self.c = c
        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))
        os.makedirs(self.log_path, exist_ok=True)
        self.net = MLP().to(device)
        self.net.apply(init_weights)
        self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
        self.criterion = nn.MSELoss()
        self.net = nn.DataParallel(self.net)
        self.dataloaders = load_ensebledata(self.c['bs'])

    def run():
        for epoch in range(1,c['n_epoch']+1):
            learningmae,learningloss,learningr_score\
                    = self.execute_epoch(epoch, 'learning')
            if not self.c['evaluate']:
                validmae,validloss,validr_score,valid_preds,valid_labels\
                    = self.execute_epoch(epoch, 'valid')
                #validmaeを蓄えておいてあとでベスト10を出力
                temp = validmae,epoch,self.c
                self.prms.append(temp)

        if not self.c['evaluate']:
            best_prms = sorted(self.prms,key=lambda x:x[0])
            with open(self.log_path + "/log.csv",'a') as f:
                writer = csv.writer(f)
                writer.writerow(['-'*20 + 'bestparameters' + '-'*20])
                writer.writerow(['model_name','lr','seed','n_epoch','auc'])
                writer.writerow(best_prms[0:10])
        #訓練後、モデルをセーブする。
        #(実行回数)_(モデル名)_(学習epoch).pth で保存。
        try : 
             model_name = 'ensemble'
             n_ep = self.c['n_epoch']
             n_ex = 0
             with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'r') as f:
                 n_ex = len(f.readlines())

             with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'a') as f:
                 writer = csv.writer(f)
                 writer.writerow([self.now,n_ex,model_name,n_ep])

             save_path = '{:0=2}'.format(n_ex)+ '_' + model_name + '_' + '{:0=3}'.format(n_ep)+'ep.pth'
             model_save_path = os.path.join(config.MODEL_DIR_PATH,save_path)
             torch.save(self.net.module.state_dict(),model_save_path)
        except FileNotFoundError:
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'w') as f:
                 writer = csv.writer(f)
                 writer.writerow(['Time','n_ex','Model_name','n_ep'])




    def execute_epoch(self, epoch, phase):
        preds, labels,total_loss= [], [],0
        if phase == 'learning':
            self.net.train()
        else:
            self.net.eval()

        for inputs_, labels_ in tqdm(self.dataloaders[phase]):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            self.optimizer.zero_grad()

            print(inputs_)
            print(labels_)


            with torch.set_grad_enabled(phase == 'learning'):
                outputs_ = self.net(inputs_)
                loss = self.criterion(outputs_,labels_)
                total_loss += loss.item()

                if phase == 'learning':
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        mae = mean_absolute_error(labels, preds)
        total_loss /= len(preds)
        r_score = r2_score(labels,preds)
        mse = mean_squared_error(labels,preds)

        print(
                f'epoch: {epoch} phase: {phase} loss: {total_loss:.3f} mae: {mae:.3f} mse: {mse:.3f} r_score{r_score:.3f}')

            
        result_list = [self.c['model_name'],self.c['lr'],self.c['seed'],epoch,phase,total_loss,mae]
            
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

        return (mae,total_loss,r_score) if (phase=='learning') else (mae,total_loss,r_score,preds,labels)

