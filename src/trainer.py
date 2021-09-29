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
from sklearn.model_selection import KFold

from utils  import *
import config
from network import *
from Dataset import load_dataloader

import csv
import matplotlib.pyplot as plt

#import pandas as pd


torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

class Trainer():
    def __init__(self, c):
        self.dataloaders = {}
        self.search = c
        self.n_seeds = len(c['seed'])
        self.n_splits = 5        
        self.loss,self.mae,self.mse,self.r_score= {},{},{},{}
        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))
        os.makedirs(self.log_path, exist_ok=True)
        self.tb_writer = tbx.SummaryWriter()

        with open(self.log_path + "/log.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['-'*20 + 'Log File' + '-'*20])

        

    def run(self):
        #実行時間計測とmae代入準備
        start = time.time()
        min_mae = 100000.0
        n_iter = 1

        #Initialization -> Score
        for phase in ['learning','valid']:
            self.loss[phase] = 0
            self.mae[phase] = 0
            self.r_score[phase] = 0
            self.mse[phase] = 0

        #CSVファイルヘッダー記述
        with open(self.log_path + "/log.csv",'a') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name','lr','seed','epoch','phase','total_loss','mae'])

        for c,param in iterate(self.search):
            print('Parameter :',c)
            self.c = c
            random.seed(self.c['seed'])
            torch.manual_seed(self.c['seed'])

            mn = self.c['model_name']

            if mn == 'Vgg16':
                self.net = Vgg16()
            elif mn == 'Vgg16-bn':
                self.net = Vgg16_bn()
            elif mn == 'Vgg19':
                self.net = Vgg19()
            elif mn == 'Vgg19-bn':
                self.net = Vgg19_bn()
            elif mn == 'Resnet18':
                self.net = Resnet18()
            elif mn == 'Resnet34':
                self.net = Resnet34()
            elif mn == 'Resnet50':
                self.net = Resnet50()
            elif mn == 'Squeezenet':
                self.net = Squeezenet()
            elif mn == 'Densenet':
                self.net = Densenet()
            elif mn == 'Inception':
                self.net = Inception()
            elif mn == 'Mobilenet-large':
                self.net = Mobilenet_large()
            elif mn == 'Mobilenet-small':
                self.net = Mobilenet_small()
            
            self.net = self.net.to(device)
            self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
            self.criterion = nn.MSELoss()
            self.net = nn.DataParallel(self.net)

            #成績の初期化
            losses,maes = {},{}
            for phase in ['learning','valid']:
                losses[phase] = []
                maes[phase] = []

            self.dataset = load_dataloader(self.c['bs'])
            kf = KFold(n_splits=5,shuffle=True,random_state=0)
            epoch_n = 1


            for learning_index,valid_index in kf.split(self.dataset['train']):
                #データセットが切り替わるたびに、ネットワークの重み,バイアスを初期化
                #utils.py -> init_weights()
                self.net.apply(init_weights)
                self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
                    
                learning_dataset = Subset(self.dataset['train'],learning_index)
                self.dataloaders['learning'] = DataLoader(learning_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())
                valid_dataset = Subset(self.dataset['train'],valid_index)
                self.dataloaders['valid'] = DataLoader(valid_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())


                for epoch in range(1, self.c['n_epoch']+1):

                    learningmae,learningloss,learningr_score,learningmse \
                        = self.execute_epoch(epoch, 'learning')

                    validmae,validloss,validr_score,validmse\
                        = self.execute_epoch(epoch, 'valid')

                    mae_sum = validmae
                    if min_mae < mae_sum:
                        min_mae = mae_sum
                        self.bestparam = self.c
                        self.bestparam['epoch'] = epoch_n
                        self.bestparam['mae'] = min_mae

                    epoch_n += 1
                    if epoch == self.c['n_epoch']:
                        self.mae['learning'] += learningmae
                        self.mae['valid'] += validmae
                        self.loss['learning'] += learningloss
                        self.loss['valid'] += validloss
                        self.mse['learning'] += learningmse
                        self.mse['valid'] += validmse
                        self.r_score['learning'] += learningr_score
                        self.r_score['valid'] += validr_score
                        
                
                #n_epoch後の処理
                save_process_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))

            #分割交差検証後の処理
            #乱数シードiterごとに、平均を取り、これを記録。
            if not (n_iter%self.n_seeds):
                temp = self.n_seeds * self.n_splits
                for phase in ['learning','valid']:
                    self.mae[phase]  /= temp
                    self.loss[phase] /= temp
                    self.mse[phase] /= temp
                    self.r_score[phase] /= temp
                    self.tb_writer.add_scalar('Loss/{}'.format(phase),self.loss[phase],self.c['n_epoch'])
                    self.tb_writer.add_scalar('Mae/{}'.format(phase),self.mae[phase],self.c['n_epoch'])
                    self.tb_writer.add_scalar('R2_score/{}'.format(phase),self.r_score[phase],self.c['n_epoch'])
                    self.tb_writer.add_scalar('Mse/{}'.format(phase),self.mse[phase],self.c['n_epoch'])
                    #print(n_iter)
                    #print(self.c['n_epoch'])
                    #print(self.c['seed'])
                    #print('{:.2f}'.format(self.mae[phase]),'{:.2f}'.format(self.loss[phase]),'{:.2f}'.format(self.mse[phase]),'{:.2f}'.format(self.r_score)[phase])
                    #initialization -> Score
                    self.mae[phase] = 0
                    self.loss[phase] = 0
                    self.mse[phase] = 0
                    self.r_score[phase] = 0


                
            n_iter += 1

        #パラメータiter後の処理。
        #def plot_history(history,num,xinfo,yinfo):
        #    plt.plot(history['learning'])
        #    plt.plot(history['valid'])
        #    plt.xlabel(xinfo)
        #    plt.ylabel(yinfo)
        #    plt.yscale('log')
        #    plt.legend(['learning','valid'],loc='upper right')
        #    save_process_path = os.path.join(config.LOG_DIR_PATH,
        #                        str(self.now))
        #    plt.savefig(save_process_path + '/history' + str(num) + '.png')
        #    plt.figure()

        #plot_history(losses,1,'epoch','loss')
        #plot_history(aucs,2,'epoch','auc')

            
        #print(self.bestparam)
        #result_best = [self.bestparam['model_name'],self.bestparam['lr'],self.bestparam['seed'],self.bestparam['n_epoch'],self.bestparam['mae']]

        #with open(self.log_path + "/log.csv",'a') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(['-'*20 + 'bestparameter' + '-'*20])
        #    writer.writerow(['model_name','lr','seed','n_epoch','mae'])
        #    writer.writerow(result_best)

        elapsed_time = time.time() - start
        print(f"実行時間 : {elapsed_time:.01f}")
        #訓練後、モデルをセーブする。
        #(実行回数)_(モデル名)_(学習epoch).pth で保存。
        try : 
             model_name = self.search['model_name']
             n_ep = self.search['n_epoch'][-1]
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


        #JSON形式でTensorboardに保存した値を残しておく。
        self.tb_writer.export_scalars_to_json('./log/all_scalars.json')
        self.tb_writer.close()

    #1epochごとの処理
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


            with torch.set_grad_enabled(phase == 'learning'):
                outputs_ = self.net(inputs_)
                loss = self.criterion(outputs_, labels_)
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


        return mae,total_loss,r_score,mse