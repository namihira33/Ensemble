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
from Dataset import *

import csv
import matplotlib.pyplot as plt
import seaborn as sns

seed = list(np.random.choice(10000,1))

c = {'model_name':'ensemble_MLP',
    'n_epoch': 16,'seed': seed, 'bs': 64, 'lr': [8e-6,9e-6],'hidden_layer' : [100]
}
#20,30,40,50,100

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

args = len(sys.argv)
if args >= 2:
    c['cv'] = int(sys.argv[1].split('=')[1])
    c['evaluate'] = int(sys.argv[2].split('=')[1])
    #c['lr'] = float(sys.argv[3].split('=')[1])

class Ensemble():
    def __init__(self,c):
        
        self.dataloaders = {}
        self.prms = []
        self.search = c
        self.n_splits = 5
        self.n_seeds = 1#len(c['seed'])
        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))
        os.makedirs(self.log_path, exist_ok=True)

    def run(self):
        #実行時間計測のための準備
        start = time.time()

        #ヒートマップ描画用のリスト
        validheat,heat_index = [],[]

        #seed平均を取るためのリスト
        seed_valid = []


        for n_iter,(c,param) in enumerate(iterate(self.search)):
            #MAEの5分割平均を取るためのリスト
            memory = [[] for x in range(1)] if self.search['cv']==0 else [[] for x in range(self.n_splits)]

            self.c = c
            print(c)

            random.seed(self.c['seed'])
            torch.manual_seed(self.c['seed'])

            self.net = MLP(hidden_layer=self.c['hidden_layer']).to(device)
            self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
            self.criterion = nn.MSELoss()
            self.net = nn.DataParallel(self.net)
            self.dataset = load_ensembledata(self.c['bs'])

            #分割のためのインスタンス作成
            kf = KFold(n_splits=self.n_splits,shuffle=True,random_state=self.c['seed'])
            divided_data = kf.split(self.dataset['train'])


            for a,(learning_index,valid_index) in enumerate(divided_data):
                #データセットが切り替わるごとにネットワークの重み、バイアスを初期化
                self.net.apply(init_weights)
                self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)

                learning_dataset = Subset(self.dataset['train'],learning_index)
                self.dataloaders['learning'] = DataLoader(learning_dataset,self.c['bs'],
                shuffle=True,num_workers=os.cpu_count())

                if not self.c['evaluate']:
                    valid_dataset = Subset(self.dataset['train'],valid_index)
                    self.dataloaders['valid'] = DataLoader(valid_dataset,self.c['bs'],
                    shuffle=True,num_workers=os.cpu_count())
            

                for epoch in range(1,self.c['n_epoch']+1):
                    learningmae,learningloss,learningr_score\
                            = self.execute_epoch(epoch, 'learning')
                    if not self.c['evaluate']:
                        validmae,validloss,validr_score,valid_preds,valid_labels\
                            = self.execute_epoch(epoch, 'valid')
                        #validmaeを蓄えておいてあとでベスト10を出力
                        temp = validmae,epoch,self.c
                        self.prms.append(temp)
                        #validmaeを蓄えておいて後で平均を算出
                        memory[a].append(validmae)

                #n_epoch後の処理
                if not self.c['cv']:
                    break
            
            #交差分割後の処理
            #平均をTensorboardに載せるようにする。
            memory = list(np.mean(memory,axis=0))
            seed_valid.append(memory)
            self.tb_writer = tbx.SummaryWriter()
            for i,valimae in enumerate(memory):
                self.tb_writer.add_scalar('Valid MAE',memory[i],i+1)
            self.tb_writer.close()

            if not self.c['evaluate']:
                if not ((n_iter+1)%self.n_seeds):
                    print(memory)
                    seed_valid = np.mean(seed_valid,axis=0)
                    print(seed_valid)
                    validheat.append(seed_valid)
                    heat_index.append(self.c['lr'])
                    seed_valid = []
        
        #パラメータiter後の処理
        #結果をヒートマップで図示
        validheat = [l[::5] for l in validheat[:]]
        print(validheat)
        fig,ax = plt.subplots(figsize=(12,6))
        xtick = list(map(lambda x:5*x-4,list(range(1,len(validheat[0])+1))))
        xtick = [str(x) + 'ep' for x in xtick]
        #heat_index = [self.c['lr']]
        sns.heatmap(validheat,annot=True,cmap='Set3',fmt='.2f',
                xticklabels=xtick,yticklabels=heat_index,vmin=2.5,vmax=10,
                cbar_kws = dict(label='Valid Age MAE'))
        ax.set_ylabel('learning rate')
        ax.set_xlabel('num of epoch')
        ax.set_title('num of hidden layers : ' + str(self.c['hidden_layer']))
        fig.savefig('./log/images/' + self.now + 'train.png')

        #実行結果の計測
        elapsed_time = time.time() - start
        print(f"実行時間 : {elapsed_time:.01f}")


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

if __name__ == '__main__':
    ensemble = Ensemble(c)
    ensemble.run()

