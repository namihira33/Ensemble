import os
import sys
import time
from datetime import datetime
import random

from tqdm import tqdm
import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression

import config
import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader 
from network import *
from Dataset import load_dataloader
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib

c = {
    'model_name': 'Resnet18',
    'seed': [0], 'bs': 64, 'lr': [1e-4], 'n_epoch': [10]
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

#グラフ内で日本語を使用可能にする。
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Evaluater():
    def __init__(self,c):
        self.dataloaders = {}
        self.c = c
        now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        model_path = ''
        args = len(sys.argv)
        with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv')) as f:
             lines = [s.strip() for s in f.readlines()]
        if args < 2 :
             target_data = lines[-1].split(',')
        else:
             if int(sys.argv[1])<=1:
                 print('Use the first data')
                 target_data = lines[-1].split(',')
             else:
                 try:
                     target_data = lines[int(sys.argv[1])].split(',')
                     self.c['image'] = int(sys.argv[2].split('=')[1])
                 except IndexError:
                     print('It does not exit. Use the first data')
                     target_data = lines[-1].split(',')

        self.n_ex = '{:0=2}'.format(int(target_data[1]))
        self.c['model_name'] = target_data[2]
        self.c['n_epoch'] = '{:0=3}'.format(int(target_data[3]))
        temp = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep.pth'
        model_path = os.path.join(config.MODEL_DIR_PATH,temp)

        self.net = make_model(self.c['model_name']).to(device)
        self.net.load_state_dict(torch.load(model_path,map_location=device))
        self.criterion = nn.BCEWithLogitsLoss()

    def run(self):
            self.dataset = load_dataloader(self.c['bs'])
            test_dataset = self.dataset['test']
            self.dataloaders['test'] = DataLoader(test_dataset,self.c['bs'],
                    shuffle=True,num_workers=os.cpu_count())

            preds, labels,paths,total_loss,accuracy= [],[],[],0,0
            right,notright = 0,0
            self.net.eval()

            for inputs_, labels_,paths_ in tqdm(self.dataloaders['test']):
                inputs_ = inputs_.to(device)
                labels_ = labels_.to(device)


                torch.set_grad_enabled(False)
                outputs_ = self.net(inputs_)
                loss = self.criterion(outputs_, labels_)
                #total_loss += loss.item()


                preds += [outputs_.detach().cpu().numpy()]
                labels += [labels_.detach().cpu().numpy()]
                paths  += paths_

                total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            
            if self.c['image']:
                tmp = 0
                fig,ax = plt.subplots(4,4,figsize=(16,16))
                for i,(pred,ans,path) in enumerate(zip(preds,labels,paths)):
                    im = Image.open(path)#.convert('RGB')

                    ax[(i%16)//4][i%4].imshow(im)
                    pred = '{:.2f}'.format(pred[0])
                    ans = ans[0]
                    ax[(i%16)//4][i%4].set_title('Predict:' + str(pred) + '  Answer:' + str(ans))
                    ax[(i%16)//4][i%4].title.set_size(20)
                    if i%16==15:
                        fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','exp'+str(i//16)+'.png'))
                        fig,ax = plt.subplots(4,4,figsize=(16,16))
                    tmp = i

                if tmp%16!=15:
                    while (tmp%16!=0):
                        ax[(tmp%16)//4][tmp%4].axis('off')
                        tmp += 1

                    fig.savefig(os.path.join(config.LOG_DIR_PATH,'images','exp' + str(i//16) + '.png'))

            #preds = np.round(preds)
            #labels = np.round(labels)
            total_loss /= len(preds)

            worst_id = np.argmax(preds-labels)
            worst = (preds-labels).max()

            print(np.argmax(preds-labels),(preds-labels).max())
            print(preds[worst_id],labels[worst_id])

            threshold = 1.01
            right += ((preds-labels) < threshold).sum()
            notright += len(preds) - ((preds - labels) < threshold).sum()

            accuracy = right / len(test_dataset)
            mae = mean_absolute_error(preds,labels)
            mse = mean_squared_error(preds,labels)
            r_score = r2_score(preds,labels)
            kappa = 0
            #kappa = cohen_kappa_score(preds,labels,weights='quadratic')
            print('accuracy :',accuracy)
            print('MAE :',mae)
            print('AE : ',mae*len(preds))
            print('MSE',mse)
            print('SE',mse*len(preds))
            print('kappa',kappa)

            fig,ax = plt.subplots()
            lr = LinearRegression()
            lr.fit(preds,labels)
            ax.scatter(preds,labels)
            ax.plot(preds,lr.predict(preds),color='red',label='Linear Regression')
            ax.set_title('予測-答え散布図')
            ax.title.set_size(20)
            ax.legend()
            ax.set_xlabel('Predict Age')
            ax.set_ylabel('Answer Age')
            fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_regression.png'
            print(fig_path)
            print(os.path.join(config.LOG_DIR_PATH,'images',fig_path))
            fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))


            fig,ax = plt.subplots()
            ax.bar(['Acc','R-score','kappa'],[accuracy,r_score,kappa],width=0.4,tick_label=['Accuracy','R-Score','kappa'],align='center')
            ax.set_title('評価値1')
            ax.set_ylabel('Score')
            ax.set_ylim(0.0,1.0)
            ax.title.set_size(20)
            ax.grid(True)
            fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_graph1.png'
            fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))

            fig,ax = plt.subplots()
            ax.bar(['MAE'],[mae],width=0.4,tick_label=['MAE'],align='center')
            ax.set_title('評価値2')
            ax.set_ylabel('Age')
            ax.set_xlim(-1,1)
            ax.title.set_size(20)
            ax.grid(True)
            fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_graph2.png'
            fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))



if __name__ == '__main__':
    evaluater = Evaluater(c)
    evaluater.run()
