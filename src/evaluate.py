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

import matplotlib.pyplot as plt

c = {
    'model_name': 'Resnet18',
    'seed': [0], 'bs': 64, 'lr': [1e-4], 'n_epoch': [10]
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

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
                 except IndexError:
                     print('It does not exit. Use the first data')
                     target_data = lines[-1].split(',')

        self.n_ex = '{:0=2}'.format(int(target_data[1]))
        self.c['model_name'] = target_data[2]
        self.c['n_epoch'] = '{:0=3}'.format(int(target_data[3]))
        temp = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep.pth'
        model_path = os.path.join(config.MODEL_DIR_PATH,temp)

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

        self.net.load_state_dict(torch.load(model_path,map_location=device))
        self.criterion = nn.BCEWithLogitsLoss()

    def run(self):
            self.dataset = load_dataloader(self.c['bs'])
            test_dataset = self.dataset['test']
            self.dataloaders['test'] = DataLoader(test_dataset,self.c['bs'],
                    shuffle=True,num_workers=os.cpu_count())

            preds, labels,total_loss,accuracy= [], [],0,0
            right,notright = 0,0
            self.net.eval()

            for inputs_, labels_ in tqdm(self.dataloaders['test']):
                inputs_ = inputs_.to(device)
                labels_ = labels_.to(device)


                torch.set_grad_enabled(False)
                outputs_ = self.net(inputs_)
                loss = self.criterion(outputs_, labels_)
                #total_loss += loss.item()


                preds += [outputs_.detach().cpu().numpy()]
                labels += [labels_.detach().cpu().numpy()]

                total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            total_loss /= len(preds)

            worst_id = np.argmax(preds-labels)
            worst = (preds-labels).max()

            print(np.argmax(preds-labels),(preds-labels).max())

            print(preds[worst_id],labels[worst_id])

            threshold = 0.5
            right += ((preds-labels) < threshold).sum()
            notright += len(preds) - ((preds - labels) < threshold).sum()

            accuracy = right / len(test_dataset)
            mae = mean_absolute_error(preds,labels)
            mse = mean_squared_error(preds,labels)
            r_score = r2_score(preds,labels)
            print('accuracy :',accuracy)
            print('MAE :',mae)
            print('AE : ',mae*len(preds))
            print('MSE',mse)
            print('SE',mse*len(preds))

            lr = LinearRegression()
            lr.fit(preds,labels)
            plt.scatter(preds,labels)
            plt.plot(preds,lr.predict(preds),color='red')
            fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_regression.png'
            print(fig_path)
            print(os.path.join(config.LOG_DIR_PATH,'images',fig_path))
            plt.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))


            fig,ax = plt.subplots()
            ax.bar(['Acc','Mae','R-score'],[accuracy,mae,r_score],width=0.4,tick_label=['Accuracy','Mae','R-Score'],align='center')
            ax.grid(True)
            fig_path = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep_graph.png'
            fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))



if __name__ == '__main__':
    evaluater = Evaluater(c)
    evaluater.run()
