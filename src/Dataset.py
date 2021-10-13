import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from PIL import Image
import os
import numpy as np
from utils import *


class NuclearCataractDatasetBase(Dataset):
    def __init__(self, root, image_list_file, transform=None):
        image_names = []
        labels = []
        self.image_list_file = image_list_file

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(',')
                if isint(items[1]):
                    label = self.get_label(int(items[1]))
                    
                    image_name = items[0]
                    image_name = os.path.join(root,image_name)
                    image_names.append(image_name)
                    labels.append(label[0])



        self.image_names = np.array(image_names)
        self.labels = np.array(labels)
        self.transform = transform


    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return (image,torch.Tensor([label])) if self.image_list_file==config.train_info_list else (image,torch.Tensor([label]),image_name)

    def __len__(self):
        return len(self.image_names)
        #return 1000

    def get_label(self, label_base):
        pass
        

class NuclearCataractDataset(NuclearCataractDatasetBase):
    def get_label(self, label_base):
        return [label_base]

#まずはn=2の場合で実装
class EnsembleDataset(Dataset):
    def __init__(self,*model_files):

        n_models = len(model_files)
        model_preds = [[] for x in range(n_models)]
        labels = []

        for i in range(n_models):
            labels = []
            model_file = os.path.join(config.LOG_DIR_PATH,model_files[i])
            print(model_file)
            with open(model_file,'r') as f:
                for line in f:
                    items = line.strip().split(',')
                    #今の所使うのは検証データだけ
                    if isfloat(items[0]) and len(items)>1:
                        model_preds[i].append(float(items[0]))
                        labels.append(int(float(items[1])))

        self.model_preds = np.array(model_preds)
        self.labels = np.array(labels)


    def __getitem__(self, index):
        f_pd,s_pd = self.model_preds[0][index],self.model_preds[1][index]
        label = self.labels[index]
        return (torch.Tensor([f_pd,s_pd]),torch.Tensor([label]))

    def __len__(self):
        return len(self.labels)
        #return 1000

    def get_label(self, label_base):
        pass

def load_ensembledata(batch_size):
    dataset = {}
    dataset['train'] = \
        EnsembleDataset('first_model.csv','second_model.csv')
    dataset['test'] = \
        EnsembleDataset('first_model.csv','second_model.csv')

    return dataset
    
    '''
    CV実装のため、データセット飲みの実装
    dataloader = {}
    dataloader['train'] = \
        torch.utils.data.DataLoader(dataset['train'],
                                    batch_size=batch_size,
                                    num_workers=os.cpu_count())
    dataloader['test'] = \
        torch.utils.data.DataLoader(dataset['test'],
                                    batch_size=batch_size,
                                    num_workers=os.cpu_count())
    return dataloader
    '''


def load_dataloader(batch_size):
    train_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],
                                                 [0.229,0.224,0.225])])
    test_transform = \
        transforms.Compose([transforms.Resize(config.image_size),
                            transforms.CenterCrop(config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],
                                                 [0.229,0.224,0.225])])
    dataset = {}
    dataset['train'] = \
        NuclearCataractDataset(root=config.data_root,
                                  image_list_file=config.train_info_list,
                                  transform=train_transform)
    dataset['test'] = \
        NuclearCataractDataset(root=config.data_root,
                                  image_list_file=config.test_info_list,
                                  transform=test_transform)

    return dataset
    '''
    CV実装のため、データセットのみの実装
    dataloader = {}
    
    dataloader['train'] = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    dataloader['test'] = \
        torch.utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    return dataloader
    '''

