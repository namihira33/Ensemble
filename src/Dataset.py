import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from PIL import Image
import os
import numpy as np
from utils import isint


class NuclearCataractDatasetBase(Dataset):
    def __init__(self, root, image_list_file, transform=None):
        image_names = []
        labels = []
        self.image_list_file = image_list_file

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(',')
                if isint(items[2]):
                    label = self.get_label(int(items[2]))
                    
                    image_name = items[1]
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

