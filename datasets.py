from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms, datasets
import argparse


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, noise_fraction=None, mode='train', target_transform=None):
        if mode == 'train':
            datatxt = 'ISIC_2019_Training_GroundTruth_sub_train_' + str(noise_fraction) + '.csv'

        if mode == 'test':
            datatxt = 'ISIC_2019_Training_GroundTruth_sub_test.csv'

        if mode == 'val':
            datatxt = 'ISIC_2019_Training_GroundTruth_sub_clean.csv'    

        if mode == 'base':
            datatxt = 'ISIC_2019_Training_GroundTruth_sub_train.csv'        

        # datatxt = 'ISIC_2019_Training_GroundTruth.csv'
        # lean_datatxt = 'ISIC_2019_Training_GroundTruth_clean.csv'
        
        # fc = open(clean_datatxt, 'r')
        fh = open(datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            imgs.append((line.split(",")[0], line.split(",")[1:]))

        self.imgs = imgs
        self.target_transform = target_transform
        self.imgs_dir = imgs_dir
        self.transform = transforms.Compose([
               transforms.Resize([224, 224]),
               transforms.ToTensor()
            ])
        
        '''
        self.val_data = []
        self.val_label = []

        for line in fc:
            line = line.rstrip()
            fm, label_val = line.split(",")[0], line.split(",")[1:]
            label_val = list(map(float, label_val))
            label_val = np.array(label_val)
            self.val_label.append(label_val)

            data_val = Image.open(self.imgs_dir+fm+'.jpg').convert('RGB')
            data_val = self.transform(data_val)
            self.val_data.append(data_val)

            self.val_data = torch.cat(self.val_data, dim=0)
            self.val_label = torch.cat(self.val_label, dim=0)
        '''

    def __getitem__(self, index):
        fn, labels = self.imgs[index]
        label = labels.index('1.0').int()

        img = Image.open(self.imgs_dir+fn+'.jpg').convert('RGB')
        img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.imgs)