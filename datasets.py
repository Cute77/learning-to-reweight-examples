from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageFilter
from torchvision import transforms, datasets
import argparse

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, noise_fraction=None, mode='train', target_transform=None):
        if mode == 'train':
            # datatxt = 'ISIC_2019_Training_GroundTruth_sub5000_train_' + str(noise_fraction) + '.csv'
            datatxt = 'ISIC_2019_Training_GroundTruth_markedgiven5000_train_' + str(noise_fraction) + '.csv'
            print(datatxt)

        if mode == 'test':
            datatxt = 'ISIC_2019_Training_GroundTruth_sub1250_test.csv'
            print(datatxt)

        if mode == 'val':
            # datatxt = 'ISIC_2019_Training_GroundTruth_sub_clean.csv'  
            datatxt = 'ISIC_2019_Training_GroundTruth_clean.csv'  
            print(datatxt)

        if mode == 'base':
            datatxt = 'ISIC_2019_Training_GroundTruth_sub5000_train.csv'        
            print(datatxt)
            
        # datatxt = 'ISIC_2019_Training_GroundTruth.csv'
        # lean_datatxt = 'ISIC_2019_Training_GroundTruth_clean.csv'
        
        # fc = open(clean_datatxt, 'r')
        fh = open(datatxt, 'r')
        imgs = []
        if "markedgiven" in datatxt:
            for line in fh:
                line = line.rstrip()
                imgs.append((line.split(" ")[0], line.split(" ")[1], line.split(" ")[2].split(",")[0], line.split(" ")[2].split(",")[1:]))
        else:
            for line in fh:
                line = line.rstrip()
                imgs.append(("0", "0", line.split(",")[0], line.split(",")[1:]))


        self.imgs = imgs
        self.target_transform = target_transform
        self.imgs_dir = imgs_dir
        self.transform = transforms.Compose([ 
               transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(degrees=180),
               # transforms.RandomGrayscale(p=0.1),
               transforms.RandomResizedCrop(size=224, scale=(0.3, 1.0)), 
               # transforms.Resize([224, 224]), 
               transforms.ToTensor(), 
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0])
            ])

    def __getitem__(self, index):
        mark, gt, fn, labels = self.imgs[index]
        gt = int(gt)
        mark = int(mark)
        label = labels.index('1.0')
        img = Image.open(self.imgs_dir+fn+'.jpg').convert('RGB')
        '''
        variance = np.random.randint(-5, 5) * 0.01
        if variance > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            img = random_noise(np.array(img), mode='gaussian', var=variance)
            img = img.astype(np.uint8)
            img = transforms.ToPILImage()(img)
        '''
        img = self.transform(img)

        return img, label, mark, gt

    def __len__(self):
        return len(self.imgs)