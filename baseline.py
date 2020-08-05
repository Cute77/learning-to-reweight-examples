import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import model
from tqdm import tqdm
import IPython
import gc
import torchvision
from datasets import BasicDataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
import data_loader as dl


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def build_model(lr):
    net = model.resnet101(pretrained=False, num_classes=9)
    # net = model.LeNet(n_out=1)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.params(), lr)
    
    return net, opt


def get_args():
    parser = argparse.ArgumentParser(description='Learning to reweight on classification tasks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=128,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-i', '--imgs_dir', metavar='ID', type=str, nargs='?', default='ISIC_2019_Training_Input/',
                        help='image path', dest='imgs_dir')
    parser.add_argument('-n', '--noise_fraction', metavar='NF', type=float, nargs='?', default=0.2,
                        help='Noise Fraction', dest='noise_fraction')

    return parser.parse_args()


args = get_args()
net, opt = build_model(lr=args.lr)

net_losses = []
plot_step = 100
net_l = 0

smoothing_alpha = 0.9
accuracy_log = []

# data_loader = dl.get_mnist_loader(args.batch_size, classes=[9, 4], proportion=0.995, mode="train")
# test_loader = dl.get_mnist_loader(args.batch_size, classes=[9, 4], proportion=0.5, mode="test")

train = BasicDataset(imgs_dir=args.imgs_dir, mode='base')
test = BasicDataset(imgs_dir=args.imgs_dir, mode='test')

data_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

data = iter(data_loader)

for epoch in range(args.epochs):
    for i in tqdm(range(int(len(train)/args.batch_size))):
    # for i in range(8000):
        net.train()
        image, labels = next(data)
        # image, labels = next(iter(data_loader))


        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        y = net(image)
        labels = labels.float()
        cost = F.binary_cross_entropy_with_logits(y, labels)
        
        opt.zero_grad()
        cost.backward()
        opt.step()
        
        if i % plot_step == 0:
            net.eval()
            
            acc = []
            for i, (test_img, test_label) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)
                
                output = net(test_img)
                predicted = (F.sigmoid(output) > 0.5).int()
                
                acc.append((predicted.float() == test_label.float()).float())

            accuracy = torch.cat(acc, dim=0).mean()
            accuracy_log.append(np.array([i, accuracy])[None])
            acc_log = np.concatenate(accuracy_log, axis=0)

print(np.mean(acc_log[-6:-1, 1]))