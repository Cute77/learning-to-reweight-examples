import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
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
import matplotlib
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def build_model(lr):
    net = model.resnet101(pretrained=True, num_classes=9)
    # net = model.LeNet(n_out=1)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.params(), lr, weight_decay=1e-4)
    
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
    parser.add_argument('-i', '--imgs-dir', metavar='ID', type=str, nargs='?', default='ISIC_2019_Training_Input/',
                        help='image path', dest='imgs_dir')
    parser.add_argument('-n', '--noise-fraction', metavar='NF', type=float, nargs='?', default=0.2,
                        help='Noise Fraction', dest='noise_fraction')

    return parser.parse_args()

# writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

args = get_args()
lr = args.lr
net, opt = build_model(lr)

net_losses = []
plot_step = 100
net_l = 0
global_step = 0

smoothing_alpha = 0.9
accuracy_log = []

# data_loader = dl.get_mnist_loader(args.batch_size, classes=[9, 4], proportion=0.995, mode="train")
# test_loader = dl.get_mnist_loader(args.batch_size, classes=[9, 4], proportion=0.5, mode="test")


# train = BasicDataset(imgs_dir=args.imgs_dir, noise_fraction=args.noise_fraction, mode='train')
train = BasicDataset(imgs_dir=args.imgs_dir, mode='base')
test = BasicDataset(imgs_dir=args.imgs_dir, mode='test')

data_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

data = iter(data_loader)
loss = nn.CrossEntropyLoss()
# loss = nn.MultiLabelSoftMarginLoss()

test_num = 0
correct_num = 0

for epoch in range(args.epochs):
    epoch_loss = 0
    correct_y = 0
    num_y = 0
    test_num = 0
    correct_num = 0
    if epoch % 20 == 0:
        lr = lr/2
    opt = torch.optim.SGD(net.params(), lr)

    for i in tqdm(range(len(train))):
    # for i in range(8000):
        net.train()
        try:
            image, labels = next(data)
        except StopIteration:
            data = iter(data_loader)
            image, labels = next(data)
        # image, labels = next(iter(data_loader))

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        y = net(image)
        cost = loss(y, labels)
        epoch_loss = epoch_loss + cost.item()
        net_losses.append(cost.item())
        # writer.add_scalar('Loss/train', cost.item(), global_step)

        _, y_predicted = torch.max(y, 1)
        correct_y = correct_y + (y_predicted.int() == labels.int()).sum().item()
        num_y = num_y + labels.size(0)

        opt.zero_grad()
        cost.backward()
        opt.step()
        global_step = global_step + 1

        if i % plot_step == 0:
            net.eval()
            
            acc = []
            for i, (test_img, test_label) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                with torch.no_grad():
                    output = net(test_img)
                _, predicted = torch.max(output, 1)
                
                test_num = test_num + test_label.size(0)
                correct_num = correct_num + (predicted.int() == test_label.int()).sum().item()
                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc, dim=0).mean()
            accuracy_log.append(np.array([i, accuracy])[None])
            acc_log = np.concatenate(accuracy_log, axis=0)
    
            IPython.display.clear_output()
            fig, axes = plt.subplots(1, 2, figsize=(13,5))
            ax1, ax2 = axes.ravel()

            ax1.plot(net_losses, label='net_losses')
            ax1.set_ylabel("Losses")
            ax1.set_xlabel("Iteration")
            ax1.legend()
            
            acc_log = np.concatenate(accuracy_log, axis=0)
            ax2.plot(acc_log[:,0],acc_log[:,1])
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Iteration')
            plt.savefig('baseline.png')

    print('epoch ', epoch)
    print('epoch loss: ', epoch_loss/len(train))
    print('epoch accuracy: ', correct_y/num_y)
    # writer.add_scalar('Accuracy/train', correct_y/num_y, epoch)
    path = 'baseline/' + str(args.noise_fraction) + '/model.pth'
    # path = 'baseline/' + str(args.noise_fraction) + '/model.pth'
    torch.save(net.state_dict(), path)
    print('test accuracy: ', np.mean(acc_log[-6:-1, 1]))
    # writer.add_scalar('Accuracy/test', correct_num/test_num, epoch)
    print('test accuracy: ', correct_num/test_num)

print(np.mean(acc_log[-6:-1, 1]))
print('Accuracy: ', correct_num/test_num)