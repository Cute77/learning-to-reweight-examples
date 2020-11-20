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
from tensorboardX import SummaryWriter
import logging
import torch.distributed as dist
from torch.utils.data import distributed
import os
import random

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  
random.seed(seed)   
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-i', '--imgs-dir', metavar='ID', type=str, nargs='?', default='ISIC_2019_Training_Input/',
                        help='image path', dest='imgs_dir')
    parser.add_argument('-n', '--noise-fraction', metavar='NF', type=float, nargs='?', default=0.2,
                        help='Noise Fraction', dest='noise_fraction')
    parser.add_argument('-f', '--fig-path', metavar='FP', type=str, nargs='?', default='baseline',
                        help='Fig Path', dest='figpath')
    parser.add_argument('-r', '--local_rank', metavar='RA', type=int, nargs='?', default=0,
                        help='from torch.distributed.launch', dest='local_rank')
    parser.add_argument('-o', '--load', metavar='LO', type=int, nargs='?', default=0,
                        help='load epoch', dest='load')

    return parser.parse_args()


args = get_args()
lr = args.lr
local_rank = args.local_rank
load = args.load
epochs = args.epochs
fig_path = args.fig_path
net, opt = build_model(lr)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
lr = lr * num_gpus
dir = 'baseline/' + fig_path
path = 'baseline/' + fig_path + '/' + str(load) + '_model.pth'
if not os.path.exists(dir):
    os.mkdir(dir)

net = torch.nn.parallel.DistributedDataParallel(
    net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, 
)
if os.path.isfile(path) and load > 0:
    logging.info(f'''Continue''')
    net.load_state_dict(torch.load(path))

scheduler = StepLR(opt, step_size=50, gamma=0.5, last_epoch=load)
writer = SummaryWriter(comment=f'name_{args.figpath}')

logging.info(f'''Starting training:
    Epochs:          {args.epochs}
    Batch size:      {args.batch_size}
    Learning rate:   {args.lr}
    Noise fraction:  {args.noise_fraction}
''')

net_losses = []
acc_test = []
acc_train = []
loss_train = []
plot_step = 100
net_l = 0
global_step = 0
test_step = 0

smoothing_alpha = 0.9
accuracy_log = []

# data_loader = dl.get_mnist_loader(args.batch_size, classes=[9, 4], proportion=0.995, mode="train")
# test_loader = dl.get_mnist_loader(args.batch_size, classes=[9, 4], proportion=0.5, mode="test")


train = BasicDataset(imgs_dir=args.imgs_dir, noise_fraction=args.noise_fraction, mode='train')
# train = BasicDataset(imgs_dir=args.imgs_dir, mode='base')
test = BasicDataset(imgs_dir=args.imgs_dir, mode='test')

train_sampler = distributed.DistributedSampler(train, num_replicas=num_gpus, rank=local_rank) if is_distributed else None

data_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, sampler=train_sampler)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

data = iter(data_loader)
loss = nn.CrossEntropyLoss()
# loss = nn.MultiLabelSoftMarginLoss()

test_num = 0
correct_num = 0

for epoch in range(load+1, epochs):
    epoch_loss = 0
    correct_y = 0
    num_y = 0
    test_num = 0
    correct_num = 0

    for i in range(len(data_loader)):
    # for i in range(8000):
        net.train()
        try:
            image, labels, _ = next(data)
        except StopIteration:
            data = iter(data_loader)
            image, labels, _ = next(data)
        # image, labels = next(iter(data_loader))

        if is_distributed:
            image = image.cuda(local_rank)
            labels = labels.cuda(local_rank)
            val_data = val_data.cuda(local_rank)
            val_labels = val_labels.cuda(local_rank)
        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        y = net(image)
        cost = loss(y, labels)
        epoch_loss = epoch_loss + cost.item()
        net_losses.append(cost.item())
        writer.add_scalar('Loss/train', cost.item(), global_step)

        _, y_predicted = torch.max(y, 1)
        correct_y = correct_y + (y_predicted.int() == labels.int()).sum().item()
        num_y = num_y + labels.size(0)
        writer.add_scalar('StepAccuracy/train', ((y_predicted.int() == labels.int()).sum().item()/labels.size(0)), global_step)

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
                writer.add_scalar('StepAccuracy/test', ((predicted.int() == test_label.int()).sum().item()/test_label.size(0)), test_step)
                test_step = test_step + 1

            accuracy = torch.cat(acc, dim=0).mean()
            accuracy_log.append(np.array([i, accuracy])[None])
            acc_log = np.concatenate(accuracy_log, axis=0)
            test_step = test_step + 1

    print('epoch ', epoch)
    print('epoch loss: ', epoch_loss/len(train))
    loss_train.append(epoch_loss/len(train))
    writer.add_scalar('EpochAccuracy/train', epoch_loss/len(train), epoch)

    print('epoch accuracy: ', correct_y/num_y)
    acc_train.append(correct_y/num_y)
    writer.add_scalar('EpochAccuracy/train', correct_y/num_y, epoch)

    path = 'baseline/' + args.figpath + '_model.pth'
    # path = 'baseline/' + str(args.noise_fraction) + '/model.pth'
    torch.save(net.state_dict(), path)

    writer.add_scalar('EpochAccuracy/test', correct_num/test_num, epoch)
    print('test accuracy: ', correct_num/test_num)
    acc_test.append(correct_num/test_num)

IPython.display.clear_output()
fig, axes = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axes.ravel()

ax1.plot(net_losses, label='train_losses')
ax1.set_ylabel("Losses")
ax1.set_xlabel("Iteration")
ax1.legend()

ax2.plot(loss_train, label='epoch_losses')
ax2.set_ylabel('Losses/train')
ax2.set_xlabel('Epoch')
ax2.legend()

ax3.plot(acc_train, label='acc_train')
ax3.set_ylabel('Accuracy/train')
ax3.set_xlabel('Epoch')
ax3.legend()

ax4.plot(acc_test, label='acc_test')
ax4.set_ylabel('Accuracy/test')
ax4.set_xlabel('Epoch')
ax4.legend()

plt.savefig(args.figpath+'.png')    

print(np.mean(acc_log[-6:-1, 1]))
print('Accuracy: ', correct_num/test_num)