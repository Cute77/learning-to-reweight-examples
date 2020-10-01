import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import model
from torchvision import models
from tqdm import tqdm
import IPython
import gc
import torchvision
from datasets import BasicDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import logging
import torch.distributed as dist
from torch.utils.data import distributed
import matplotlib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import sys
import higher

'''
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
'''


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def build_model(lr):
    net = models.resnet101(pretrained=True, num_classes=9)

    if torch.cuda.is_available():
        # net.cuda()
        net = net.cuda()
        # torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.parameters(), lr, weight_decay=1e-4)
    
    return net, opt


def train_net(noise_fraction, 
              fig_path, 
              lr=1e-3,
              momentum=0.9, 
              batch_size=128,
              dir_img='ISIC_2019_Training_Input/',
              save_cp=True,
              dir_checkpoint='checkpoints/ISIC_2019_Training_Input/',
              epochs=10):

    # torch.distributed.is_nccl_available()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    # device_ids = list(map(int, device_id.split(',')))
    # print(device_ids)

    net, opt = build_model(lr)

    train = BasicDataset(dir_img, noise_fraction, mode='train')
    test = BasicDataset(dir_img, noise_fraction, mode='test')
    val = BasicDataset(dir_img, noise_fraction, mode='val')
    # n_test = int(len(dataset) * test_percent)
    # n_train = len(dataset) - n_val
    # train, test = random_split(dataset, [n_train, n_test])

    # train_sampler = distributed.DistributedSampler(train)
    # test_sampler = distributed.DistributedSampler(test)
    # val_sampler = distributed.DistributedSampler(val)

    data_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=5, shuffle=False, num_workers=16, pin_memory=True)
    
    # data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.995, mode="train")
    # test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.5, mode="test")

    val_data, val_labels = next(iter(val_loader))
    val_data = val_data.cuda()
    val_labels = val_labels.cuda()

    data = iter(data_loader)
    loss = nn.CrossEntropyLoss(reduction="none")
    writer = SummaryWriter(comment=f'name_{args.figpath}')
    
    plot_step = 100
    accuracy_log = []
    net_losses = []
    acc_test = []
    acc_train = []
    loss_train = []
    net_l = 0
    global_step = 0
    test_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Noise fraction:  {noise_fraction}
        Image dir:       {dir_img}
        Model dir:       {fig_path}
    ''')


    for epoch in range(epochs):
        epoch_loss = 0
        correct_y = 0
        num_y = 0
        test_num = 0
        correct_num = 0
        for i in tqdm(range(len(data_loader))):
            net.train()
            # Line 2 get batch of data
            try:
                image, labels = next(data)
            except StopIteration:
                data = iter(data_loader)
                image, labels = next(data)

            image = image.cuda()
            labels = labels.cuda()
            image.requires_grad = False
            labels.requires_grad = False

            with higher.innerloop_ctx(net, opt) as (meta_net, meta_opt):
                y_f_hat = meta_net(image)            
                cost = loss(y_f_hat, labels)
                eps = torch.zeros(cost.size())
                eps = eps.requires_grad_()
                l_f_meta = torch.sum(cost * eps)
                meta_opt.step(l_f_meta)

                y_g_hat = meta_net(val_data)
                l_g_meta = torch.mean(loss(y_g_hat, val_labels))
                grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
                # print(type(grad_eps))
            
            # Line 11 computing and normalizing the weights
            w_tilde = torch.clamp(-grad_eps, min=0)
            norm_c = torch.sum(w_tilde)

            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde

            # Lines 12 - 14 computing for the loss with the computed weights
            # and then perform a parameter update
            # with torch.no_grad():
            y_f_hat = net(image)
            _, y_predicted = torch.max(y_f_hat, 1)
            correct_y = correct_y + (y_predicted.int() == labels.int()).sum().item()
            num_y = num_y + labels.size(0) 
            writer.add_scalar('StepAccuracy/train', ((y_predicted.int() == labels.int()).sum().item()/labels.size(0)), global_step)
            
            cost = loss(y_f_hat, labels)

            # cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
            l_f = torch.sum(cost * w)
            net_losses.append(l_f.item())
            writer.add_scalar('StepLoss/train', l_f.item(), global_step)
            epoch_loss = epoch_loss + l_f.item()

            opt.zero_grad()
            l_f.backward()
            opt.step()
            
            if i % plot_step == 0:
                net.eval()

                acc = []
                for m, (test_img, test_label) in enumerate(test_loader):
                    test_img = to_var(test_img, requires_grad=False)
                    test_label = to_var(test_label, requires_grad=False)

                    with torch.no_grad():
                        output = net(test_img)
                    _, predicted = torch.max(output, 1)
                    # print(type(predicted))
                    # predicted = to_var(predicted, requires_grad=False)
                    # print(type(predicted))
                    # test_label = test_label.float()

                    # print(type((predicted == test_label).float()))
                    test_num = test_num + test_label.size(0)
                    correct_num = correct_num + (predicted.int() == test_label.int()).sum().item()
                    acc.append((predicted.int() == test_label.int()).float())
                    writer.add_scalar('StepAccuracy/test', ((predicted.int() == test_label.int()).sum().item()/test_label.size(0)), test_step)
                    test_step = test_step + 1
                
        print('epoch ', epoch)

        print('epoch loss: ', epoch_loss/len(data_loader))
        loss_train.append(epoch_loss/len(data_loader))
        writer.add_scalar('EpochLoss/train', epoch_loss/len(data_loader), epoch)

        print('epoch accuracy: ', correct_y/num_y)
        acc_train.append(correct_y/num_y)
        writer.add_scalar('EpochAccuracy/train', correct_y/num_y, epoch)

        # path = 'baseline/' + args.figpath + '_model.pth'
        # path = 'baseline/' + str(args.noise_fraction) + '/model.pth'
        # torch.save(net.state_dict(), path)

        print('test accuracy: ', correct_num/test_num)
        writer.add_scalar('EpochAccuracy/test', correct_num/test_num, epoch)
        acc_test.append(correct_num/test_num)

        path = 'baseline/' + fig_path + '_model.pth'
        torch.save(net.state_dict(), path) 
    
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

    plt.savefig(fig_path+'.png')
        # return accuracy
    return net


def get_args():
    parser = argparse.ArgumentParser(description='Learning to reweight on classification tasks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-i', '--imgs-dir', metavar='ID', type=str, nargs='?', default='ISIC_2019_Training_Input/',
                        help='image path', dest='imgs_dir')
    parser.add_argument('-n', '--noise-fraction', metavar='NF', type=float, nargs='?', default=0.2,
                        help='Noise Fraction', dest='noise_fraction')
    parser.add_argument('-c', '--checkpoint-dir', metavar='CD', type=str, nargs='?', default='checkpoints/ISIC_2019_Training_Input/',
                        help='checkpoint path', dest='dir_checkpoint')
    parser.add_argument('-f', '--fig-path', metavar='FP', type=str, nargs='?', default='baseline',
                        help='Fig Path', dest='figpath')
    # parser.add_argument('-d', '--device-id', metavar='DI', type=str, nargs='?', default='0',
    #                   help='divices tot use', dest='device_id')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    try:
        net = train_net(lr=args.lr, 
                                  fig_path=args.figpath,
                                  momentum=0.9, 
                                  batch_size=args.batch_size, 
                                  dir_img=args.imgs_dir,
                                  save_cp=True,
                                  dir_checkpoint=args.dir_checkpoint,
                                  noise_fraction=args.noise_fraction,
                                  epochs=args.epochs)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    