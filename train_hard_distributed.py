import sys
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
from torch.optim.lr_scheduler import StepLR
import higher 

os.environ["CUDA_VISIBEL_DEVICES"] = "4,5"


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


def build_model(lr, local_rank):
    net = models.resnet101(pretrained=True, num_classes=9)
    net = net.cuda(local_rank)
    opt = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': lr}], lr, weight_decay=1e-4)
    
    return net, opt


def train_net(noise_fraction, 
              fig_path,
              local_rank,
              lr=1e-3,
              momentum=0.9, 
              batch_size=128,
              dir_img='ISIC_2019_Training_Input/',
              save_cp=True,
              dir_checkpoint='checkpoints/ISIC_2019_Training_Input/',
              epochs=10, 
              load=-1):

    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    is_distributed = num_gpus > 1
    lr = lr * num_gpus
    
    path = 'baseline/' + fig_path + '_' + str(load) + '_model.pth'
    if is_distributed:
        torch.cuda.set_device(local_rank) 
        print("local_rank:", local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )    
        # net, opt = build_model(lr, local_rank)
        # synchronize()
        net, opt = build_model(lr, local_rank)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, 
        )
        if os.path.isfile(path):
            logging.info(f'''Continue''')
            net.load_state_dict(torch.load(path))

    else:
        net, opt = build_model(lr, local_rank)
        if os.path.isfile(path):
            logging.info(f'''Continue''')
            net.load_state_dict(torch.load(path))
        # net, opt = build_model(lr, local_rank)
    
    train = BasicDataset(dir_img, noise_fraction, mode='train')
    test = BasicDataset(dir_img, noise_fraction, mode='test')
    val = BasicDataset(dir_img, noise_fraction, mode='val')
    # n_test = int(len(dataset) * test_percent)
    # n_train = len(dataset) - n_val
    # train, test = random_split(dataset, [n_train, n_test])

    train_sampler = distributed.DistributedSampler(train, num_replicas=num_gpus, rank=local_rank) if is_distributed else None

    data_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=5, shuffle=False, num_workers=8, pin_memory=True)
    
    # data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.995, mode="train")
    # test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.5, mode="test")

    val_data, val_labels = next(iter(val_loader))
    if is_distributed:
        val_data = val_data.cuda(local_rank)
        val_labels = val_labels.cuda(local_rank)
    else:
        val_data = val_data.cuda()
        val_labels = val_labels.cuda()

    data = iter(data_loader)
    loss = nn.CrossEntropyLoss(reduction="none")
    writer = SummaryWriter(comment=f'name_{args.figpath}')
    scheduler = StepLR(opt, step_size=50, gamma=0.5, last_epoch=load)
    
    plot_step = 10
    net_losses = []
    acc_test = []
    acc_train = []
    train_iter = []
    test_iter = []
    loss_train = []
    global_step = 0
    test_step = 0
    mixup_labels = torch.ones([32, 9]).cuda(local_rank)
    mixup_labels.requires_grad = True

    if local_rank == 0:
        logging.info(f'''Starting training:
            Devices:         {num_gpus}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Checkpoints:     {save_cp}
            Noise fraction:  {noise_fraction}
            Image dir:       {dir_img}
            Model dir:       {fig_path}
            From epoch:      {load}
        ''')

    meta_net = models.resnet101(pretrained=True, num_classes=9)
    if torch.cuda.is_available():
        if is_distributed:
            meta_net.cuda(local_rank)
        else:
            meta_net.cuda()

    if is_distributed:
        meta_net = torch.nn.parallel.DistributedDataParallel(
            meta_net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True,
        )

    for epoch in range(load+1, epochs):
        net.train()
        epoch_loss = 0
        correct_y = 0
        num_y = 0
        test_num = 0
        correct_num = 0
        for i in range(len(data_loader)):
            print(i)
            # print('train: ', len(train))
            # print(len(data_loader))
            # Line 2 get batch of data
            try:
                image, labels = next(data)
            except StopIteration:
                data = iter(data_loader)
                image, labels = next(data)

            # meta_net.load_state_dict(net.state_dict())
            if is_distributed:
                image = image.cuda(local_rank)
                labels = labels.cuda(local_rank)
            else:
                image = image.cuda()
                labels = labels.cuda()                
            image.requires_grad = False
            labels.requires_grad = False
            
            with higher.innerloop_ctx(net, opt) as (meta_net, meta_opt):
                y_f_hat = meta_net(image)
                cost = loss(y_f_hat, labels)
                eps = torch.zeros(cost.size()).cuda(local_rank)
                eps = eps.requires_grad_()
                l_f_meta = torch.sum(cost * eps)
                # meta_net.zero_grad()
                meta_opt.step(l_f_meta)
                # grads = torch.autograd.grad(l_f_meta, (meta_net.parameters()), create_graph=True, retain_graph=True)
                # meta_net.module.update_parameters(lr, source_parameters=grads)
                y_g_hat = meta_net(val_data)
        
                #loss = nn.CrossEntropyLoss()
                l_g_meta = torch.mean(loss(y_g_hat, val_labels))
                # print(l_g_meta)
                # print(eps)
                # l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat, val_labels)

                grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, create_graph=True, retain_graph=True, allow_unused=True)[0].detach()
                print("eps: ", grad_eps)
                # print(grad_eps)
                # Line 11 computing and normalizing the weights

            beta_tilde = torch.sigmoid(-grad_eps)
            # print(w_tilde)
            # norm_c = torch.sum(beta_tilde)

            # if norm_c != 0:
            #     beta = beta_tilde / norm_c
            # else:
            #     beta = beta_tilde
            beta = beta_tilde

            if epoch == 11 or epoch == 21 or epoch == 31 or epoch == 101 or epoch == 151:                 
                if i == 0:
                    bs = beta
                else:
                    bs = torch.cat([bs, beta])

            # Lines 12 - 14 computing for the loss with the computed weights
            # and then perform a parameter update
            # with torch.no_grad():
            y_f_hat = net(image)

            _, y_predicted = torch.max(y_f_hat, 1)
            correct_y = correct_y + (y_predicted.int() == labels.int()).sum().item()
            num_y = num_y + labels.size(0) 
            writer.add_scalar('StepAccuracy/train', ((y_predicted.int() == labels.int()).sum().item()/labels.size(0)), global_step)
            train_iter.append((y_predicted.int() == labels.int()).sum().item())
            
            beta = beta.cuda(local_rank)
            mixup_labels = beta * labels + (1-beta) * y_predicted
            print('beta: ', beta)
            print('labels: ', labels)
            print('y_predicted: ', y_predicted)
            print('mixup_labels: ', mixup_labels)
            # mixup_labels = mixup_labels.cpu()
            # mixup_labels = torch.LongTensor(mixup_labels).cuda(local_rank)
            cost = loss(y_f_hat, mixup_labels.long())
            w = torch.ones(cost.size()).cuda(local_rank)
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

                for m, (test_img, test_label) in enumerate(test_loader):
                    test_img = test_img.cuda(local_rank)
                    test_label = test_label.cuda(local_rank)
                    test_img.requires_grad = False
                    test_label.requires_grad = False

                    with torch.no_grad():
                        output = net(test_img)
                    _, predicted = torch.max(output, 1)
                    # print(type(predicted))
                    # predicted = to_var(predicted, requires_grad=False)
                    # print(type(predicted))
                    # test_label = test_label.float()

                    # print(type((predicted == test_label).float()))
                    test_num = test_num + test_label.size(0)
                    # print(test_num)
                    correct_num = correct_num + (predicted.int() == test_label.int()).sum().item()
                    # acc.append((predicted.int() == test_label.int()).float())
                    writer.add_scalar('StepAccuracy/test', ((predicted.int() == test_label.int()).sum().item()/test_label.size(0)), test_step)
                    test_iter.append((predicted.int() == test_label.int()).sum().item())
                    test_step = test_step + 1
        '''
        print('epoch ', epoch)

        print('epoch loss: ', epoch_loss/len(train))
        loss_train.append(epoch_loss/len(train))
        writer.add_scalar('EpochLoss/train', epoch_loss/len(train), epoch)

        print('epoch accuracy: ', correct_y/num_y)
        acc_train.append(correct_y/num_y)
        writer.add_scalar('EpochAccuracy/train', correct_y/num_y, epoch)

        # path = 'baseline/' + args.figpath + '_model.pth'
        # path = 'baseline/' + str(args.noise_fraction) + '/model.pth'
        # torch.save(net.state_dict(), path)

        print('test accuracy: ', correct_num/test_num)
        writer.add_scalar('EpochAccuracy/test', correct_num/test_num, epoch)
        acc_test.append(correct_num/test_num)
        '''

        if (epoch == 11 or epoch == 21 or epoch == 31 or epoch == 101 or epoch == 151) and local_rank == 0:
            bs = bs.cpu().numpy().tolist()
            plt.hist(x=bs, bins=20)
            plt.savefig(fig_path+'_'+str(epoch)+'_beta.png')
            print('beta saved')

        if is_distributed and local_rank == 0 and epoch % 5 == 0:
            path = 'baseline/' + fig_path + '_' + str(epoch) + '_model.pth'
            torch.save(net.state_dict(), path) 

        if is_distributed and local_rank == 0:
            torch.save(net.state_dict(), path) 
            print('local_rank: ', local_rank)
            print('epoch ', epoch)

            print('epoch loss: ', epoch_loss/len(data_loader))
            loss_train.append(epoch_loss/len(data_loader))
            writer.add_scalar('EpochLoss/train', epoch_loss/len(data_loader), epoch)

            print('epoch accuracy: ', correct_y/num_y)
            acc_train.append(correct_y/num_y)
            writer.add_scalar('EpochAccuracy/train', correct_y/num_y, epoch)

            print('test accuracy: ', correct_num/test_num)
            writer.add_scalar('EpochAccuracy/test', correct_num/test_num, epoch)
            acc_test.append(correct_num/test_num)

        if not is_distributed and epoch % 5 == 0:
            path = 'baseline/' + fig_path + '_' + str(epoch) + '_model.pth'
            torch.save(net.state_dict(), path)

        if not is_distributed:
            # torch.save(net.state_dict(), path)
            print('epoch ', epoch)

            print('epoch loss: ', epoch_loss/len(data_loader))
            loss_train.append(epoch_loss/len(train))
            writer.add_scalar('EpochLoss/train', epoch_loss/len(data_loader), epoch)

            print('epoch accuracy: ', correct_y/num_y)
            acc_train.append(correct_y/num_y)
            writer.add_scalar('EpochAccuracy/train', correct_y/num_y, epoch)

            print('test accuracy: ', correct_num/test_num)
            writer.add_scalar('EpochAccuracy/test', correct_num/test_num, epoch)
            acc_test.append(correct_num/test_num)   

    IPython.display.clear_output()
    fig, axes = plt.subplots(2, 3)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    ax1.plot(net_losses, label='train_losses')
    ax1.set_ylabel("Losses")
    ax1.set_xlabel("Iteration")
    ax1.legend()

    ax2.plot(loss_train, label='epoch_losses')
    ax2.set_ylabel('Losses/train')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    ax3.plot(acc_train, label='acc_train_epoch')
    ax3.set_ylabel('Accuracy/trainEpoch')
    ax3.set_xlabel('Epoch')
    ax3.legend()

    ax4.plot(train_iter, label='acc_train_iteration')
    ax4.set_ylabel('Accuracy/trainIteration')
    ax4.set_xlabel('Iteration')
    ax4.legend()

    ax5.plot(acc_test, label='acc_test_epoch')
    ax5.set_ylabel('Accuracy/testEpoch')
    ax5.set_xlabel('Epoch')
    ax5.legend()

    ax6.plot(test_iter, label='acc_train_iteration')
    ax6.set_ylabel('Accuracy/trainIteration')
    ax6.set_xlabel('Iteration')
    ax6.legend()

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
    parser.add_argument('-r', '--local_rank', metavar='RA', type=int, nargs='?', default=0,
                        help='from torch.distributed.launch', dest='local_rank')
    parser.add_argument('-o', '--load', metavar='LO', type=int, nargs='?', default=0,
                        help='load epoch', dest='load')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    torch.backends.cudnn.benchmark = True
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
                                  epochs=args.epochs,
                                  local_rank=args.local_rank, 
                                  load = args.load)
        # print('Test Accuracy: ', accuracy)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    