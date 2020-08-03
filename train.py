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
import os
import argparse
import logging

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def build_model(lr):
    net = model.resnet101(pretrained=False, num_classes=9)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.params(), lr)
    
    return net, opt


def train_net(noise_fraction, 
              lr=1e-3,
              momentum=0.9, 
              batch_size=128,
              dir_img='ISIC_2019_Training_Input/',
              save_cp=True,
              dir_checkpoint='checkpoints/ISIC_2019_Training_Input/',
              epochs=10):

    train = BasicDataset(dir_img, noise_fraction, mode='train')
    test = BasicDataset(dir_img, noise_fraction, mode='test')
    val = BasicDataset(dir_img, noise_fraction, mode='val')
    # n_test = int(len(dataset) * test_percent)
    # n_train = len(dataset) - n_val
    # train, test = random_split(dataset, [n_train, n_test])
    data_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=50, shuffle=False, num_workers=8, pin_memory=True)
    
    # data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.995, mode="train")
    # test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.5, mode="test")

    val_data, val_labels = next(iter(val_loader))
    val_data = to_var(val_data, requires_grad=False)
    val_labels = to_var(val_labels, requires_grad=False)
    
    net, opt = build_model(lr)
    plot_step = 100
    accuracy_log = []
    data = iter(data_loader)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Noise fraction:  {noise_fraction}
        Image dir:       {dir_img}
    ''')

    for epoch in range(epochs):
        net.train()
        for i in tqdm(range(len(train))):
            # Line 2 get batch of data
            image, labels = next(data)
            # since validation data is small I just fixed them instead of building an iterator
            # initialize a dummy network for the meta learning of the weights
            meta_net = model.resnet101(pretrained=False, num_classes=9)
            meta_net.load_state_dict(net.state_dict())

            if torch.cuda.is_available():
                meta_net.cuda()

            image = to_var(image, requires_grad=False)
            labels = to_var(labels, requires_grad=False)

            # Lines 4 - 5 initial forward pass to compute the initial weighted loss
            # with torch.no_grad():
                # print(image.shape)
            y_f_hat = meta_net(image)
            
            labels = labels.float()
            cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
            # print('cost:', cost)
            eps = to_var(torch.zeros(cost.size()))
            # print('eps: ', eps)
            l_f_meta = torch.sum(cost * eps)

            meta_net.zero_grad()

            # Line 6 perform a parameter update
            grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True, allow_unused=True)
            meta_net.update_params(lr, source_params=grads)
            
            # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
            # with torch.no_grad():
            y_g_hat = meta_net(val_data)

            val_labels = val_labels.float()
            l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat, val_labels)

            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
            
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

            labels = labels.float()
            cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
            l_f = torch.sum(cost * w)

            opt.zero_grad()
            l_f.backward()
            opt.step()
            
            if i % plot_step == 0:
                net.eval()

                acc = []
                for i, (test_img, test_label) in enumerate(test_loader):
                    test_img = to_var(test_img, requires_grad=False)
                    test_label = to_var(test_label, requires_grad=False)

                    output = net(test_img)
                    predicted = (F.sigmoid(output) > 0.5)
                    # print(type(predicted))
                    # predicted = to_var(predicted, requires_grad=False)
                    # print(type(predicted))
                    # test_label = test_label.float()

                    # print(type((predicted == test_label).float()))
                    acc.append((predicted.float() == test_label.float()).float())

                accuracy = torch.cat(acc, dim=0).mean()
                accuracy_log.append(np.array([i, accuracy])[None])
                acc_log = np.concatenate(accuracy_log, axis=0)
                
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

        # return accuracy
    return np.mean(acc_log[-6:-1, 1])


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
    parser.add_argument('-c', '--checkpoint-dir', metavar='CD', type=str, nargs='?', default='checkpoints/ISIC_2019_Training_Input/',
                        help='checkpoint path', dest='dir_checkpoint')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    try:
        accuracy = train_net(lr=args.lr,
                             momentum=0.9, 
                             batch_size=args.batch_size, 
                             dir_img=args.imgs_dir,
                             save_cp=True,
                             dir_checkpoint=args.dir_checkpoint,
                             noise_fraction=args.noise_fraction)
        print('Test Accuracy: ', accuracy)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    