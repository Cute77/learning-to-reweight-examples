import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
from data_loader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import IPython
import gc
import matplotlib
import torchvision
from datasets import BasicDataset
from torch.utils.data import DataLoader, random_split


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def build_model():
    net = torchvision.models.resnet101(pretrained=True)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark=True

    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])
    
    return net, opt


def train_net(lr=1e-3,
              momentum=0.9, 
              batch_size=100,
              num_iterations=8000, 
              dir_img=args.imgs_dir,
              save_cp=True,
              dir_checkpoint='checkpoints/'):

    train = BasicDataset(dir_img, mode='train')
    test = BasicDataset(dir_img, mode='test')
    # n_test = int(len(dataset) * test_percent)
    # n_train = len(dataset) - n_val
    # train, test = random_split(dataset, [n_train, n_test])
    data_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.995, mode="train")
    # test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.5, mode="test")

    val_data = to_var(data_loader.dataset.val_data, requires_grad=False)
    val_labels = to_var(data_loader.dataset.val_label, requires_grad=False)

    net, opt = build_model()
    
    meta_losses_clean = []
    net_losses = []
    plot_step = 100

    smoothing_alpha = 0.9
    
    meta_l = 0
    net_l = 0
    accuracy_log = []
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        # Line 2 get batch of data
        image, labels = next(iter(data_loader))
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_net = torchvision.models.resnet101(pretrained=True)
        meta_net.load_state_dict(net.state_dict())

        if torch.cuda.is_available():
            meta_net.cuda()

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat  = meta_net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat,labels, reduce=False)
        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)

        meta_net.zero_grad()

                # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(hyperparameters['lr'], source_params=grads)
        
        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)

        l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat,val_labels)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
        
        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps,min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
        l_f = torch.sum(cost * w)

        opt.zero_grad()
        l_f.backward()
        opt.step()
        
        if i % plot_step == 0:
            net.eval()

            acc = []
            for (test_img, test_label) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = (F.sigmoid(output) > 0.5).numpy()

                acc.append((predicted.numpy() == test_label.numpy()).float())

            accuracy = torch.cat(acc,dim=0).mean()
            accuracy_log.append(np.array([i,accuracy])[None])
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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=100,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--imgs_dir', metavar='ID', type=str, nargs='?', default='ISIC_2019_Training_Input/',
                        help='Learning rate', dest='lr')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    try:
        accuracy = train_net(lr=args.lr,
              momentum=0.9, 
              batch_size=args.batch_size,
              num_iterations=8000, 
              dir_img=args.imgs_dir,
              save_cp=True,
              dir_checkpoint='checkpoints/')
        print('Test Accuracy: ', accuracy)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    