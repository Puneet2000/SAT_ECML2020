import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import model
import backbone
from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file
import torchvision
import torchvision.transforms as transforms
import attack
import matplotlib.pyplot as plt
import warnings
from TinyImagenet import *
import torchvision.models as models
warnings.filterwarnings("ignore")

SAVE_DIR = '/DATA1/puneet/interpretable/checkpoints'

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(trainloader, model, optimization, start_epoch, stop_epoch, params,config):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')
    loss_fn = nn.CrossEntropyLoss()
    pgd = attack.AttackPGD(config)
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        print_freq = 50
        avg_loss=0
        correct ,total =0,0
        for i, (x,y) in enumerate(trainloader):
            x,y = x.cuda(), y.cuda()
            x_m,y_a,y_b,lam = mixup_data(x,y)
            optimizer.zero_grad()
            scores_m,_ = model.forward(x_m)
            scores,_ = model.forward(x)
            predicted = torch.argmax(scores,1)
            correct += (predicted==y).sum().item()
            total += predicted.size(0)
            loss = mixup_criterion(loss_fn,scores_m,y_a,y_b,lam)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.data.item()

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Train Acc {:f}'.format(epoch, i, len(trainloader), avg_loss/float(i+1),100.*correct/total ))

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            state_dict = {}
            state_dict['epoch'] = epoch
            state_dict['feature'] = model.feature.state_dict()
            state_dict['classifier'] =  model.classifier.state_dict()
            torch.save(state_dict, outfile)

    return model

def test(testloader, model,params,config):    
    pgd = attack.AttackPGD(config)
    model.eval()
    correct ,total =0,0
    for i, (x,y) in enumerate(testloader):
        x,y = x.cuda(), y.cuda()
        # print(y)
        x = pgd.attack(model,x,y)
        scores,_ = model.forward(x)
        predicted = torch.argmax(scores,1)
        correct += (predicted==y).sum().item()
        total += predicted.size(0)
        print('Accuracy {:f}'.format(100.*correct/total))


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args()
    print(params)

    if params.dataset == 'cifar10':
        params.num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./../root_cifar', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=12)

        testset = torchvision.datasets.CIFAR10(root='./../root_cifar', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=True, num_workers=12)
        config = {
            'epsilon': 3.0 / 255,
            'num_steps': 5,
            'step_size': 2.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }

    elif params.dataset == 'cifar100':
        params.num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./../root_cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=12)

        testset = torchvision.datasets.CIFAR100(root='./../root_cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=True, num_workers=12)
        config = {
            'epsilon': 1.0 / 255,
            'num_steps': 5,
            'step_size': 2.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }

    elif params.dataset == 'svhn':
        params.num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.SVHN(root='./../root_shvn', split='train', download=True,transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=2)

        testset = torchvision.datasets.SVHN(root='./../root_shvn', split='test',  download=True,transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=False, num_workers=2)
        config = {
            'epsilon': 8.0 / 255,
            'num_steps': 5,
            'step_size': 2.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }

    elif params.dataset == 'tiny-img':
        params.num_classes = 200
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # trainset = torchvision.datasets.ImageFolder('/DATA1/puneet/tiny-imagenet-200/train', transform_train)
        trainset = TinyImageNet('/DATA1/puneet/tiny-imagenet-200', 'train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=2)

        # testset = torchvision.datasets.ImageFolder('/DATA1/puneet/tiny-imagenet-200/val', transform_test)
        testset = TinyImageNet('/DATA1/puneet/tiny-imagenet-200', 'train',transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=False, num_workers=2)
        config = {
            'epsilon': 8.0 / 255,
            'num_steps': 5,
            'step_size': 2.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }


    model = model.Model(net=params.model,num_classes= params.num_classes)
    # model = nn.DataParallel(model,device_ids=[0,1])
    optimization = 'Adam'
    params.checkpoint_dir = '%s/%s/teacher/%s_%s' %( SAVE_DIR,params.dataset, params.model,'mixup')
    print('checkpoints dir',params.checkpoint_dir)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        if params.iter !=-1:
            resume_file = get_assigned_file(params.checkpoint_dir,params.iter)
        else:
            resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            print('Resume file is: ', resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.classifier.load_state_dict(tmp['classifier'])
            model.feature.load_state_dict(tmp['feature'])

    model.cuda()

    if params.test is None:
        model = train(trainloader, model, optimization, start_epoch, stop_epoch, params,config)
    else:
        test(testloader, model, params,config)
