import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import model
import backbone
from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file, TripletLoss, test_dict
import torchvision
import torchvision.transforms as transforms
import attack
import matplotlib.pyplot as plt
import warnings
from gradcam import *
from TinyImagenet import *
import distortions
warnings.filterwarnings("ignore")

SAVE_DIR = '/DATA1/puneet/interpretable/checkpoints'

def train(trainloader,teacher,params,config): 
    if params.exp == 'gbp':
        gbp= GuidedBackprop(teacher)
    elif params.exp == 'sgrad':
        gbp= SmoothGrad(teacher)
    elif params.exp == 'gcam++':
        gbp= GradCAMpp('resnet', 'layer4',teacher)
    
    epsilon = params.e/255.
    correct ,total =0,0
    for i, (x,y) in enumerate(trainloader):
        x,y = x.cuda(), y.cuda()
        target_mask = []
        for j in range(x.size(0)):
            m = gbp.forward(x[j].unsqueeze(0))
            target_mask.append(m)
        target_mask = torch.cat(target_mask,0)
        x = x - epsilon*target_mask
        x = torch.clamp(x, 0, 1)
        scores,_ = teacher(x)

        predicted = torch.argmax(scores,1)
        correct += (predicted==y).sum().item()
        total += predicted.size(0)
        print('Test Acc {:f}'.format(100.*correct/total))

    return model

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
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./../root_cifar', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=12)

        testset = torchvision.datasets.CIFAR10(root='./../root_cifar', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=False, num_workers=12)
        config = {
            'epsilon': 8.0 / 255,
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
            'epsilon': 1.0 / 255,
            'num_steps': 10,
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

    
    teacher = model.Model(net='_'.join(params.teacher.split('_')[:-1]),num_classes= params.num_classes)
    teacher_dir = '%s/%s/teacher/%s' %( SAVE_DIR,params.dataset, params.teacher)
    teacher_file = get_resume_file(teacher_dir)
    print('Teacher file:',teacher_file)
    tmp = torch.load(teacher_file)
    teacher.feature.load_state_dict(tmp['feature'])
    teacher.classifier.load_state_dict(tmp['classifier'])
    teacher.eval()


    teacher.cuda()
    train(testloader, teacher, params,config)
