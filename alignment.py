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
import distortions
from  scipy import stats
warnings.filterwarnings("ignore")

SAVE_DIR = '/DATA1/puneet/interpretable/checkpoints'

def alignment(trainloader,teacher,org,params,config): 
    gbp= SmoothGrad(teacher)
    attacker  = attack.AttackPGD(config) 
    n = 0
    avg_sim = 0.
    p = 1. - np.power(0.6,params.iter/10.0)
    for i, (x,y) in enumerate(trainloader):
        x,y = x.cuda(), y.cuda()

        target_mask = []
        for j in range(x.size(0)):
            m = gbp.forward(x[j].unsqueeze(0))
            target_mask.append(m)
        target_mask = torch.cat(target_mask,0).cuda()
        # target_mask = target_mask.view(target_mask.size(0),-1)
        
        noise = torch.sign(torch.FloatTensor(x.size()).uniform_(-1,1)).cuda()
        prob = torch.Tensor(x.size()).fill_(p)
        m = torch.distributions.bernoulli.Bernoulli(prob)
        w = m.sample()
        w = w.cuda()
        grad = (1-w)*noise + w*target_mask
        grad =  grad.view(grad.size(0),-1)

        x.requires_grad = True
        loss =F.cross_entropy(org(x)[0],y)
        loss.backward()

        delta = torch.sign(x.grad)
        delta = delta.view(delta.size(0),-1)

        sim = torch.abs(torch.nn.CosineSimilarity(1)(delta,-grad))
        avg_sim += sim.sum()
        n += x.size(0)
        print(avg_sim/n)

def adv(trainloader,teacher,org,params,config): 
    gbp= GuidedBackprop(teacher)
    gbp_org = GuidedBackprop(org)
    n = 0
    avg_sim = 0.
    for i, (x,y) in enumerate(trainloader):
        x,y = x.cuda(), y.cuda()

        target_mask = []
        for j in range(x.size(0)):
            m = gbp.forward(x[j].unsqueeze(0))
            target_mask.append(m)
        target_mask = torch.cat(target_mask,0).cuda()
        target_mask = torch.abs(target_mask.view(target_mask.size(0),-1))

        target_mask2 = []
        for j in range(x.size(0)):
            m = gbp_org.forward(x[j].unsqueeze(0))
            target_mask2.append(m)
        target_mask2 = torch.cat(target_mask2,0).cuda()
        target_mask2 = torch.abs(target_mask2.view(target_mask2.size(0),-1))
        

        sim = stats.spearmanr(target_mask.cpu().numpy(),target_mask2.cpu().numpy(),axis=1)[0]
        avg_sim += sim.sum()
        n += x.size(0)
        print(avg_sim/n)

        

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

        trainset = torchvision.datasets.CIFAR100(root='./../root_cifar100', train = True, download=True,transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR100(root='./../root_cifar100', train = False,  download=True,transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=False, num_workers=0)
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

    org = model.Model(net='ResNet10',num_classes= params.num_classes)
    teacher_dir = '%s/%s/teacher/%s' %( SAVE_DIR,params.dataset, 'ResNet10_adv')
    teacher_file = get_assigned_file(teacher_dir,params.iter)
    print('Org file:',teacher_file)
    tmp = torch.load(teacher_file)
    org.feature.load_state_dict(tmp['feature'])
    org.classifier.load_state_dict(tmp['classifier'])
    org.cuda()

    alignment(testloader, teacher, org, params,config)
