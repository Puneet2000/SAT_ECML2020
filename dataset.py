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
from io_utils import model_dict, parse_args, get_resume_file
import torchvision
import torchvision.transforms as transforms
import attack
import matplotlib.pyplot as plt
import warnings
from gradcam import *
from utils import *
warnings.filterwarnings("ignore")


def make_dataset(trainloader,model):
    model.eval()
    dataset = []
    labels = []
    for i, (x,y) in enumerate(trainloader):
        print(i)
        x,y = x.cuda(), y.cuda()
        x_ = []
        p = dict(type='resnet', arch=model, layer_name='layer4', input_size=(32,32))
        gradcam_pp = GradCAMpp(p)
        for j in range(x.size(0)):
            mask_pp, _ = gradcam_pp(x[j].unsqueeze(0))
            xm = mask_pp*x[j].unsqueeze(0)
            x_.append(xm)
        x = torch.cat(x_,0).cuda()
        dataset.append(x.cpu().detach())
        labels.append(y.cpu().detach())
    return torch.cat(dataset,0), torch.cat(labels,0)

def test(testloader, model,params,config):    
    pgd = attack.AttackPGD(config)
    model.eval()
    correct ,total =0,0
    for i, (x,y) in enumerate(testloader):
        x,y = x.cuda(), y.cuda()
        x_ = []
        p = dict(type='resnet', arch=model, layer_name='layer4', input_size=(32,32))
        gradcam_pp = GradCAMpp(p)
        for j in range(x.size(0)):
            mask_pp, _ = gradcam_pp(x[j].unsqueeze(0))
            xm = mask_pp*x[j].unsqueeze(0)
            x_.append(xm)
        x = torch.cat(x_,0).cuda()
        scores = model.forward(x)
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
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
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
            'epsilon': 8.0 / 255,
            'num_steps': 10,
            'step_size': 2.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }

    elif params.dataset == 'mnist':
        pass
  
    model = model.Model(net=params.model,num_classes= params.num_classes)
    optimization = 'Adam'
    params.checkpoint_dir = './checkpoints/%s/%s_%s' %( params.dataset, params.model, params.method)
    print('checkpoints dir',params.checkpoint_dir)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            print('Resume file is: ', resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.classifier.load_state_dict(tmp['classifier'])
            model.feature.load_state_dict(tmp['feature'])

    # model = nn.DataParallel(model)
    model.cuda()

    # model = train(trainloader, model, optimization, start_epoch, stop_epoch, params,config)
    #test(testloader, model, params,config)
    dataset,labels = make_dataset(trainloader,model)
    outdir = './../robust_{}'.format(params.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    torch.save(dataset,'{}/dataset.pt'.format(outdir))
    torch.save(labels,'{}/labels.pt'.format(outdir))