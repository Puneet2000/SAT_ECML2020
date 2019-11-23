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

def train(trainloader, model, teacher,optimization, start_epoch, stop_epoch, params,config): 

    
    attacker  = attack.AttackTRADES(config) 
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loss_fn2 = nn.MSELoss()
    epsilon = params.e/255.
    criterion_kl = nn.KLDivLoss(size_average=False)
    for epoch in range(start_epoch,stop_epoch):
        t=0
        model.train()
        print_freq = 50
        avg_closs=0
        avg_mloss=0
        correct ,total =0,0
        p = 1. - np.power(params.a,epoch/10.0)
        for i, (x,y,target_mask) in enumerate(trainloader):
            x,y = x.cuda(), y.cuda()
            target_mask = target_mask.cuda()
            a = time.time()
            noise = torch.sign(torch.FloatTensor(x.size()).uniform_(-1,1)).cuda()
            prob = torch.Tensor(x.size()).fill_(p)
            m = torch.distributions.bernoulli.Bernoulli(prob)
            w = m.sample()
            w = w.cuda()
            grad = (1-w)*noise + w*target_mask

            x_adv = attacker.attack3(model,x,y,grad)

            scores,_ = model(x)
            closs = loss_fn(scores,y) + (1.0 / x.size(0)) * criterion_kl(F.log_softmax(model(x_adv)[0], dim=1),F.softmax(model(x)[0], dim=1))

            predicted = torch.argmax(scores,1)
            correct += (predicted==y).sum().item()
            total += predicted.size(0)

            optimizer.zero_grad()
            closs.backward()
            optimizer.step()
            b = time.time()
            avg_closs += closs.data.item()
            t += b-a
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | CLoss {:f}  | Train Acc {:f}'.format(epoch, i, len(trainloader), avg_closs/float(i+1),100.*correct/total))
                print(t)
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

def distort(image,func):
    image = image.permute(1,2,0)
    image = image.cpu().numpy()
    image = func(image)
    return torch.from_numpy(image).float().permute(2,0,1)

def test_distortion(testloader, model,params):    
    model.eval()
    func, vals = test_dict[params.test]
    for val in vals:
        print('Testing at value {:f}'.format(val))
        correct ,total =0,0
        for i, (x,y) in enumerate(testloader):
            x_ = []
            for j in range(x.size(0)):
                x_.append(distort(x[j],lambda z: func(z,val)))
            x = torch.stack(x_,0)
            x,y = x.cuda(), y.cuda()
            scores,_= model(x)
            predicted = torch.argmax(scores,1)
            correct += (predicted==y).sum().item()
            total += predicted.size(0)
        print('Accuracy {:f}'.format(100.*correct/total))

def test_adv(testloader, model,params,config): 
    model.eval()
    _,epsilons = test_dict['adv']
    for e in epsilons:
        print('Testing at epsilon {:f}'.format(e))
        config['epsilon'] = e
        attacker  = attack.AttackPGD(config)
        correct ,total =0,0
        for i, (x,y) in enumerate(testloader):
            x,y = x.cuda(), y.cuda()
            x = attacker.attack(model,x,y)
            scores,_ = model(x)
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

        trainset = torchvision.datasets.CIFAR100(root='./../root_cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=12)

        testset = torchvision.datasets.CIFAR100(root='./../root_cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=True, num_workers=12)
        config = {
            'epsilon': 4.0 / 255,
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
        trainset = TinyImageNet('/DATA1/tiny-imagenet-200', 'train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=2)

        # testset = torchvision.datasets.ImageFolder('/DATA1/puneet/tiny-imagenet-200/val', transform_test)
        testset = TinyImageNet('/DATA1/tiny-imagenet-200', 'val',transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=False, num_workers=2)
        config = {
            'epsilon': 8.0 / 255,
            'num_steps': 5,
            'step_size': 2.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }
    elif params.dataset == 'flower':
        params.num_classes = 17
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
        trainset = Flower('/home/puneet/interpretable/saliency2/dataset/17flowers', 'train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=12)

        # testset = torchvision.datasets.ImageFolder('/DATA1/puneet/tiny-imagenet-200/val', transform_test)
        testset = Flower('/home/puneet/interpretable/saliency2/dataset/17flowers', 'val',transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=params.bs, shuffle=False, num_workers=12)
        config = {
            'epsilon': 1.0 / 255,
            'num_steps': 5,
            'step_size': 2.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }

  

    model = model.Model(net=params.model,num_classes= params.num_classes)


    optimization = 'Adam'
    params.checkpoint_dir = '%s/%s/student_adv/%s_%s' %( SAVE_DIR,params.dataset, params.model, 'bbox_trades')
    if params.exp != 'gbp':
        params.checkpoint_dir += '_%s'%(params.exp)
    if params.e !=8.0 :
        params.checkpoint_dir += '_eps{}'.format(params.e)
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
        model = train(trainloader, model, None, optimization, start_epoch, stop_epoch, params,config)
    elif params.test == 'adv':
        test_adv(testloader,model,params,config)
    else:
        test_distortion(testloader,model,params)
