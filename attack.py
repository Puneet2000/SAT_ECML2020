import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import autograd
from io_utils import *
import sys
import copy

class AttackPGD(nn.Module):
    def __init__(self, config,student=False):
        super(AttackPGD, self).__init__()
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.student = student
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'

    def attack(self, basic_net,inputs, targets):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits,_ = basic_net(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return x

    def attack3(self, basic_net,inputs, targets,target_mask):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        m = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.5))
        if int(m.sample()) == 1:
            x = x.detach() - self.epsilon*target_mask
            x = torch.clamp(x, 0, 1)
        else:
            for i in range(self.num_steps):
                x.requires_grad_()
            
                with torch.enable_grad():
                    logits,_ = basic_net(x)
                    loss = F.cross_entropy(logits, targets, size_average=False)
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.step_size*torch.sign(grad.detach())
                x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
                x = torch.clamp(x, 0, 1)

        return x
        
    def attack4(self, basic_net,inputs, targets):
        m = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.5))
        if int(m.sample()) == 1:
            x_adv = inputs.detach() + torch.zeros_like(inputs).uniform_(-8/255., 8/255.).cuda()
            x_adv = torch.clamp(x_adv, 0, 1)
        else:
            x_adv = self.attack(basic_net,inputs,targets)

        return x_adv

class AttackTRADES(nn.Module):
    def __init__(self, config,student=False):
        super(AttackTRADES, self).__init__()
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.criterion_kl = nn.KLDivLoss(size_average=False)

    def attack(self, basic_net,inputs, targets):
        x = inputs.detach()
        x_adv = x + 0.001 * torch.randn(x.shape).cuda().detach()
        for i in range(self.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = self.criterion_kl(F.log_softmax(basic_net(x_adv)[0], dim=1),F.softmax(basic_net(x)[0], dim=1))
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size*torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x- self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

    def attack3(self, basic_net,inputs, targets,target_mask):
        m = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.5))
        if int(m.sample()) == 1:
            x_adv = inputs.detach() - self.epsilon*target_mask
            x_adv = torch.clamp(x_adv, 0, 1)
        else:
            x_adv = self.attack(basic_net,inputs,targets)

        return x_adv

    def attack4(self, basic_net,inputs, targets):
        m = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.5))
        if int(m.sample()) == 1:
            x_adv = inputs.detach() + torch.zeros_like(inputs).uniform_(-8/255., 8/255.).cuda()
            x_adv = torch.clamp(x_adv, 0, 1)
        else:
            x_adv = self.attack(basic_net,inputs,targets)

        return x_adv