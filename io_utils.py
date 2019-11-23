import numpy as np
import os
import glob
import argparse
import backbone
import torch
import operator as op
import functools as ft
import torch.nn as nn
import torch
import torch.nn.functional as F
import distortions
model_dict = dict(
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101,
            ResNet152 = backbone.ResNet152,
            WideResNet28_10 = backbone.WideResNet28_10) 

test_dict =  dict(
             adv = (None,[1./255,2./255,3./255,4./255,5./255,6./255,7./255,8./255]),
             contrast = (distortions.adjust_contrast,[0.1,0.3, 0.5, 0.7, 0.9, 1]),
             low_pass = (distortions.low_pass_filter,[1,2,4,8,16, 32]),
             high_pass = (distortions.high_pass_filter,[2.82,2 ,1.414,1,0.7,0.5])
    )

def parse_args():
    parser = argparse.ArgumentParser(description= 'ATCNN')
    parser.add_argument('--dataset'     , default='cifar10',        help='cifar10/svhn')
    parser.add_argument('--model'       , default='ResNet34',      help='model: ResNet{10|34|50}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='std',   help='std/adv') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--num_classes' , default=10, type=int, help='total number of classes in softmax') #make it larger than the maximum label value in base class
    parser.add_argument('--save_freq'   , default=5, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch'  , default=100, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
    parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('--bs' , default=64, type=int, help='batch-size')
    parser.add_argument('--iter' , default=-1, type=int, help='pick a certain resume file')
    parser.add_argument("--device",nargs="*", type=int,default=[0])
    parser.add_argument('--teacher'       , default='ResNet10',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    parser.add_argument('--test'      , default=None, help='continue from previous trained model with largest epoch')
    parser.add_argument('--e'      , default=8.0, type=float, help='continue from previous trained model with largest epoch')
    parser.add_argument('--a'      , default=0.6, type=float, help='continue from previous trained model with largest epoch')
    parser.add_argument('--exp'       , default='gbp',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--teacher2'       , default='ResNet10',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    parser.add_argument('--exp2'       , default='gbp',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}')
    
    return parser.parse_args()

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def reduce_mean(x, keepdim=True):
    numel = ft.reduce(op.mul, x.size()[1:])
    x = reduce_sum(x, keepdim=keepdim)
    return x / numel


def reduce_min(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.min(a, keepdim=keepdim)[0]
    return x


def reduce_max(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.max(a, keepdim=keepdim)[0]
    return x


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def l2r_dist(x, y, keepdim=True, eps=1e-8):
    d = (x - y)**2
    d = reduce_sum(d, keepdim=keepdim)
    d += eps  # to prevent infinite gradient at 0
    return d.sqrt()


def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)


def l1_dist(x, y, keepdim=True):
    d = torch.abs(x - y)
    return reduce_sum(d, keepdim=keepdim)


def l2_norm(x, keepdim=True):
    norm = reduce_sum(x*x, keepdim=keepdim)
    return norm.sqrt()


def l1_norm(x, keepdim=True):
    return reduce_sum(x.abs(), keepdim=keepdim)


def rescale(x, x_min=-1., x_max=1.):
    return x * (x_max - x_min) + x_min


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
