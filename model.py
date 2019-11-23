import backbone
from io_utils import model_dict
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,net,num_classes,in_channels=3):
        super(Model, self).__init__()
        self.feature    = model_dict[net]()
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_classes)
        self.classifier.bias.data.fill_(0)

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores,out
                     
class Decoder(nn.Module):
    def __init__(self,feat_dim=512,out_channels=1):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(feat_dim,48*4*4)
        self.decoder = nn.Sequential(
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, out_channels, 4, stride=2, padding=1)   # [batch, 3, 32, 32]
            # ,nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0),48,4,4)
        decoded = self.decoder(x)
        # decoded = 0.5*(decoded+1)
        decoded = decoded - decoded.min()
        decoded = decoded/(decoded.max() + 1e-4)
        # decoded = (decoded>0.5).float()
        return decoded

class Decoder2(nn.Module):
    def __init__(self,feat_dim=512,out_channels=1):
        super(Decoder2, self).__init__()
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( feat_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64* 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64* 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(64* 2,    64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(    64,      out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        decoded = self.decoder(x)
        # decoded = decoded - decoded.min()
        # decoded = decoded/(decoded.max() + 1e-4)
        return decoded

class MaskModel(nn.Module):
    def __init__(self,net,num_classes,decoder=1):
        super(MaskModel, self).__init__()
        self.feature    = model_dict[net]()
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_classes,bias=False)
        if decoder ==2:
            self.decoder = Decoder2(self.feature.final_feat_dim)
        else:
            self.decoder = Decoder(self.feature.final_feat_dim)

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        mask = self.decoder(out)
        return scores, mask

    def forward2(self,x):
        out  = self.feature.forward(x)
        scores = self.classifier(out)
        mask = self.decoder(out)
        scores = self.classifier(self.feature(mask*x))
        return scores

class ContrastModel(nn.Module):
    def __init__(self,net,num_classes,decoder=1):
        super(ContrastModel, self).__init__()
        self.feature    = model_dict[net]()
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_classes,bias=False)
        self.contrast = nn.Linear(self.feature.final_feat_dim, 3,bias=False)

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        cscores = self.contrast(out)
        return scores, cscores


class ExplanationModel(nn.Module):
    def __init__(self,net,num_classes,decoder=1):
        super(ExplanationModel, self).__init__()
        self.feature    = model_dict[net]()
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_classes)
        if decoder ==2:
            self.decoder = Decoder2(self.feature.final_feat_dim,out_channels=3)
        elif decoder ==3:
            self.decoder = Decoder3(self.feature.final_feat_dim,out_channels=3)
        else:
            self.decoder = Decoder(self.feature.final_feat_dim,out_channels=3)

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        e = self.decoder(out)
        return scores, e

class VAEModel(nn.Module):
    def __init__(self,num_classes):
        super(VAEModel, self).__init__()
        self.vae   = model_dict['vae']()
        self.classifier = nn.Linear(self.vae.z_size, num_classes)

    def forward(self,x):
        (m,lvar),z, xr  = self.vae(x)
        scores  = self.classifier(z)
        return scores,(m,lvar),xr

class Decoder3(nn.Module):
    def __init__(self, feat_dim=512,ngf=64,out_channels=1):
        super(Decoder3, self).__init__()
        self.ngf = ngf
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, ngf* 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, out_channels)


    def forward(self, x):
        out_code = self.fc(x)
        out_code = out_code.view(-1, self.ngf, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = out_code - out_code.min()
        out_code = out_code/(out_code.max()+1e-4)

        return out_code

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )
    return block