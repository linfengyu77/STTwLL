import torch
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        kernel_size = 3
        padding  = 'same'
        features = 64
        layers   = []
        channels = 1
        num_of_layers = 5
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.cnn(x)
        return x


class dictloss(nn.Module):
    def __init__(self):
        super(dictloss, self).__init__()
        self.mseloss  = nn.MSELoss()
        self.maeloss  = nn.L1Loss()
        self.sml1loss = nn.SmoothL1Loss()

    def forward(self, d, x, ss, vb, npatches, \
                    patches, npp, sRef, A, Tarr, meanY, ds, lam2, device):
        x   = torch.tensor(x).to(device)
        vb  = torch.tensor(vb).to(device)
        ss_p_sum = torch.zeros(ss.shape).to(device)
        npp  = torch.tensor(npp).to(device)
        A    = torch.tensor(A).to(device)
        sRef = torch.tensor(sRef).to(device)
        patches  = torch.tensor(patches).type(torch.long).to(device)
        Tarr = torch.tensor(Tarr).to(device) 
        npatches = torch.tensor(npatches).to(device)
        meanY = torch.tensor(meanY).to(device)
        ds   = torch.tensor(ds).to(device)
        lam2 = torch.tensor(lam2).to(device)
        ss_b = d.squeeze() @ x + meanY
        
        for k in torch.arange(0, npatches, dtype=torch.long):
            ss_p_sum[patches[:, k], 0] = ss_p_sum[patches[:, k], 0] + ss_b[:, k]
    
        ss_f = (lam2 * ds + ss_p_sum)/(lam2 + npp)
        ss   = ss_f * vb
        Tref = A @ (ss + sRef)
        return self.mseloss(Tref, Tarr)