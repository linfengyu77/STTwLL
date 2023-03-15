from turtle import forward
import torch
from torch import nn
from collections import OrderedDict
import numpy as np
import os
import matplotlib.pyplot as plt
from utlis import rmse

torch.set_default_dtype(torch.float64)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)
    
class Network(torch.nn.Module):
    def __init__(self, layers, netType, devices):
        super(Network, self).__init__()
        self.device = devices
        if netType == 'fnn':
            self.layers = FNN(layers)
        elif netType == 'cnn':
            self.layers = CNN(layers)

    def inverion(self, refslowess):
        for ii in range(self.npix):
            distc = torch.sqrt((self.xxc - self.xxc1[ii]) ** 2 + (self.yyc - self.yyc1[ii]) **2 )
            distc = torch.flatten(distc)
            self.sig_L[ii, :] = torch.exp(-distc/self.scaL)
        invsig_L = torch.linalg.solve(self.sig_L, torch.eye(self.npix).to(self.device))
        Tref = torch.matmul(self.tomoMtrix, (refslowess * torch.ones(self.npix, 1).to(self.device)))
        # time perturbation
        dT = self.travelTime - Tref
        G = torch.transpose(self.tomoMtrix, 0, 1) @ self.tomoMtrix + self.eta * invsig_L
        ds = torch.linalg.solve(G, torch.transpose(self.tomoMtrix, 0, 1)) @ dT
        return ds

    def forward(self, refslowess, eta, scaLen, tomoMtrix, travelTime, xxc, yyc, npix):
        self.npix = npix
        self.sig_L = torch.zeros((npix, npix)).to(self.device)
        self.xxc = xxc
        self.yyc = yyc
        self.xxc1 = torch.flatten(xxc)
        self.yyc1 = torch.flatten(yyc)
        self.eta  = eta
        self.scaL = scaLen
        self.tomoMtrix = tomoMtrix
        self.travelTime = travelTime
        self.sref = refslowess

        y = self.layers(self.sref)
        z = self.inverion(y)
        return y, z


class FNN(torch.nn.Module):
    def __init__(self, layers):
        super(FNN, self).__init__()
        # parameters
        self.depth = len(layers) - 1
        self.activation = torch.nn.ReLU
        # self.activation = torch.nn.Dropout
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation(0.2)))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)
        self.double()
    
    def forward(self, x):
        return self.layers(x)


class CNN(torch.nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(CNN, self).__init__()
        kernel_size = 3
        padding = 'same'
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.cnn = nn.Sequential(*layers)
        self.fnn = nn.Linear(2016, 1)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        y = self.cnn(x)
        y = torch.flatten(y, 1)
        y = self.fnn(y)
        return y



class Testing():
    def __init__(self, layers, netType, devices):
        super(Testing, self).__init__()
        self.device = devices
        if netType == 'fnn':
            self.nn = FNN(layers).to(devices)
        elif netType == 'cnn':
            self.nn = CNN(layers).to(devices)

    def inverion(self, refslowess):
        for ii in range(self.npix):
            distc = torch.sqrt((self.xxc - self.xxc1[ii]) ** 2 + (self.yyc - self.yyc1[ii]) **2 )
            distc = torch.flatten(distc)
            self.sig_L[ii, :] = torch.exp(-distc/self.scaL)
        invsig_L = torch.linalg.solve(self.sig_L, torch.eye(self.npix).to(self.device))
        Tref = torch.matmul(self.tomoMtrix, (refslowess * torch.ones(self.npix, 1).to(self.device)))
        # time perturbation
        dT = self.travelTime - Tref
        G = torch.transpose(self.tomoMtrix, 0, 1) @ self.tomoMtrix + self.eta * invsig_L
        ds = torch.linalg.solve(G, torch.transpose(self.tomoMtrix, 0, 1)) @ dT
        return ds
    
    def predict(self, refslowess, eta, scaLen, tomoMtrix, \
                    travelTime, xxc, yyc, npix, vb, sTrue, model_path):
        self.npix = npix
        self.sig_L = torch.zeros((npix, npix)).to(self.device)
        self.xxc = xxc
        self.yyc = yyc
        self.xxc1 = torch.flatten(xxc)
        self.yyc1 = torch.flatten(yyc)
        self.eta  = eta
        self.scaL = scaLen
        self.tomoMtrix = tomoMtrix
        self.travelTime = travelTime
        self.sRef = refslowess
        self.vb   = vb
        self.sTrue = sTrue

        # checkpoint = torch.load(os.path.join(model_path, 'model.pth'), map_location=self.device)
        # for key in list(checkpoint.keys()):
        #     if 'layers.' in key:
        #         checkpoint[key.replace('layers.', '')] = checkpoint[key]
        #         del checkpoint[key]
        # self.nn.load_state_dict(checkpoint)


        state_dict = torch.load(os.path.join(model_path, 'model.pth'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v
        self.nn.load_state_dict(new_state_dict)

        # self.nn.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        self.nn.eval()
        self.y = self.nn(self.sRef)
        self.z = self.inverion(self.y)
        return self.y
         
    def plotDinv(self, save_path):
        extent = 0, 100, 0, 100
        vb = self.vb.detach().cpu().numpy()
        sRef = self.sRef.detach().cpu().numpy()
        y = self.inverion(self.sRef)
        y = y.detach().cpu().numpy()
        y = np.reshape(y * vb + sRef, self.sTrue.shape)

        hsT = self.sTrue[48, :]
        hsP = y[48, :]
        vsT = self.sTrue[:, 40]
        vsP = y[:, 40]

        vb = np.reshape(vb, self.sTrue.shape)
        hmask = vb[48, :]
        vmask = vb[:, 40]

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(self.sTrue, extent=extent)
        ax1.set_xlabel("Range (km)")
        ax1.set_ylabel("Range (km)")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(y, extent=extent)
        plt.hlines(48, 0, 100)
        plt.vlines(40, 0, 100)
        ax2.set_xlabel("Range (km)")
        ax2.set_ylabel("Range (km)")
        ax2.set_title('RMSE:{:.4f}'.format(rmse(y, self.sTrue, vb)))
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(223)
        ax3.plot(hsT)
        ax3.plot(hsP)
        ax3.set_xlabel("Range (km)")
        ax3.set_ylabel("Slowness (s/km)")
        ax3.set_title('Horizone slice RMSE:{:.4f}'.format(rmse(hsP, hsT, hmask)))
        ax3.legend(['True','Predict'])
        ax3.margins(0)

        ax4 = fig.add_subplot(224)
        ax4.plot(vsT, range(len(vsT)))
        ax4.plot(vsP, range(len(vsP)))
        ax4.set_ylabel("Range (km)")
        ax4.set_xlabel("Slowness (s/km)")
        ax4.set_title('Vertical slice RMSE:{:.4f}'.format(rmse(vsP, vsT, vmask)))
        ax4.legend(['True','Predict'])
        ax4.margins(0)
        plt.savefig(save_path + 'conventional.png')


    def plotIinv(self, save_path):
        extent = 0, 100, 0, 100
        vb = self.vb.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy() + self.z.detach().cpu().numpy() * vb
        y = np.reshape(y,  self.sTrue.shape)

        hsT = self.sTrue[48, :]
        hsP = y[48, :]
        vsT = self.sTrue[:, 40]
        vsP = y[:, 40]

        vb = np.reshape(vb, self.sTrue.shape)
        hmask = vb[48, :]
        vmask = vb[:, 40]

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(self.sTrue, extent=extent)
        ax1.set_xlabel("Range (km)")
        ax1.set_ylabel("Range (km)")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(y, extent=extent)
        plt.hlines(48, 0, 100)
        plt.vlines(40, 0, 100)
        ax2.set_xlabel("Range (km)")
        ax2.set_ylabel("Range (km)")
        ax2.set_title('RMSE:{:.4f}'.format(rmse(y, self.sTrue, vb)))
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(223)
        ax3.plot(hsT)
        ax3.plot(hsP)
        ax3.set_xlabel("Range (km)")
        ax3.set_ylabel("Slowness (s/km)")
        ax3.set_title('Horizone slice RMSE:{:.4f}'.format(rmse(hsP, hsT, hmask)))
        ax3.legend(['True','Predict'])
        ax3.margins(0)

        ax4 = fig.add_subplot(224)
        ax4.plot(vsT, range(len(vsT)))
        ax4.plot(vsP, range(len(vsP)))
        ax4.set_ylabel("Range (km)")
        ax4.set_xlabel("Slowness (s/km)")
        ax4.set_title('Vertical slice RMSE:{:.4f}'.format(rmse(vsP, vsT, vmask)))
        ax4.legend(['True','Predict'])
        ax4.margins(0)
        plt.savefig(save_path + 'my.png')