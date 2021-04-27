import torch
import gPyTorch
from gPyTorch import gConv2d
from gPyTorch import gNetFromList
from gPyTorch import opool
from torch.nn import functional as F
from torch.nn.modules.module import Module
import dihedral12 as d12
import numpy as np
import diffusion
import icosahedron
from nibabel import load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from torch.nn import GroupNorm, Linear, ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import dihedral12 as d12
from torch.nn import MaxPool2d
import copy
import time
import nifti2traintest as ntt


class lNetFromList(Module):
    """
    This class will give us a linear network from a list of filters
    """
    def __init__(self,filterlist,activationlist=None):
        super(lNetFromList,self).__init__()
        self.activationlist=activationlist
        self.lins=[]
        if activationlist is None: #if activation list is None turn it into list of nones to avoid error below
            self.activationlist=[None for i in range(0,len(filterlist)-1)]
        for i in range(0,len(filterlist)-1):
            self.lins.append(Linear(filterlist[i],filterlist[i+1]))
        self.lins=ModuleList(self.lins)

    def forward(self,x):
        for idx,lin in enumerate(self.lins):
            activation=self.activationlist[idx]
            if activation == None:
                x=lin(x)
            else:
                x=activation(lin(x))
        return x

def get_accuracy(net,input_val,target_val):
    x=net(input_val)
    #accuracy =(1- ((pred - target_val) / target_val).abs()).abs()
    norm = x.norm(dim=-1)
    norm = norm.view(-1, 1)
    norm = norm.expand(norm.shape[0], 3)
    x = x / norm
    accuracy = x * target_val
    accuracy = torch.rad2deg( torch.arccos( accuracy.sum(dim=-1).abs()))
    return accuracy.mean().detach().cpu()

class net(Module):
    """
    This will create the entire network
    """
    def __init__(self,linfilterlist,gconfilterlist,shells,H,lactivationlist=None,gactivationlist=None):
        super(net,self).__init__()
        self.input=input
        self.linfilterlist=linfilterlist
        self.gconfilterlist=gconfilterlist
        self.lactivationlist=lactivationlist
        self.gactivationlist=gactivationlist
        self.shells=shells
        self.H=H
        self.h = 5*(self.H+1)
        self.w = self.H+1
        self.last = self.gconfilterlist[-1]

        self.gConvs=gNetFromList(self.H,self.gconfilterlist,shells,activationlist=gactivationlist)
        self.pool = opool(self.last)
        self.mx = MaxPool2d([2,2])
        self.lins=lNetFromList(linfilterlist,activationlist=lactivationlist)

    def forward(self,x):
        x = self.gConvs(x)
        x = self.pool(x)
        x = self.mx(x)
        x = x.view(-1,int(self.last * self.h * self.w / 4))
        x = self.lins(x)

        return x

def train(net,input,target,input_val,target_val, loss,lr,batch_size,factor,patience,Nepochs,lossname=None,netname=None):

    criterion = loss
    optimizer = optim.Adamax(net.parameters(), lr=lr)  # , weight_decay=0.001)
    optimizer.zero_grad()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)

    running_loss = 0

    train = torch.utils.data.TensorDataset(input, target)
    trainloader = DataLoader(train, batch_size=batch_size)


    epochs_list=[]
    loss_list=[]

    acc_list=[]
    epoch_acc_list=[]

    for epoch in range(0, Nepochs):
        print(epoch)
        for n, (inputs, targets) in enumerate(trainloader, 0):
            optimizer.zero_grad()

            output = net(inputs.cuda())
            #print(output.shape)
            #print(targets.shape)
            loss = criterion(output, targets)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(running_loss / len(trainloader))
            loss_list.append(running_loss / len(trainloader))
            epochs_list.append(epoch)
            #print(output[0])
            if np.isnan(running_loss / len(trainloader)) == 1:
                break

        scheduler.step(running_loss)
        running_loss = 0.0
        if (epoch % 10)==9:
            fig_err, ax_err = plt.subplots()
            ax_err.plot(epochs_list,np.log10(loss_list))
            if lossname is None:
                lossname='loss.png'
            plt.savefig(lossname)
            plt.close(fig_err)
            # accuracy=get_accuracy(net,input_val,target_val)
            # acc_list.append(accuracy)
            # epoch_acc_list.append(epoch)
            # fig_acc, ax_acc = plt.subplots()
            # ax_acc.plot(epoch_acc_list,acc_list)
            # plt.savefig('./accuracy2.png')
            # plt.close(fig_acc)
            if netname is None:
                netname='net'
            torch.save(net.state_dict(), netname)


    return net






