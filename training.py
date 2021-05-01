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
from gPyTorch import gNetFromList
import copy
import time
import nifti2traintest as ntt
import pickle

def path_from_modelParams(modelParams):
    def array2str(A):
        out=str(A[0])
        for i in range(1,len(A)):
            out=out + '-'+(str(A[i]))
        return out

    path = 'Ntrain-' + str(modelParams['Ntrain'])
    path = path + '_Nepochs-' + str(modelParams['Nepochs'])
    path = path + '_patience-' + str(modelParams['patience'])
    path = path + '_factor-' + str(modelParams['factor'])
    path = path + '_lr-' + str(modelParams['lr'])
    path = path + '_batch_size-'+ str(modelParams['batch_size'])
    path = path + '_interp-' + str(modelParams['interp'])
    path = path + '_glayers-'+ array2str(modelParams['gfilterlist'])
    path = path + '_gactivation0-' + str(modelParams['gactivationlist'][0].__str__()).split()[1]
    path = path + '_linlayers-' + array2str(modelParams['linfilterlist'])
    path = path + '_lactivation0-' + str(modelParams['lactivationlist'][0].__str__()).split()[1]
    path = path + '_' + str(modelParams['misc'])

    return path

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

class gnet(Module):
    """
    This will create the entire network
    """
    def __init__(self,linfilterlist,gconfilterlist,shells,H,lactivationlist=None,gactivationlist=None):
        super(gnet,self).__init__()
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



class trainer:
    def __init__(self,modelParams,Xtrain,Ytrain):
        """
        Class to create and train networks
        :param modelParams: A dict with all network parameters
        :param Xtrain: Cuda Xtrain data
        :param Ytrain: Cuda Ytrain data
        """
        self.modelParams=modelParams
        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        self.net=[]

    def makeNetwork(self):
        self.net = gnet(self.modelParams['linfilterlist'],self.modelParams['gfilterlist'] ,
                        self.modelParams['shells'],self.modelParams['H'],
                        self.modelParams['lactivationlist'],
                        self.modelParams['gactivationlist'])
        self.net = self.net.cuda()

    def train(self):
        outpath = path_from_modelParams(self.modelParams)
        lossname = outpath + 'loss.png'
        netname = outpath + 'net'
        criterion = self.modelParams['loss']
        optimizer = optim.Adamax(self.net.parameters(), lr=self.modelParams['lr'])  # , weight_decay=0.001)
        optimizer.zero_grad()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.modelParams['factor'],
                                      patience=self.modelParams['patience'],
                                      verbose=True)
        running_loss = 0
        train = torch.utils.data.TensorDataset(self.Xtrain, self.Ytrain)
        trainloader = DataLoader(train, batch_size=self.modelParams['batch_size'])

        epochs_list = []
        loss_list = []

        for epoch in range(0, self.modelParams['Nepochs']):
            print(epoch)
            for n, (inputs, targets) in enumerate(trainloader, 0):
                optimizer.zero_grad()

                output = self.net(inputs.cuda())
                loss = criterion(output, targets)
                loss = loss.sum()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                print(running_loss / len(trainloader))
                loss_list.append(running_loss / len(trainloader))
                epochs_list.append(epoch)
                # print(output[0])
                if np.isnan(running_loss / len(trainloader)) == 1:
                    break

            scheduler.step(running_loss)
            running_loss = 0.0
            if (epoch % 10) == 9:
                fig_err, ax_err = plt.subplots()
                ax_err.plot(epochs_list, np.log10(loss_list))
                if lossname is None:
                    lossname = 'loss.png'
                plt.savefig(lossname)
                plt.close(fig_err)
                if netname is None:
                    netname = 'net'
                torch.save(self.net.state_dict(), netname)

    def save_modelParams(self,path):
        with open(path + 'modelParams.pkl','wb') as f:
            pickle.dump(self.modelParams,f,pickle.HIGHEST_PROTOCOL)




