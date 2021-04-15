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
#from numpy import load
import time
import nifti2traintest as ntt
import training


def Myloss(output, target):
    x = output
    y = target
    norm = x.norm(dim=-1)
    norm = norm.view(-1, 1)
    norm = norm.expand(norm.shape[0], 3)
    x = x / norm
    loss = x * y
    loss = 1 - loss.sum(dim=-1).abs()
    return loss.mean()


def convert2cuda(X_train,Y_train,start,end):
    X_train_p = np.copy( X_train)
    #X_train_p = np.copy(X_train)
    Y_train_p = 1 * (np.copy(Y_train[:,start:end]))
    X_train_p[np.isinf(X_train_p)] = 0

    inputs = X_train_p
    inputs = torch.from_numpy(inputs.astype(np.float32))
    input = inputs.detach()
    input = input.cuda()

    target = Y_train_p
    targets = torch.from_numpy(target.astype(np.float32))
    target = targets.detach()
    target = target.cuda()

    return input,target

N_train=5000
N_test=1000
N_valid=100
basepath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"
Xtrain,Ytrain,Xtest,Ytest,Xvalid,Yvalid, ico,diff=ntt.load(basepath,N_train,N_test,N_valid)

#FA,L1,L2,L3,V1x,V1y,V1z,V2x,V2y,V2z,V3x,V3y,V3z
input_train,target_train=convert2cuda(Xtrain,Ytrain,4,7)
input_val,target_val=convert2cuda(Xvalid,Yvalid,4,7)
input_test,target_test=convert2cuda(Xtest,Ytest,4,7)

H=ico.m+1
h = 5 * (H + 1)
w = H + 1
gfilterlist=[3,32,16,8]
last=gfilterlist[-1]
linfilterlist=[int(last * h * w / 4),3,3]

net=training.net(linfilterlist,gfilterlist,3,H)

net=net.cuda()
training.train(net,input_train,target_train,input_val,target_val,Myloss,1e-2,1,0.5,25,200)
#
# def train(input,target,filterlist,lr,H,Nepochs):
#     #H = ico.m + 1
#     h = 5 * (H + 1)
#     w = H + 1
#     last = 32
#
#     class Net(Module):
#         def __init__(self):
#             super(Net, self).__init__()
#             self.flat = 2160
#             self.gConvs=gNetFromList(H,filterlist,3)
#             self.pool = opool(last)
#             self.mx = MaxPool2d([2, 2])
#             self.fc1 = Linear(int(last * h * w / 4), 3)  # ,end - start-1)
#             #self.fc2 = Linear(100, 3)
#             #self.fc3 = Linear(3, 3)
#
#         def forward(self, x):
#             x = self.gConvs(x)
#             x = self.pool(x)
#             x = self.mx(x)
#             x = x.view(-1, int(last * h * w / 4))
#             x = self.fc1(x)
#             #x = self.fc2(x)
#             #x = self.fc3(x)
#
#             return x
#
#     def Myloss(output, target):
#         x = output
#         y = target
#         norm = x.norm(dim=-1)
#         norm = norm.view(-1, 1)
#         norm = norm.expand(norm.shape[0], 3)
#         x = x / norm
#         loss = x * y
#         loss = 1 - loss.sum(dim=-1)
#         return loss.mean().abs()
#
#
#     net = Net().cuda()
#
#
#     criterion = nn.MSELoss()
#     #criterion = nn.SmoothL1Loss()
#     criterion = Myloss
#     optimizer = optim.Adamax(net.parameters(), lr=lr)  # , weight_decay=0.001)
#     optimizer.zero_grad()
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25, verbose=True)
#
#     running_loss = 0
#
#     train = torch.utils.data.TensorDataset(input, target)
#     trainloader = DataLoader(train, batch_size=16)
#
#     for epoch in range(0, Nepochs):
#         print(epoch)
#         for n, (inputs, targets) in enumerate(trainloader, 0):
#             # print(n)
#
#             optimizer.zero_grad()
#
#             output = net(inputs.cuda())
#
#             loss = criterion(output, targets)
#             #print(loss)
#             loss=loss.sum()
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         else:
#             print(running_loss / len(trainloader))
#             if np.isnan(running_loss / len(trainloader))==1:
#                 break
#         # if i%N_train==0:
#         #    print('[%d, %5d] loss: %.3f' %
#         #          ( 1, i + 1, running_loss / 100))
#         scheduler.step(running_loss)
#         running_loss = 0.0
#
#     return net
#
# N_train=10000
# N_test=4000
# basepath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"
# Xtrain,Ytrain,Xtest,Ytest,Xvalid,Yvalid, ico,diff=ntt.load(basepath,N_train,2000,100)
# input_train,target_train=convert2cuda(Xtrain,Ytrain)
# input_test,target_test=convert2cuda(Xtest,Ytest)
#
# filterlist = [3, 8, 16, 24, 32]
# net=train(input_train,target_train,filterlist,1e-6,ico.m+1,350)
#
# test=net(input_test)
# norm = test.norm(dim=-1)
# norm = norm.view(-1, 1)
# norm = norm.expand(norm.shape[0], 3)
# test=test/norm
#
# dots=[]
# for i in range(0,400):
#     dots.append(abs(np.dot(test[i,:].cpu().detach(),Ytest[i,:])))