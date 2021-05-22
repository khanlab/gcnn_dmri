import extract3dDiffusion
import gPyTorch
import dihedral12 as d12
import matplotlib.pylab as plt
import torch
import icosahedron
#from mayavi import mlab
import stripy
import diffusion
from joblib import Parallel, delayed
import numpy as np
import time
import gPyTorch
import torch
import extract3dDiffusion
import os
import matplotlib.pyplot as plt
from torch.nn.modules.module import Module
from torch.nn import Linear
from torch.nn import functional as F
from torch.nn import ELU
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import Conv3d
import torch.nn as nn
import os
import nifti2traintest

def zscore(Xtrain):
    #Xtrain = 1-Xtrain
    #return Xtrain
    Xtrain_mean = Xtrain.mean(axis=0)
    Xtrain_std = Xtrain.std(axis=0)
    return (Xtrain - Xtrain_mean)/Xtrain_std

def preproc(X6,S06,X12,S012):
    for i in range(0,len(X12)):
        for a in range(0,X6.shape[1]):
            for b in range(0, X6.shape[2]):
                for c in range(0, X6.shape[3]):
                    S0mean=S012[i,a,b,c].mean()
                    if S0mean==0:
                        print('zero mean encountered')
                        print(S012[i,a,b,c])
                        S0mean=1
                    X6[i,a,b,c]=X6[i,a,b,c]/S0mean
                    X12[i, a, b, c] = X12[i, a, b, c] / S0mean
    return X6,X12


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#load 6 dirs
datapath="/home/uzair/PycharmProjects/dgcnn/data/6/"
dtipath="./data/sub-100206/dtifit"
outpath="/home/uzair/PycharmProjects/dgcnn/data/6/"

chnk=extract3dDiffusion.chunk_loader(outpath)
X6t,S06,Y6=chnk.load(cut=100)

#load 12 dirs


datapath="/home/uzair/PycharmProjects/dgcnn/data/90/"
dtipath="./data/sub-100206/dtifit"
outpath="/home/uzair/PycharmProjects/dgcnn/data/90/"

chnk=extract3dDiffusion.chunk_loader(outpath)
X12t,S012,Y12=chnk.load(cut=100)

X6,X12=preproc(X6t,S06,X12t,S012)

I,J,T=d12.padding_basis(11)

X6=X6[:,:,:,:,I[0,:,:],J[0,:,:]]
X12=X12[:,:,:,:,I[0,:,:],J[0,:,:]]

targets=X12-X6

inputs=X6[0:10]
targets=targets[0:10]

inputs=inputs.reshape((inputs.shape[0],1)+tuple(inputs.shape[1:]))
targets=targets.reshape((targets.shape[0],1)+tuple(targets.shape[1:]))



inputs=torch.from_numpy(inputs).contiguous()
targets=torch.from_numpy(targets).contiguous()

inputs=inputs.float().cuda()
targets=targets.float().cuda()

inputs=inputs.view(-1,1,12,60)
targets=targets.view(-1,1,12,60)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gconv=gPyTorch.gNetFromList(11,[1,16,16,16,16,16,16,16,1],shells=1,activationlist= [F.relu,F.relu,F.relu,
                                                                                             F.relu,F.relu,F.relu,
                                                                                                 F.relu,
                                                                                       None])
        self.opool=gPyTorch.opool(1)


    def forward(self,x):
        x=self.gconv(x)
        x=self.opool(x)

        return x


net=Net().cuda()


criterion = nn.MSELoss()
#criterion=nn.SmoothL1Loss()
#criterion=nn.CosineSimilarity()
#criterion=Myloss


optimizer = optim.Adamax(net.parameters(), lr=1e-2)#, weight_decay=0.001)
optimizer.zero_grad()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)



running_loss = 0

train = torch.utils.data.TensorDataset(inputs, targets)
trainloader = DataLoader(train, batch_size=16)

train_loader_iter = iter(trainloader)
imgs, labels = next(train_loader_iter)

for epoch in range(0, 100):
    print(epoch)
    for n, (input, target) in enumerate(trainloader, 0):
        # print(n)

        optimizer.zero_grad()

        output = net(input.cuda())

        loss = criterion(output, target)
        loss=loss.sum()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(running_loss / len(trainloader))
    # if i%N_train==0:
    #    print('[%d, %5d] loss: %.3f' %
    #          ( 1, i + 1, running_loss / 100))
    scheduler.step(running_loss)
    running_loss = 0.0

inputs=X6[300:310]
targets=X12-X6
targets=targets[300:310]
upsample=X12[300:310]

inputs=inputs.reshape((inputs.shape[0],1)+tuple(inputs.shape[1:]))
targets=targets.reshape((targets.shape[0],1)+tuple(targets.shape[1:]))
upsample=upsample.reshape((upsample.shape[0],1)+tuple(upsample.shape[1:]))


inputs=torch.from_numpy(inputs).contiguous()
targets=torch.from_numpy(targets).contiguous()
upsample=torch.from_numpy(upsample).contiguous()

inputs=inputs.float().cuda()
targets=targets.float().cuda()

inputs=inputs.view(-1,1,12,60)
targets=targets.view(-1,1,12,60)
upsample=upsample.view(-1,1,12,60)


out=net(inputs[0:4])+inputs[0:4]

for i in range(0,4):
    fig,ax=plt.subplots(3)
    ax[0].imshow(inputs[i,0].detach().cpu())
    ax[1].imshow(out[i,0].detach().cpu())
    ax[2].imshow(upsample[i, 0].detach().cpu())
    #ax[3].imshow(targets[i,0].detach().cpu())


def plotter(X1,X2,i):
    fig,ax=plt.subplots(2,1)
    ax[0].imshow(X1[i, 2, 2, 2, :, :])
    ax[1].imshow(X2[i, 2, 2, 2, :, :])


