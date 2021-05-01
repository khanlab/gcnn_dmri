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
from torch.nn import GroupNorm, Linear, ModuleList, BatchNorm2d
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
from gPyTorch import gNetFromList
from training import lNetFromList
import os
from sklearn import preprocessing

# #traning with full tensor see below in commnented code for individual vector traning
#
# #have to convert Y to a tensor and also back
# def dti2tensor(Y):
#     #takes in whole Y tensor including FA
#     out=np.zeros([len(Y),6])
#     triu_inds=np.triu_indices(3)
#     for p in range(0,len(Y)):
#         #the sequence is FA,L1,L2,L3,V1x,V1y,V1z,V2x,V2y,V2z,V3x,V3y,V3z
#         #eigen values
#         L1=Y[p,1]
#         L2=Y[p,2]
#         L3=Y[p,3]
#         #eigen vectors
#         V1=Y[p,4:7]
#         V2=Y[p,7:10]
#         V3=Y[p,10:]
#         #make diffusion tensor
#         P=np.asarray( [V1,V2,V3]).T
#         Pinv=np.linalg.inv(P)
#         Q=np.diag([L1,L2,L3])
#         D=np.matmul(Q,Pinv)
#         D=np.matmul(P,D)
#         out[p,:]=D[triu_inds]
#     return out
#
# def tensor2dti(Y):
#     out=np.zeros([len(Y),12])
#     D=np.zeros([3,3])
#     triu_inds=np.triu_indices(3)
#     for p in range(0,len(Y)):
#         D[triu_inds]=Y[p,:]
#         D[1,0]=D[0,1]
#         D[2,0]=D[0,2]
#         D[2,1]=D[1,2]
#         vals,vecs=np.linalg.eig(D)
#         idx=np.flip(np.argsort(vals))
#         out[p,0:3]=vals[idx]
#         out[p,3:6]=vecs[:,idx[0]]
#         out[p,6:9]=vecs[:,idx[1]]
#         out[p,9:12]=vecs[:,idx[2]]
#     return out
#
#
# def convert2cuda(X_train,Y_train):
#     X_train_p = np.copy(1-X_train)
#     #X_train_p = np.copy(X_train)
#     Y_train_p = np.copy(Y_train)
#     X_train_p[np.isinf(X_train_p)] = 0
#     X_train_p[np.isnan(X_train_p)] = 0
#     Y_train_p[np.isinf(Y_train_p)] = 0
#     Y_train_p[np.isnan(Y_train_p)] = 0
#     inputs = X_train_p
#     inputs = torch.from_numpy(inputs.astype(np.float32))
#     input = inputs.detach()
#     input = input.cuda()
#     target = Y_train_p
#     targets = torch.from_numpy(target.astype(np.float32))
#     target = targets.detach()
#     target = target.cuda()
#     return input,target
#
# def standardizeData(Xtrain,Ytrain):
#
#     Xmean=Xtrain.mean(axis=0)
#     Xstd=Xtrain.std(axis=0)
#
#     Xmean[np.isnan(Xmean)==1]=0
#     Xstd[np.isnan(Xstd)==1]=0
#
#     Ymean=Ytrain.mean(axis=0)
#     Ystd=Ytrain.std(axis=0)
#
#     Ymean[np.isnan(Ymean)==1]=0
#     Ystd[np.isnan(Ystd)==1]=0
#
#     Xtrain_out=(Xtrain - Xmean)/Xstd
#     Ytrain_out= (Ytrain - Ymean)/Ystd
#
#     Xtrain_out[np.isnan(Xtrain_out)==1]=0
#     Ytrain_out[np.isnan(Ytrain_out)==1]=0
#
#     return Xtrain_out, Xmean, Xstd, Ytrain_out , Ymean, Ystd
#
#
#
# class Net(Module):
#     def __init__(self,gB,lB1,lB2,H,gfilterlist,linfilterlist):
#         super(Net,self).__init__()
#         self.input=input
#         self.H=H
#         self.gfilterlist=gfilterlist
#         self.linfilterlist=linfilterlist
#         self.h = 5*(self.H+1)
#         self.w = self.H+1
#         self.last = self.gfilterlist[-1]
#
#         self.gB=gB
#         self.pool = opool(self.last)
#         self.mx = MaxPool2d([2,2])
#         self.lB1=lB1
#         self.lB2=lB2
#
#
#     def forward(self,x):
#         x = self.gB(x)
#         x = self.pool(x)
#         x = self.mx(x)
#         x = x.view(-1,int(self.last * self.h * self.w / 4))
#         x1 = self.lB1(x)
#         x2 = self.lB2(x)
#         x=torch.cat([x1,x2],-1)
#         return x
#
# class dtitraining:
#     def __init__(self,modelParams,
#                  datapath,dtipath,path_and_name):
#         self.modelParams=modelParams
#         self.H=modelParams['H']
#         self.h= 5 * (H + 1)
#         self.w=  H + 1
#         self.gfilterlist=modelParams[ 'gfilterlist']
#         self.linfilterlist=modelParams['linfilterlist']
#         self.gactivationlist=modelParams['gactivationlist']
#         self.lactivationlist=modelParams['lactivationlist']
#         self.loss=modelParams['loss']
#         self.batch_size=modelParams['batch_size']
#         self.lr=modelParams['lr']
#         self.factor=modelParams['factor']
#         self.Nepochs=modelParams['Nepochs']
#         self.patience=modelParams['patience']
#         self.Ntrain=modelParams['Ntrain']
#         self.Ntest=modelParams['Ntest']
#         self.Nvalid=modelParams['Nvalid']
#         self.interp=modelParams['interp']
#         self.path_and_name=path_and_name
#         self.datapath=datapath
#         self.dtipath=dtipath
#         self.Xmean=[]
#         self.Xstd=[]
#         self.Ymean=[]
#         self.Ystd=[]
#
#         self.Xtrain=[]
#         self.Ytrain=[]
#         self.Xtest=[]
#         self.Ytest=[]
#         self.Xvalid=[]
#         self.Yvalid =[]
#         self.ico=[]
#         self.diff=[]
#
#         self.input_train=[]
#         self.target_train=[]
#
#     def loadData(self):
#         self.Xtrain,self.Ytrain,self.Xtest,self.Ytest,self.Xvalid,self.Yvalid, self.ico,self.diff=ntt.load(self.datapath,self.dtipath,self.Ntrain,self.Ntest,self.Nvalid,interp=self.interp)
#         self.Ytrain=self.Ytrain[:,4:10]#dti2tensor(self.Ytrain)
#         #self.Xtrain,self.Xmean,self.Xstd,self.Ytrain,self.Ymean,self.Ystd = standardizeData(self.Xtrain,self.Ytrain)
#
#     def train(self):
#         self.Xtrain,self.Ytrain=convert2cuda(self.Xtrain,self.Ytrain)
#         gConvBlock = gNetFromList(self.H,self.gfilterlist,3,self.gactivationlist)
#         lBlock1=lNetFromList(self.linfilterlist,self.lactivationlist)
#         #linfilterlist[-1]=2
#         lBlock2=lNetFromList(self.linfilterlist,self.lactivationlist)
#         net=Net(gConvBlock,lBlock1,lBlock2,H,self.gfilterlist,self.linfilterlist)
#         net=net.cuda()
#         lossname=self.path_and_name+'loss.png'
#         netname=self.path_and_name+'net'
#         training.train(net,self.Xtrain,self.Ytrain,0,0,self.loss,self.lr,self.batch_size,self.factor,self.patience,self.Nepochs,lossname=lossname,netname=netname)
                

#load the data in DTI format

def dotLoss(output, target):

    a = output[:,0:3]
    ap = target[:,0:3]
    b= output[:,3:]
    bp= target[:,3:]


    #vz=-(a[:,0]*b[:,0] + a[:,1]*b[:,1])/a[:,2]
    #vz[a[:,2]<1e-6]=0
    #vz=vz.view([-1,1])
    #b=torch.cat([b,vz],dim=-1)
    #bp=torch.cat([bp,vz],dim=-1)
    # norma = a.norm(dim=-1).view(-1,1)
    # norma = norma.expand(norma.shape[0], 3)
    # normb = b.norm(dim=-1).view(-1,1)
    # normb = normb.expand(normb.shape[0], 2)
    # a = a / norma
    # b = b / normb
    a=F.normalize(a,dim=-1)
    b=F.normalize(b,dim=-1)
    #bp=F.normalize(bp,dim=-1)
    lossa = a*ap
    lossb = b*bp
    lossab = b*a
    #print(lossa,lossb)
    #print(lossb)
    lossa=lossa.sum(dim=-1).abs()
    lossb=lossb.sum(dim=-1).abs()
    lossab=lossab.sum(dim=-1).abs()
    eps=1e-6
    lossa[(lossa-1).abs()<eps]=1.0
    lossb[(lossb-1).abs()<eps]=1.0
    lossab[(lossab-1).abs()<eps]=1.0

    lossa = torch.arccos(lossa).mean()
    lossb = torch.arccos(lossb).mean()
    lossab =(np.pi/2- torch.arccos(lossab).mean()).abs()
    #print(lossa,lossb)
    return (0.5)*lossa + (0.5)*(0.5*lossb+0.5*lossab)


#datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
#dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"

datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"
dtipath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/dti"

H=5
h= 5 * (H + 1)
w=  H + 1
gfilterlist=[3,4,8,16,32,64,128]
gactivationlist=[F.relu for i in range(0,len(gfilterlist)-1)]
linfilterlist=[int(gfilterlist[-1] * h * w / 4),64,32,16,8,3]
lactivationlist=[F.relu for i in range(0,len(linfilterlist)-1)]
lactivationlist[-1]=None
modelParams={'H':5,
             'shells':3,
             'gfilterlist': gfilterlist,
             'linfilterlist': linfilterlist,
             'gactivationlist': gactivationlist ,
             'lactivationlist': lactivationlist,
             'loss': dotLoss,
             'batch_size': 32,
             'lr': 1e-2,
             'factor': 0.5,
             'Nepochs': 300,
             'patience': 50,
             'Ntrain': 100000,
             'Ntest': 1,
             'Nvalid': 1,
             'interp': 'linear',
             'misc':''
            }


trnr=training.trainer(modelParams,1,1)
trnr.makeNetwork()

