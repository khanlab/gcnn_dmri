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
import sys

def convert2cuda(X_train,Y_train):
    X_train_p = np.copy(1/X_train)
    #X_train_p = np.copy(X_train)
    Y_train_p = np.copy(Y_train[:,4:7])
    X_train_p[np.isinf(X_train_p)] = 0
    X_train_p[np.isnan(X_train_p)] = 0
    Y_train_p[np.isinf(Y_train_p)] = 0
    Y_train_p[np.isnan(Y_train_p)] = 0
    inputs = X_train_p
    inputs = torch.from_numpy(inputs.astype(np.float32))
    input = inputs.detach()
    input = input.cuda()
    target = Y_train_p
    targets = torch.from_numpy(target.astype(np.float32))
    target = targets.detach()
    target = target.cuda()
    return input,target

def dotLoss(output, target):

    a = output[:,0:3]
    ap = target[:,0:3]
    a=F.normalize(a,dim=-1)
    lossa = a*ap
    lossa=lossa.sum(dim=-1).abs()
    eps=1e-6
    lossa[(lossa-1).abs()<eps]=1.0

    lossa = torch.arccos(lossa).mean()
    return lossa

#datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
#dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"

#datapath="/home/u2hussai/scratch/dtitraining/downsample/sub-124220/"
#dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-124220/dtifit"

#datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"
#dtipath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/dti"

basepath = sys.argv[1] #path where the subjects are
subjectfile= sys.argv[2] #path where the subject list file is
bvec = sys.argv[3] #how many bvectors

#get the subjectlist 
with open(subjectfile) as f:
    subjects=f.read().splitlines()

def zscore(Xtrain):
    #Xtrain = 1-Xtrain
    return Xtrain
    #Xtrain_mean = Xtrain.mean(axis=0)
    #Xtrain_std = Xtrain.std(axis=0)
    #return (Xtrain - Xtrain_mean)/Xtrain_std



max=20000
#stack all the subjects into one Xtrain, Ytrain
X_train = zscore(np.load(basepath+'/'+subjects[0]+'/'+bvec+'/'+'X_train_20000.npy')[0:max])
print(X_train.shape)
Y_train = np.load(basepath+'/'+subjects[0]+'/'+bvec+'/'+'Y_train_20000.npy')[0:max]
for s in range(1,len(subjects)):
    X_train=np.row_stack((X_train,zscore(np.load(basepath+'/'+subjects[s]+'/'+bvec+'/'+'X_train_20000.npy')[0:max])))
    Y_train=np.row_stack((Y_train,np.load(basepath+'/'+subjects[s]+'/'+bvec+'/'+'Y_train_20000.npy')[0:max]))
    
    

Ntrain=len(X_train)
# X_train, Y_train, ico, diff=ntt.load(datapath,dtipath,Ntrain)
print(X_train.shape,Y_train.shape)
X_train,Y_train=convert2cuda(X_train,Y_train)

H=5
h= 5 * (H + 1)
w=  H + 1
gfilterlist=[1,4,8,16,32,64,128]
gactivationlist=[F.relu for i in range(0,len(gfilterlist)-1)]
linfilterlist=[int(gfilterlist[-1] * h * w / 4),32,16,8,3]
lactivationlist=[F.relu for i in range(0,len(linfilterlist)-1)]
lactivationlist[-1]=None
#gactivationlist=None
modelParams={'H':5,
             'shells':1,
             'gfilterlist': gfilterlist,
             'linfilterlist': linfilterlist,
             'gactivationlist': gactivationlist ,
             'lactivationlist': lactivationlist,
             'loss': dotLoss,
             'bvec_dirs': bvec,
             'batch_size': 16,
             'lr': 1e-2,
             'factor': 0.65,
             'Nepochs': 200,
             'patience': 20,
             'Ntrain': Ntrain,
             'Ntest': 1,
             'Nvalid': 1,
             'interp': 'inverse_distance',
             'basepath': '/home/u2hussai/scratch/dtitraining/networks/',
             'type': 'V1',
             'misc':''
            }

print(X_train.shape)
print(Y_train.shape)
trnr=training.trainer(modelParams,X_train,Y_train)
trnr.makeNetwork()
trnr.save_modelParams()
trnr.train()
