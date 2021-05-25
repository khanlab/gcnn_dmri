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
import nifti2traintest


def preproc(X,S0):
    #X will have dimensions [N,shells,h,w]
    #S0 will have dimensions [N,shells, # of b0 measurements]
    for i in range(0,X.shape[0]):
        for s in range(0,S0.shape[1]):
            Smean=S0[i,s,:].mean()
            if Smean ==0: 
                print(S0[i,s,:])
                print('zero mean encountered')
                Smean = X[i,s,:,:].max() # not sure if the this is the best appraoch 
            X[i,s,:,:]=X[i,s,:,:]/Smean
    #we need to standardize #take mean and std over all voxels (included diffusion directions)
    X = (X-X.mean())/X.std()
    #X=torch.from_numpy(X).contiguous().cuda().float()
    return X

def load_preproc(path,max): #max is how many you want to take 
    X=np.load(path+'X_train_20000.npy')[0:max]
    S0X=np.load(path+'S0X_train_20000.npy')[0:max]
    Y=np.load(path+'Y_train_20000.npy')[0:max]
    S0Y=np.load(path+'S0Y_train_20000.npy')[0:max]
    
    X=preproc(X,S0X)
    Y=preproc(Y,S0Y)

    return X,Y


basepath = sys.argv[1] #path where the subjects are
subjectfile= sys.argv[2] #path where the subject list file is
bvec = sys.argv[3] #how many bvectors

#get the subjectlist 
with open(subjectfile) as f:
    subjects=f.read().splitlines()



max=10000
#stack all the subjects into one Xtrain, S0X, Ytrain, S0Y
X_train, Y_train= load_preproc(basepath+'/'+subjects[0]+'/'+bvec+'/',max)
for s in range(1,len(subjects)):
    X_temp, Y_temp= load_preproc(basepath+'/'+subjects[s]+'/'+bvec+'/',max)
    X_train = np.row_stack((X_train,X_temp))
    Y_train = np.row_stack((Y_train,Y_temp))
    
#S0X_train, X_train,S0Y_train,Y_train = nifti2traintest.loadDownUp(sys.argv[1],sys.argv[2],sys.argv[3],5000)

#print(X_train.shape,S0X_train.shape)



#X=preproc(X_train,S0X_train)
#Y=preproc(Y_train,S0Y_train)

X=torch.from_numpy(X_train).contiguous().cuda().float()
Y=torch.from_numpy(Y_train).contiguous().cuda().float()

#X,Y=preproc(X_train,S0X_train,Y_train,S0Y_train)

print(X.shape,Y.shape)


#network stuff
H = 11
h = 5 * (H + 1)
w =  H + 1
Ntrain=len(X_train)


gfilterlist=[1,16,16,16,16,16,16,1]
gactivationlist=[F.relu for i in range(0,len(gfilterlist)-1)]
gactivationlist[-1]=None

modelParams={'H':H,
             'shells':1,
             'gfilterlist': gfilterlist,
             'linfilterlist': None,
             'gactivationlist': gactivationlist ,
             'lactivationlist': None,
             'loss': nn.MSELoss(),
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
             'type': 'residual',
             'misc':'residual'
            }

trnr=training.trainer(modelParams,X,Y-X)
trnr.makeNetwork()
trnr.save_modelParams()
trnr.train()
