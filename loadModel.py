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
from gPyTorch import gNetFromList
from training import lNetFromList
import diffusion
import nibabel as nib
from torch.nn import functional as F




class Net(Module):
    def __init__(self,gB,lB1,lB2,H,gfilterlist,linfilterlist):
        super(Net,self).__init__()
        self.input=input
        self.H=H
        self.gfilterlist=gfilterlist
        self.linfilterlist=linfilterlist
        self.h = 5*(self.H+1)
        self.w = self.H+1
        self.last = self.gfilterlist[-1]

        self.gB=gB
        self.pool = opool(self.last)
        self.mx = MaxPool2d([2,2])
        self.lB1=lB1
        self.lB2=lB2
        
        
    def forward(self,x):
        x = self.gB(x)
        x = self.pool(x)
        x = self.mx(x)
        x = x.view(-1,int(self.last * self.h * self.w / 4))
        x1 = self.lB1(x)
        x2 = self.lB2(x)
        x=torch.cat([x1,x2],-1)
        return x


#this part uses mse for dot product see commented code below
def tensor2dti(Y):
    out=np.zeros([len(Y),12])
    D=np.zeros([3,3])
    triu_inds=np.triu_indices(3)
    for p in range(0,len(Y)):
        D[triu_inds]=Y[p,:]
        D[1,0]=D[0,1]
        D[2,0]=D[0,2]
        D[2,1]=D[1,2]
        vals,vecs=np.linalg.eig(D)
        idx=np.flip(np.argsort(vals))
        out[p,0:3]=vals[idx]
        out[p,3:6]=vecs[:,idx[0]]
        out[p,6:9]=vecs[:,idx[1]]
        out[p,9:12]=vecs[:,idx[2]]
    return out

def standardizeDataFromKnown(Xtrain,Xmean,Xstd):

    Xtrain_out=(Xtrain - Xmean)/Xstd

    Xtrain_out[np.isnan(Xtrain_out)==1]=0
    
    return Xtrain_out



def unstandardizeOutputFromKnown(Ytrain,Ymean,Ystd):
    return Ytrain*Ystd +Ymean


def convert2cuda(X_train):
    X_train_p = np.copy(1-X_train)
    X_train_p[np.isinf(X_train_p)] = 0
    inputs = X_train_p
    inputs = torch.from_numpy(inputs.astype(np.float32))
    input = inputs.detach()
    input = input.cuda()
    return input    


def dotLoss(output, target):

    a = output[:,0:3]
    ap = target[:,0:3]
    b= output[:,3:]
    bp= target[:,3:]
    # norma = a.norm(dim=-1).view(-1,1)
    # norma = norma.expand(norma.shape[0], 3)
    # normb = b.norm(dim=-1).view(-1,1)
    # normb = normb.expand(normb.shape[0], 2)
    # a = a / norma
    # b = b / normb
    a=F.normalize(a,dim=-1)
    b=F.normalize(b,dim=-1)
    lossa = a*ap
    lossb = b*bp
    #print(lossa,lossb)
    #print(lossb)
    lossa=lossa.sum(dim=-1).abs()
    lossb=lossb.sum(dim=-1).abs()
    eps=1e-3
    lossa[(lossa-1).abs()<eps]=1.0
    lossb[(lossb-1).abs()<eps]=1.0
    lossa = torch.arccos(lossa).mean()
    lossb = torch.arccos(lossb).mean()
    #print(lossa,lossb)
    return (0.666666)*lossa + (0.3333333)*lossb



# #convert to flat images from whole diffusion volume (time consuming)
datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"

#load already converted volume
Xtrain=np.load('Xtrain_large.npy') #this is required only currently
#Ytrain=np.load('Ytrain_large.npy')



#load the data in DTI format
datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"

#load the data in DTI format
datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"

H=5
h= 5 * (H + 1)
w=  H + 1
gfilterlist=[3,4,8,16,32,64,128]
gactivationlist=[F.relu for i in range(0,len(gfilterlist)-1)]
linfilterlist=[int(gfilterlist[-1] * h * w / 4),64,32,16,8,3]
lactivationlist=[F.relu for i in range(0,len(linfilterlist)-1)]
lactivationlist[-1]=None


modelParams={'H':5,
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
            }


#path_and_name='/home/u2hussai/scratch/dtitraining/'+str(modelParams['gfilterlist'][-1])+'_'+str(modelParams['linfilterlist'][1])+'_'+modelParams['loss'].__str__()+'/'
path_and_name='/home/u2hussai/scratch/dtitraining/'+str(modelParams['gfilterlist'][-1])+'_'+str(modelParams['linfilterlist'][1])+'_'+'dotLoss'+'/'



gConvBlock = gNetFromList(H,gfilterlist,3,gactivationlist)
lBlock1=lNetFromList(linfilterlist,lactivationlist)
#linfilterlist[-1]=2
lBlock2=lNetFromList(linfilterlist,lactivationlist)
net=Net(gConvBlock,lBlock1,lBlock2,H,gfilterlist,linfilterlist)
net=net.cuda()
net.load_state_dict(torch.load(path_and_name+ 'net'))


#Xtrain=standardizeDataFromKnown(Xtrain,Xmean,Xstd)

inputs=convert2cuda(Xtrain)


batch_size=1000
prediction=np.zeros([len(inputs),6])

for p in range(0,len(prediction),batch_size):
    print(p)
    prediction[p:p+batch_size,:]=net(inputs[p:p+batch_size,:]).cpu().detach()

#prediction=unstandardizeOutputFromKnown(prediction,Ymean,Ystd)
#prediction=prediction

#prediction[:,0:3]=F.normalize(prediction[:,0:3],dim=-1)
#prediction[:,3:]=F.normalize( prediction[:,3:],dim=-1)


def dti_nifti(prediction_dti,datapata,dtipath):
    def normalize(v):
        v=np.asarray(v)
        norm=np.sqrt(np.sum(v*v))
        return v/norm
    
    
    def save_create_nifti(nii,affine,name):
        nii=nib.Nifti1Image(nii,affine)
        nib.save(nii,name)
    diff=diffusion.diffVolume()
    diff.getVolume(folder=datapath)
    dti= diffusion.dti()
    dti.load(dtipath)
    #get voxel list
    i, j, k = np.where(diff.mask.get_fdata() == 1)
    voxels = np.asarray([i, j, k]).T
    v1nii=np.zeros_like(dti.V1.get_fdata())
    v2nii=np.zeros_like(dti.V2.get_fdata())
    #v1nii[:,:,:,0]=1
    #v2nii[:,:,:,0]=1
    for p in range(0,len(i)):
        v1nii[i[p],j[p],k[p],:]=prediction[p,0:3]
        #vec1=prediction[p,0:3]
        #vec2=prediction[p,3:]
        #v2z=-(vec2[0]*vec1[0]+vec2[1]*vec1[1])/vec1[2]
        #if vec1[2]< 1e-8:
        #    v2z=0
        v2nii[i[p],j[p],k[p],0:3]=prediction[p,3:]
        #v2nii[i[p],j[p],k[p],2]=v2z
        v1nii[i[p],j[p],k[p],:]=normalize(v1nii[i[p],j[p],k[p],:])
        v2nii[i[p],j[p],k[p],:]=normalize(v2nii[i[p],j[p],k[p],:])
        #v3nii[i[p],j[p],k[p],:]=prediction_dti[p,9:12]
    save_create_nifti(v1nii,dti.V1.affine,'/home/u2hussai/scratch/V1_mse.nii.gz')
    save_create_nifti(v2nii,dti.V1.affine,'/home/u2hussai/scratch/V2_mse.nii.gz')
    #save_create_nifti(v3nii,dti.V1.affine,'V3_mse.nii.gz')



# def dti_nifti(prediction_dti,datapata,dtipath):
#     def save_create_nifti(nii,affine,name):
#         nii=nib.Nifti1Image(nii,affine)
#         nib.save(nii,name)
#     diff=diffusion.diffVolume()
#     diff.getVolume(folder=datapath)
#     dti= diffusion.dti()
#     dti.load(dtipath)
#     #get voxel list
#     i, j, k = np.where(diff.mask.get_fdata() == 1)
#     voxels = np.asarray([i, j, k]).T
#     L1nii=np.zeros_like(dti.L1.get_fdata())
#     L2nii=np.zeros_like(dti.L1.get_fdata())
#     L3nii=np.zeros_like(dti.L1.get_fdata())
#     v1nii=np.zeros_like(dti.V1.get_fdata())
#     v2nii=np.zeros_like(dti.V2.get_fdata())
#     v3nii=np.zeros_like(dti.V2.get_fdata())
#     for p in range(0,len(i)):
#         L1nii[i[p],j[p],k[p]]=prediction_dti[p,0]
#         L2nii[i[p],j[p],k[p]]=prediction_dti[p,1]
#         L3nii[i[p],j[p],k[p]]=prediction_dti[p,2]
#         v1nii[i[p],j[p],k[p],:]=prediction_dti[p,3:6]
#         v2nii[i[p],j[p],k[p],:]=prediction_dti[p,6:9]
#         v3nii[i[p],j[p],k[p],:]=prediction_dti[p,9:12]
#     save_create_nifti(L1nii,dti.L1.affine,'L1_mse.nii.gz')
#     save_create_nifti(L2nii,dti.L1.affine,'L2_mse.nii.gz')
#     save_create_nifti(L3nii,dti.L1.affine,'L3_mse.nii.gz')
#     save_create_nifti(v1nii,dti.V1.affine,'V1_mse.nii.gz')
#     save_create_nifti(v2nii,dti.V1.affine,'V2_mse.nii.gz')
#     save_create_nifti(v3nii,dti.V1.affine,'V3_mse.nii.gz')

dti_nifti(prediction,datapath,dtipath)
        









# #convert to flat images from whole diffusion volume (time consuming)
# datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
# dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"



# #load already converted volume
# Xtrain=np.load('Xtrain_large.npy')
# Ytrain=np.load('Ytrain_large.npy')

# def convert2cuda(X_train,Y_train,start,end):
#     X_train_p = np.copy(1-X_train)
#     #X_train_p = np.copy(X_train)
#     Y_train_p = (np.copy(Y_train[:,start:end]))
#     X_train_p[np.isinf(X_train_p)] = 0

#     inputs = X_train_p
#     inputs = torch.from_numpy(inputs.astype(np.float32))
#     input = inputs.detach()
#     input = input.cuda()

#     target = Y_train_p
#     targets = torch.from_numpy(target.astype(np.float32))
#     target = targets.detach()
#     target = target.cuda()

#     return input,target


# start=4
# end=10
# inputs,targets=convert2cuda(Xtrain,Ytrain,start,end)

# #load net work
# H=5#ico.m+1
# h = 5 * (H + 1)
# w = H + 1
# gfilterlist=[3,4,8,16,32,64]
# gactivationlist=[F.relu for i in range(0,len(gfilterlist)-1)]
# last=gfilterlist[-1]
# linfilterlist=[int(last * h * w / 4),32,16,8,4,3]
# lactivationlist=[F.relu for i in range(0,len(linfilterlist)-1)]
# lactivationlist[-1]=None

# #lgconv block
# gConvBlock = gNetFromList(H,gfilterlist,3,gactivationlist)
# lBlock1=lNetFromList(linfilterlist,lactivationlist)
# lBlock2=lNetFromList(linfilterlist,lactivationlist)

# #contruct the net
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


#         self.gB=gB
#         self.pool = opool(self.last)
#         self.mx = MaxPool2d([2,2])
#         self.lB1=lB1
#         self.lB2=lB2
        
#     def forward(self,x):
#         x = self.gB(x)
#         x = self.pool(x)
#         x = self.mx(x)
#         x = x.view(-1,int(self.last * self.h * self.w / 4))
#         x1 = self.lB1(x)
#         x2 = self.lB2(x)

#         x=torch.cat([x1,x2],-1)

#         return x


# net=Net(gConvBlock,lBlock1,lBlock2,H,gfilterlist,linfilterlist)

# net.load_state_dict(torch.load('./net'))
# net=net.cuda()



# batch_size=1000
# prediction=np.zeros_like(targets.cpu().detach())

# for p in range(0,len(prediction),batch_size):
#     print(p)
#     prediction[p:p+batch_size,:]=net(inputs[p:p+batch_size,:]).cpu().detach()

# #construct nifti
# diff=diffusion.diffVolume()
# diff.getVolume(folder=datapath)

# dti= diffusion.dti()
# dti.load(dtipath)

# def save_create_nifti(nii,affine,name):
#     nii=nib.Nifti1Image(nii,affine)
#     nib.save(nii,name)

# #get voxel list
# i, j, k = np.where(diff.mask.get_fdata() == 1)
# voxels = np.asarray([i, j, k]).T

# v1nii=np.zeros_like(dti.V1.get_fdata())
# v2nii=np.zeros_like(dti.V2.get_fdata())

# for p in range(len(i)):
#     vec1=prediction[p,0:3]
#     vec2=prediction[p,3:]
#     vec1=vec1/(np.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]+vec1[2]*vec1[2]))
#     vec2=vec2/(np.sqrt(vec2[0]*vec2[0]+vec2[1]*vec2[1]+vec2[2]*vec2[2]))
#     v1nii[i[p],j[p],k[p],:]=vec1
#     v2nii[i[p],j[p],k[p],:]=vec2

# save_create_nifti(v1nii,dti.V1.affine,'dtifit_V1_networkboth_lite.nii.gz')
# save_create_nifti(v2nii,dti.V1.affine,'dtifit_V2_networkboth_lite.nii.gz')


