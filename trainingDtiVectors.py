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

#traning with full tensor see below in commnented code for individual vector traning

#have to convert Y to a tensor and also back
def dti2tensor(Y):
    #takes in whole Y tensor including FA
    out=np.zeros([len(Y),6])
    triu_inds=np.triu_indices(3)
    for p in range(0,len(Y)):
        #the sequence is FA,L1,L2,L3,V1x,V1y,V1z,V2x,V2y,V2z,V3x,V3y,V3z
        #eigen values
        L1=Y[p,1]
        L2=Y[p,2]
        L3=Y[p,3]
        #eigen vectors
        V1=Y[p,4:7]
        V2=Y[p,7:10]
        V3=Y[p,10:]
        #make diffusion tensor
        P=np.asarray( [V1,V2,V3]).T
        Pinv=np.linalg.inv(P)
        Q=np.diag([L1,L2,L3])
        D=np.matmul(Q,Pinv)
        D=np.matmul(P,D)
        out[p,:]=D[triu_inds]
    return out

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

def convert2cuda(X_train,Y_train):
    X_train_p = np.copy(1-X_train)
    #X_train_p = np.copy(X_train)
    Y_train_p = np.copy(Y_train)
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

def standardizeData(Xtrain,Ytrain):

    Xmean=Xtrain.mean(axis=0)
    Xstd=Xtrain.std(axis=0) 
    
    Xmean[np.isnan(Xmean)==1]=0
    Xstd[np.isnan(Xstd)==1]=0

    Ymean=Ytrain.mean(axis=0)
    Ystd=Ytrain.std(axis=0)

    Ymean[np.isnan(Ymean)==1]=0
    Ystd[np.isnan(Ystd)==1]=0

    Xtrain_out=(Xtrain - Xmean)/Xstd
    Ytrain_out= (Ytrain - Ymean)/Ystd

    Xtrain_out[np.isnan(Xtrain_out)==1]=0
    Ytrain_out[np.isnan(Ytrain_out)==1]=0
    
    return Xtrain_out, Xmean, Xstd, Ytrain_out , Ymean, Ystd


    
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

class dtitraining:
    def __init__(self,modelParams,
                 datapath,dtipath,path_and_name):
        self.modelParams=modelParams
        self.H=modelParams['H']
        self.h= 5 * (H + 1)
        self.w=  H + 1
        self.gfilterlist=modelParams[ 'gfilterlist']
        self.linfilterlist=modelParams['linfilterlist']
        self.gactivationlist=modelParams['gactivationlist']
        self.lactivationlist=modelParams['lactivationlist']
        self.loss=modelParams['loss']
        self.batch_size=modelParams['batch_size']
        self.lr=modelParams['lr']
        self.factor=modelParams['factor']
        self.Nepochs=modelParams['Nepochs']
        self.patience=modelParams['patience']
        self.Ntrain=modelParams['Ntrain']
        self.Ntest=modelParams['Ntest']
        self.Nvalid=modelParams['Nvalid']
        self.interp=modelParams['interp']
        self.path_and_name=path_and_name
        self.datapath=datapath
        self.dtipath=dtipath
        self.Xmean=[]
        self.Xstd=[]
        self.Ymean=[]
        self.Ystd=[]
    
        self.Xtrain=[]
        self.Ytrain=[]
        self.Xtest=[]
        self.Ytest=[]
        self.Xvalid=[]
        self.Yvalid =[]
        self.ico=[]
        self.diff=[]

        self.input_train=[]
        self.target_train=[]

    def loadData(self):
        self.Xtrain,self.Ytrain,self.Xtest,self.Ytest,self.Xvalid,self.Yvalid, self.ico,self.diff=ntt.load(self.datapath,self.dtipath,self.Ntrain,self.Ntest,self.Nvalid,interp=self.interp)
        self.Ytrain=self.Ytrain[:,4:10]#dti2tensor(self.Ytrain)
        #self.Xtrain,self.Xmean,self.Xstd,self.Ytrain,self.Ymean,self.Ystd = standardizeData(self.Xtrain,self.Ytrain)

    def train(self):
        self.Xtrain,self.Ytrain=convert2cuda(self.Xtrain,self.Ytrain)
        gConvBlock = gNetFromList(self.H,self.gfilterlist,3,self.gactivationlist)
        lBlock1=lNetFromList(self.linfilterlist,self.lactivationlist)
        #linfilterlist[-1]=2
        lBlock2=lNetFromList(self.linfilterlist,self.lactivationlist)
        net=Net(gConvBlock,lBlock1,lBlock2,H,self.gfilterlist,self.linfilterlist)
        net=net.cuda()
        lossname=self.path_and_name+'loss.png'
        netname=self.path_and_name+'net'
        training.train(net,self.Xtrain,self.Ytrain,0,0,self.loss,self.lr,self.batch_size,self.factor,self.patience,self.Nepochs,lossname=lossname,netname=netname)
                

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

if not os.path.exists(path_and_name):
    os.makedirs(path_and_name)

dti_train=dtitraining(modelParams,datapath,dtipath,path_and_name)
dti_train.loadData()
dti_train.train()


# np.save(path_and_name+'Xmean.npy',dti_train.Xmean)
# np.save(path_and_name+'Xstd.npy',dti_train.Xstd)
# np.save(path_and_name+'Ymean.npy',dti_train.Ymean)
# np.save(path_and_name+'Ystd.npy',dti_train.Ystd)




# N_train=50000
# N_test=40
# N_valid=50
# Xtrain,Ytrain,Xtest,Ytest,Xvalid,Yvalid, ico,diff=ntt.load(datapath,dtipath,N_train,N_test,N_valid,interp='linear')

# Ytrainp=dti2tensor(Ytrain)
# input_train,target_train=convert2cuda(Xtrain,Ytrainp)

# #initialize network


# # #lgconv block
# gConvBlock = gNetFromList(H,gfilterlist,3,gactivationlist)
# lBlock1=lNetFromList(linfilterlist,lactivationlist)
# # lBlock2=lNetFromList(linfilterlist,lactivationlist)


# def diagloss(output, target):    
#     def getDiagOffDiag(a):
#         d=torch.zeros([len(a),3])
#         od=torch.zeros([len(a),3])
#         for p in range(0,len(a)):
#             d[p,0]=a[p,0]
#             d[p,1]=a[p,3]
#             d[p,2]=a[p,5]
#             od[p,0]=a[p,1]
#             od[p,1]=a[p,2]
#             od[p,2]=a[p,4]
#             #d_max=1#torch.FloatTensor(d.abs().mean(-1))
#             #od_max=1#torch.FloatTensor(od.abs().mean(-1))
#         #dp=torch.zeros_like(d)
#         #odp=torch.zeros_like(od)
#         #for i in range(0,3):
#         #    dp[:,i]=d[:,i]/d_max
#         #    odp[:,i]=od[:,i]/od_max
#         #return d/d.abs().max(-1)[0],od/od.abs().max(-1)[0]
#         return d,od

#     def convert2matrix(a): #convert to matrix from list of 6
#         D=torch.zeros([3,3])
#         D=D.cuda()
#         triu_inds=torch.triu_indices(3,3)
#         D[triu_inds[0],triu_inds[1]]=a[:]
#         D[1,0]=D[0,1]
#         D[2,0]=D[0,2]
#         D[2,1]=D[1,2]
#         return D

#     diff=torch.zeros(len(target))
#     for p in range(0,len(output)):
#         delta=output[p,:]-target[p,:]
#         delta=convert2matrix(delta)
#         delta=torch.matmul(delta,delta)
#         diff[p]=delta[0,0]+delta[1,1]+delta[2,2]
    
#     Aout,Bout=getDiagOffDiag(output)
#     Atar,Btar=getDiagOffDiag(target)
    
#     Cout=torch.column_stack([Aout,Bout,Bout])
#     Ctar=torch.column_stack([Atar,Btar,Btar])

#     mseloss=nn.MSELoss()
#     #loss1=mseloss(Aout,Atar)
#     #loss2=mseloss(Bout,Btar)
    
#     #return loss1+2*loss2
#     #return diff.mean()
#     return mseloss(Cout,Ctar)

# # #contruct the net


# net=Net(gConvBlock,lBlock1,H,gfilterlist,linfilterlist)
# net=net.cuda()
# training.train(net,input_train,target_train,0,0,diagloss,1e-2,8,1,0.5,25,150,netname='net_diagloss_heavy')






# import torch
# import gPyTorch
# from gPyTorch import gConv2d
# from gPyTorch import gNetFromList
# from gPyTorch import opool
# from torch.nn import functional as F
# from torch.nn.modules.module import Module
# import dihedral12 as d12
# import numpy as np
# import diffusion
# import icosahedron
# from nibabel import load
# import matplotlib.pyplot as plt
# import random
# from torch.nn import GroupNorm, Linear, ModuleList
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.optim as optim
# from torch import nn
# from torch.utils.data import DataLoader
# import dihedral12 as d12
# from torch.nn import MaxPool2d
# import copy
# #from numpy import load
# import time
# import nifti2traintest as ntt
# import training
# from gPyTorch import gNetFromList
# from training import lNetFromList


# #traning with full tensor see below in commnented code for individual vector traning

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
        

# def convert2cuda(X_train,Y_train):
#     X_train_p = np.copy(1-X_train)
#     #X_train_p = np.copy(X_train)
#     Y_train_p = 10000*np.copy(Y_train)
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

# #load the data in DTI format
# datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
# dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"

# N_train=50000
# N_test=40
# N_valid=50
# Xtrain,Ytrain,Xtest,Ytest,Xvalid,Yvalid, ico,diff=ntt.load(datapath,dtipath,N_train,N_test,N_valid,interp='inverse_distance')

# Ytrainp=dti2tensor(Ytrain)
# input_train,target_train=convert2cuda(Xtrain,Ytrainp)

# #initialize network
# H=5#ico.m+1
# h = 5 * (H + 1)
# w = H + 1
# gfilterlist=[3,4,8,16,32,64]
# gactivationlist=[F.relu for i in range(0,len(gfilterlist)-1)]
# last=gfilterlist[-1]
# linfilterlist=[int(last * h * w / 4),32,16,8,4,6]
# lactivationlist=[F.relu for i in range(0,len(linfilterlist)-1)]
# lactivationlist[-1]=None

# # #lgconv block
# gConvBlock = gNetFromList(H,gfilterlist,3,gactivationlist)
# lBlock1=lNetFromList(linfilterlist,lactivationlist)
# # lBlock2=lNetFromList(linfilterlist,lactivationlist)


# def diagloss(output, target):    
#     def getDiagOffDiag(a):
#         d=torch.zeros([len(a),3])
#         od=torch.zeros([len(a),3])
#         for p in range(0,len(a)):
#             d[p,0]=a[p,0]
#             d[p,1]=a[p,3]
#             d[p,2]=a[p,5]
#             od[p,0]=a[p,1]
#             od[p,1]=a[p,2]
#             od[p,2]=a[p,4]
#             #d_max=1#torch.FloatTensor(d.abs().mean(-1))
#             #od_max=1#torch.FloatTensor(od.abs().mean(-1))
#         #dp=torch.zeros_like(d)
#         #odp=torch.zeros_like(od)
#         #for i in range(0,3):
#         #    dp[:,i]=d[:,i]/d_max
#         #    odp[:,i]=od[:,i]/od_max
#         #return d/d.abs().max(-1)[0],od/od.abs().max(-1)[0]
#         return d,od

#     def convert2matrix(a): #convert to matrix from list of 6
#         D=torch.zeros([3,3])
#         D=D.cuda()
#         triu_inds=torch.triu_indices(3,3)
#         D[triu_inds[0],triu_inds[1]]=a[:]
#         D[1,0]=D[0,1]
#         D[2,0]=D[0,2]
#         D[2,1]=D[1,2]
#         return D

#     diff=torch.zeros(len(target))
#     for p in range(0,len(output)):
#         delta=output[p,:]-target[p,:]
#         delta=convert2matrix(delta)
#         delta=torch.matmul(delta,delta)
#         diff[p]=delta[0,0]+delta[1,1]+delta[2,2]
    
#     Aout,Bout=getDiagOffDiag(output)
#     Atar,Btar=getDiagOffDiag(target)
    
    
    
#     Cout=torch.column_stack([Aout,Bout,Bout])
#     Ctar=torch.column_stack([Atar,Btar,Btar])

#     mseloss=nn.MSELoss()
#     #loss1=mseloss(Aout,Atar)
#     #loss2=mseloss(Bout,Btar)
    
#     #return loss1+2*loss2
#     #return diff.mean()
#     return mseloss(Cout,Ctar)

# # #contruct the net
# class Net(Module):
#     def __init__(self,gB,lB1,H,gfilterlist,linfilterlist):
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
        
        
#     def forward(self,x):
#         x = self.gB(x)
#         x = self.pool(x)
#         x = self.mx(x)
#         x = x.view(-1,int(self.last * self.h * self.w / 4))
#         x = self.lB1(x)
#         return x

# net=Net(gConvBlock,lBlock1,H,gfilterlist,linfilterlist)
# net=net.cuda()
# training.train(net,input_train,target_train,0,0,diagloss,1e-2,8,1,0.5,25,150,netname='net_diagloss')




# def Myloss(output, target):
    
#     def normm(x):
#         norm = x.norm(dim=-1)
#         norm = norm.view(-1, 1)
#         norm = norm.expand(norm.shape[0], 3)
#         return x / norm

#     def dott(x,y):
#         x=normm(x)
#         y=normm(y)
#         return x * y


    
#     x1 = output[:,0:3]
#     y1 = target[:,0:3]
#     x2 = output[:,3:]
#     y2 = target[:,3:]

#     # print(x1.shape)
#     # print(y1.shape)
#     # print(x2.shape)
#     # print(y2.shape)    

#     loss1 =dott(x1,y1)
#     loss1=1-loss1.sum(-1).abs()
    
#     loss2 =dott(x2,y2)
#     loss2=1-loss2.sum(-1).abs()

#     loss2a =dott(x2,y1)
#     loss2a=loss2a.sum(-1).abs()
    
#     loss = (1/3)*loss1 + (1/3)*loss2+ (1/3)*loss2a

#     return torch.mean(loss)


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

# N_train=50000
# N_test=40
# N_valid=50

# print('loading data')

# datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
# dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"
# #datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"
# #dtipath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/dti"


# Xtrain,Ytrain,Xtest,Ytest,Xvalid,Yvalid, ico,diff=ntt.load(datapath,dtipath,N_train,N_test,N_valid,interp='inverse_distance')

# #FA,L1,L2,L3,V1x,V1y,V1z,V2x,V2y,V2z,V3x,V3y,V3z
# start=4
# end=10
# input_train,target_train=convert2cuda(Xtrain,Ytrain,start,end)
# input_val,target_val=convert2cuda(Xvalid,Yvalid,start,end)
# input_test,target_test=convert2cuda(Xtest,Ytest,start,end)

# #initialize network
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


# #net=training.net(linfilterlist,gfilterlist,3,H,gactivationlist=gactivationlist,lactivationlist=lactivationlist)

# net=Net(gConvBlock,lBlock1,lBlock2,H,gfilterlist,linfilterlist)
# net=net.cuda()
# training.train(net,input_train,target_train,input_val,target_val,Myloss,0.5e-2,8,1,0.5,75,350)

# #
# # def train(input,target,filterlist,lr,H,Nepochs):
# #     #H = ico.m + 1
# #     h = 5 * (H + 1)
# #     w = H + 1
# #     last = 32
# #
# #     class Net(Module):
# #         def __init__(self):
# #             super(Net, self).__init__()
# #             self.flat = 2160
# #             self.gConvs=gNetFromList(H,filterlist,3)
# #             self.pool = opool(last)
# #             self.mx = MaxPool2d([2, 2])
# #             self.fc1 = Linear(int(last * h * w / 4), 3)  # ,end - start-1)
# #             #self.fc2 = Linear(100, 3)
# #             #self.fc3 = Linear(3, 3)
# #
# #         def forward(self, x):
# #             x = self.gConvs(x)
# #             x = self.pool(x)
# #             x = self.mx(x)
# #             x = x.view(-1, int(last * h * w / 4))
# #             x = self.fc1(x)
# #             #x = self.fc2(x)
# #             #x = self.fc3(x)
# #
# #             return x
# #
# #     def Myloss(output, target):
# #         x = output
# #         y = target
# #         norm = x.norm(dim=-1)
# #         norm = norm.view(-1, 1)
# #         norm = norm.expand(norm.shape[0], 3)
# #         x = x / norm
# #         loss = x * y
# #         loss = 1 - loss.sum(dim=-1)
# #         return loss.mean().abs()
# #
# #
# #     net = Net().cuda()
# #
# #
# #     criterion = nn.MSELoss()
# #     #criterion = nn.SmoothL1Loss()
# #     criterion = Myloss
# #     optimizer = optim.Adamax(net.parameters(), lr=lr)  # , weight_decay=0.001)
# #     optimizer.zero_grad()
# #     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25, verbose=True)
# #
# #     running_loss = 0
# #
# #     train = torch.utils.data.TensorDataset(input, target)
# #     trainloader = DataLoader(train, batch_size=16)
# #
# #     for epoch in range(0, Nepochs):
# #         print(epoch)
# #         for n, (inputs, targets) in enumerate(trainloader, 0):
# #             # print(n)
# #
# #             optimizer.zero_grad()
# #
# #             output = net(inputs.cuda())
# #
# #             loss = criterion(output, targets)
# #             #print(loss)
# #             loss=loss.sum()
# #             loss.backward()
# #             optimizer.step()
# #             running_loss += loss.item()
# #         else:
# #             print(running_loss / len(trainloader))
# #             if np.isnan(running_loss / len(trainloader))==1:
# #                 break
# #         # if i%N_train==0:
# #         #    print('[%d, %5d] loss: %.3f' %
# #         #          ( 1, i + 1, running_loss / 100))
# #         scheduler.step(running_loss)
# #         running_loss = 0.0
# #
# #     return net
# #
# # N_train=10000
# # N_test=4000
# # basepath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"
# # Xtrain,Ytrain,Xtest,Ytest,Xvalid,Yvalid, ico,diff=ntt.load(basepath,N_train,2000,100)
# # input_train,target_train=convert2cuda(Xtrain,Ytrain)
# # input_test,target_test=convert2cuda(Xtest,Ytest)
# #
# # filterlist = [3, 8, 16, 24, 32]
# # net=train(input_train,target_train,filterlist,1e-6,ico.m+1,350)
# #
# test=net(input_test)
# norm = test.norm(dim=-1)
# norm = norm.view(-1, 1)
# norm = norm.expand(norm.shape[0], 3)
# # plt.figure()
# # plt.scatter(test.cpu().detach(),target_test.cpu().detach())
# # plt.axis('equal')
# # plt.savefig('test1.png')


# dots=[]
# for i in range(0,400):
#     dots.append(abs(np.dot(test[i,:].cpu().detach(),Ytest[i,:])))

# plt.figure()
# plt.hist(dots.cpu().detach())
# plt.savefig('test1.png')
