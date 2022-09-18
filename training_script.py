import os
import torch
from preprocessing import training_data
import numpy as np
import icosahedron
import dihedral12 as d12
import trainingScalars as training
from dataGrab import data_grab
from torch.nn import functional as F
from torch import nn
import sys

#grab training data
N_subjects=int(sys.argv[1])
X, Xflat, S0X, Y, S0Y, mask_train, interp_matrix, interp_matrix_ind=data_grab(N_subjects,'/home/u2hussai/project/u2hussai/niceData/training/')

#initalize the network
H=5 #size if 2d grid will be h=5+1 w=5*(5+1)
Nc=16 # 16x16x16 patch size
ico=icosahedron.icomesh(m=H-1) #get the icosahedorn
I, J, T = d12.padding_basis(H=H)  # for padding

Ndirs = 6 #number of diffusion directions
Nscalars = 1 #this is to hold on to S0
Nshells = 1 #we use 1 shell
Cinp = 64 #this is the number of "effective" 3d filters
Cin = Cinp*(Nscalars + Nshells*Ndirs) #the number of actual filters needed

#3d convs
filterlist3d=[9,Cin,Cin,Cin] #3d layers, 9 = 2 (T1,T2) + 7 (S0 + 6 diffusion directions)
activationlist3d = [F.relu for i in range(0,len(filterlist3d)-1)] #3d layers activatiions

#2d gconvs
gfilterlist2d =[Cinp,Cinp,Cinp,Cinp,1] #gconv layers
gactivationlist2d = [F.relu for i in range(0,len(gfilterlist2d)-1)] #gconv layers activations
gactivationlist2d[-1]=None #turning of last layer activation

#model configuration
modelParams={'H':H,
             'shells':Nshells,
             'gfilterlist': gfilterlist2d,
             'linfilterlist': None,
             'gactivationlist': gactivationlist2d,
             'lactivationlist': None,
             'filterlist3d' : filterlist3d,
             'activationlist3d':activationlist3d,
             'loss': nn.MSELoss(),
             'bvec_dirs': Ndirs,
             'batch_size': 1,
             'lr': 1e-4,
             'factor': 0.5,
             'Nepochs': 20,
             'patience': 7,
             'Ntrain': X.shape[0],
             'Ntest': 1,
             'Nvalid': 1,
             'interp': 'inverse_distance',
             'basepath': '/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/',
             'type': 'ip-on',
             'misc':'residual5dscalar'
            }

#training class
trnr = training.trainer(modelParams,
                        Xtrain=X, Ytrain=Y-Xflat, S0Ytrain=S0Y-S0X, interp_matrix_train=interp_matrix,
                        interp_matrix_ind_train=interp_matrix_ind,mask=mask_train,
                        Nscalars=Nscalars,Ndir=Ndirs,ico=ico,
                        B=1,Nc=Nc,Ncore=100,core=ico.core_basis,
                        core_inv=ico.core_basis_inv,
                        zeros=ico.zeros,
                        I=I,J=J)
trnr.makeNetwork()
trnr.net=trnr.net.cuda()
trnr.save_modelParams() #save model parameters
trnr.train() #this will save checkpoints