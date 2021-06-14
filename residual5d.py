import torch
from gPyTorch import opool
from torch.nn.modules.module import Module
import numpy as np
from torch.nn import functional as F
from gPyTorch import opool
from torch.nn import ModuleList
from torch.nn import Conv3d
from torch import nn

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d
from gPyTorch import gNetFromList
import pickle
#c
import preprocessing
import nibabel as nib
import torch
import numpy as np
import diffusion
import icosahedron
import dihedral12 as d12
from torch.nn import InstanceNorm3d
import training


from preprocessing import training_data
import os
import sys

################### HELPER FUNCTIONS ################


########################## DATA ####################
##we need to get data from various subjects and stack it 

##params for data grab
N_subjects =5#sys.argv[1]
N_per_sub=200
Nc=16
sub_path = '/home/u2hussai/scratch/dtitraining/downsample_cut_pad/' #sys.argv[2] #path for input subjects
bdir = str(6) #sys.argv[3] #number of bvec directions
H=5 #lets keep this small for intial run
h=H+1
w=5*h


##loop through subjects
#get the subjectlist
dti_base= sub_path  #'/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/'#this will change after padding
subjects = os.listdir(sub_path)
if N_subjects > len(subjects):
    raise ValueError('Number of subjects requested is greater than those available')

#data arrays
#X = torch.empty([N_subjects,N_per_sub,Nc,Nc,Nc,h,w])
#Y = torch.empty([N_subjects,N_per_sub,Nc,Nc,Nc,h,w])

X=[]
Y=[]

for sub in range(0,N_subjects):
    print(subjects[sub])
    this_path = sub_path + '/' + subjects[sub] + '/' + bdir + '/'
    this_dti_path = dti_base + subjects[sub] + '/'+bdir+'/dtifit'
    this_dti_mask_path = this_path + '/nodif_brain_mask.nii.gz'
    this_subject=training_data(this_path,this_dti_path,this_dti_mask_path,H,N_per_sub)
    #X[sub]= this_subject.X #X and Y are already standarized on a per subject basis
    #Y[sub]= this_subject.Y 
    X.append(this_subject.X)
    Y.append(this_subject.Y)

X = torch.cat(X)
Y = torch.cat(Y)

#this could have changed due to available voxels
# N_subjects = X.shape[0]
# N_per_sub = Y.shape[1]

# X=X.reshape([N_subjects*N_per_sub,Nc,Nc,Nc,h,w])
# Y=Y.reshape([N_subjects*N_per_sub,Nc,Nc,Nc,h,w])

print(X.shape)

############################## NETWORK ##########################
filterlist3d= [1,16,16,16,16,16]
activationlist3d = [F.relu for i in range(0,len(filterlist3d)-1)]
#activationlist3d[-1]=None

gfilterlist2d = [16,16,16,16,1]
gactivationlist2d = [F.relu for i in range(0,len(gfilterlist2d)-1)]
gactivationlist2d[-1]=None

modelParams={'H':5,
             'shells':filterlist3d[-1],
             'gfilterlist': gfilterlist2d,
             'linfilterlist': None,
             'gactivationlist': gactivationlist2d,
             'lactivationlist': None,
             'filterlist3d' : filterlist3d, 
             'activationlist3d':activationlist3d,
             'loss': nn.MSELoss(),
             'bvec_dirs': int(bdir),
             'batch_size': 1,
             'lr': 1e-2,
             'factor': 0.5,
             'Nepochs': 50,
             'patience': 10,
             'Ntrain': X.shape[0],
             'Ntest': 1,
             'Nvalid': 1,
             'interp': 'inverse_distance',
             'basepath': '/home/u2hussai/scratch/dtitraining/networks/',
             'type': 'V1',
             'misc':'residual5d'
            }



trnr = training.trainer(modelParams,X,Y-X,multigpu=False)
trnr.makeNetwork()
trnr.net=trnr.net.cuda()
trnr.save_modelParams()
trnr.train()





