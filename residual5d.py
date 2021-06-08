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
N_subjects = 3#sys.argv[1]
N_per_sub=100
Nc=16
sub_path = '/home/u2hussai/scratch/dtitraining/downsample/' #sys.argv[2] #path for input subjects
bdir = str(6) #sys.argv[3] #number of bvec directions
H=5 #lets keep this small for intial run
h=H+1
w=5*h


##loop through subjects
#get the subjectlist
dti_base= '/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/'#this will change after padding
subjects = os.listdir(sub_path)
if N_subjects > len(subjects):
    raise ValueError('Number of subjects requested is greater than those available')

#data arrays
X = torch.empty([N_subjects,N_per_sub,Nc,Nc,Nc,h,w])
Y = torch.empty([N_subjects,N_per_sub,Nc,Nc,Nc,h,w])

for sub in range(0,N_subjects):
    this_path = sub_path + '/' + subjects[sub] + '/' + bdir + '/'
    this_dti_path = dti_base + subjects[sub] + '/dtifit'
    this_dti_mask_path = this_path + '/nodif_brain_mask.nii.gz'
    this_subject=training_data(this_path,this_dti_path,this_dti_mask_path,H,N_per_sub)
    X[sub]= this_subject.X #X and Y are already standarized on a per subject basis
    Y[sub]= this_subject.Y 

X=X.reshape([N_subjects*N_per_sub,Nc,Nc,Nc,h,w])
Y=Y.reshape([N_subjects*N_per_sub,Nc,Nc,Nc,h,w])

print(X.shape)

############################## NETWORK ##########################
filterlist3d= [1,8,8]
activationlist3d = [F.relu for i in range(0,len(filterlist3d)-1)]

gfilterlist2d = [8,8,8,8,8,1]
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
             'Nepochs': 100,
             'patience': 10,
             'Ntrain': X.shape[0],
             'Ntest': 1,
             'Nvalid': 1,
             'interp': 'inverse_distance',
             'basepath': '/home/u2hussai/scratch/dtitraining/networks/',
             'type': 'V1',
             'misc':'residual5d'
            }



trnr = training.trainer(modelParams,X,Y-X)
trnr.makeNetwork()
trnr.save_modelParams()
trnr.train()





