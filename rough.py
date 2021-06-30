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
N_subjects =1#sys.argv[1]
N_per_sub=100
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
X=[]
Y=[]
mask_train= []

for sub in range(0,N_subjects):
    print(subjects[sub])
    this_path = sub_path + '/' + subjects[sub] + '/' + bdir + '/'
    this_dti_in_path = dti_base + subjects[sub] + '/' + bdir + '/dtifit'
    this_dti_path = dti_base + subjects[sub] + '/'+str(90)+'/dtifit'
    this_dti_mask_path = this_path + '/nodif_brain_mask.nii.gz'
    this_subject=training_data(this_path,this_dti_in_path, this_dti_path,this_dti_mask_path,H,N_per_sub,Nc=Nc)
    #X[sub]= this_subject.X #X and Y are already standarized on a per subject basis
    #Y[sub]= this_subject.Y
    X.append(this_subject.X)
    Y.append(this_subject.Y)
    mask_train.append(this_subject.mask_train)

X = torch.cat(X)
Y = torch.cat(Y)
mask_train = torch.cat(mask_train)

print('the shape of mask is ',mask_train.shape)

#we need a validation set, say 10 random patches
X_valid = []
Y_valid = []
mask_valid =[]
N_per_sub=10
sub=-1 #take the last one
this_path = sub_path + '/' + subjects[sub] + '/' + bdir + '/'
this_dti_path = dti_base + subjects[sub] + '/'+bdir+'/dtifit'
this_dti_mask_path = this_path + '/nodif_brain_mask.nii.gz'
this_subject=training_data(this_path,this_dti_path,this_dti_path,this_dti_mask_path,H,N_per_sub,Nc=Nc)
#X[sub]= this_subject.X #X and Y are already standarized on a per subject basis
#Y[sub]= this_subject.Y
X_valid.append(this_subject.X)
Y_valid.append(this_subject.Y)
mask_valid.append(this_subject.mask_train)

X_valid = torch.cat(X_valid)
Y_valid = torch.cat(Y_valid)
mask_valid = torch.cat(mask_valid)


############################## NETWORK ##########################
filterlist3d=[int(t) for t in sys.argv[1].split(',')] #[1,128,128]
activationlist3d = [F.relu for i in range(0,len(filterlist3d)-1)]
#activationlist3d[-1]=None

gfilterlist2d =[int(t) for t in sys.argv[2].split(',')] #[16,16,16,16,1]
gactivationlist2d = [F.relu for i in range(0,len(gfilterlist2d)-1)]
gactivationlist2d[-1]=None




modelParams={'H':H,
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
             'lr': 1e-4,
             'factor': 0.5,
             'Nepochs': 20,
             'patience': 3,
             'Ntrain': X.shape[0],
             'Ntest': 1,
             'Nvalid': 1,
             'interp': 'inverse_distance',
             'basepath': '/home/u2hussai/scratch/dtitraining/networks/',
             'type': 'V1',
             'misc':'residual5d'
            }

ico = icosahedron.icomesh(m=H-1)

shp = X.shape
X = X.view(shp[0:4] + (1,) + shp[-2:])
Y = Y.view(shp[0:4] + (1,) + shp[-2:])

shp = X_valid.shape
X_valid = X_valid.view(shp[0:4] + (1,) + shp[-2:])
Y_valid = Y_valid.view(shp[0:4] + (1,) + shp[-2:])


trnr = training.trainer(modelParams,X,Y-X,mask=mask_train,Xvalid=X_valid,Yvalid=Y_valid-X_valid,maskvalid=mask_valid,B=1,Nc=Nc,Ncore=100,core=ico.core_basis,
                        core_inv=ico.core_basis_inv,zeros=ico.zeros,I=ico.I_internal,J=ico.J_internal)
trnr.makeNetwork()
trnr.net=trnr.net.cuda()
trnr.save_modelParams()
trnr.train()




# test = trnr.Xtrain
# #test = test.view((1,)+test.shape)

# plt.imshow(test[5,5,0,10,0])
# plt.figure()
# plt.imshow(Y[5,5,0,10,0])

# this_test=test[5].view((1,)+test.shape[1:])
# out =this_test+ trnr.net(this_test.cuda())
# plt.figure()
# plt.imshow(out[0,5,5,10,0].detach().numpy())
# import diffusion
# import icosahedron
# import dihedral12 as d12
# import numpy as np
# import torch
# import preprocessing
# from torch.nn.modules.module import Module
#
# #here we want to do 3d convolutions directly on the diffusion data in the first block
# #and then map it to the flat icosahedron for 2d convlutions
# ######################## GET DATA ################################
# diffpath = './data/downsample_cut_pad/sub-100206/6/'
# Nc=16
# diff = diffusion.diffVolume(diffpath)
#
#
# #get coords
# Ntrain=200
# dtipath = './data/downsample_cut_pad/sub-100206/6/dtifit'
# maskpath = './data/downsample_cut_pad/sub-100206/6/nodif_brain_mask.nii.gz'
# preproc = preprocessing.training_data(diffpath,dtipath,maskpath,5,Ntrain)
#
# #inputs
# X = torch.from_numpy( diff.vol.get_fdata()[preproc.xp,preproc.yp,preproc.zp])
# X = X.view(Ntrain,Nc,Nc,Nc,7)
#
# #outputs
# Y = preproc.Y
#
#
# #diff2ico
# input = X[0,0,0,0,1:]
# M = preproc.diff_input.interpolation_matrices[0]
# sph2ico_basis = preproc.diff_input.sphere_to_flat_basis(preproc.ico)
# zero_out = np.isnan( sph2ico_basis)
# sph2ico_basis[zero_out]=0
# sph2ico_basis= sph2ico_basis.astype(int)
# I,J,T = d12.padding_basis(preproc.ico.H)
#
#
# class diff2ico(Module):
#     def __init__(self,M,sph2ico,zero_out,I,J,Cin,Ndirs=6):
#         super(diff2ico,self).__init__()
#         self.M = M
#         self.sph2ico = sph2ico
#         self.zero_out = zero_out
#         self.I = I
#         self.J = J
#         self.Cin = Cin
#         self.Ndirs = Ndirs
#
#     def forward(self,x):
#         #x will have shape [B,Nc,Nc,Nc,Cin*Ndirs]
#         B = x.shape[0]
#         Nc = x.shape[1]
#         Cin = self.Cin
#         Ndirs = self.Ndirs
#
#         x = x.view([B*Nc*Nc*Nc*Cin,Ndirs]).T
#         x = torch.matmul(self.M,x)
#         x = x[self.sph2ico,:]
#         x = x[self.I[0,:,:],self.J[0,:,:],:]
#         x = x[]
#
#
#
#
