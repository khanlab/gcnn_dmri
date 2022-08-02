import os
import torch
from preprocessing import training_data
import numpy as np
import icosahedron
import dihedral12 as d12
import trainingScalars as training

def data_grab(N_subjects,subs_path,b_dirs=6,H=5,Nc=16,N_patch=500):
    """
    This function will grab all the relevant data
    :param N_subjects: Number of subjects
    :param subs_path: Path for subjects
    """

    subjects = os.listdir(subs_path) #get subjects
    if N_subjects > len(subjects): #check if enough subjects available
        raise ValueError('Number of subjects requested is greater than those available')

    #initialize data lists, these will be contactenated as tensors later
    X=[]
    Y=[]
    S0Y=[]
    Xflat =[]
    S0X=[]
    mask_train= []
    interp_matrix = []
    interp_matrix_ind = []

    #loop through all the subjects
    for sub in range(0,N_subjects):
        print('Loading data from subject: ',subjects[sub])
        diffusion_with_bdirs_path = subs_path + '/' + subjects[sub] + '/diffusion/' + str(b_dirs)  + '/diffusion/'
        dti_with_bdirs_path       = subs_path + '/' + subjects[sub] + '/diffusion/' + str(b_dirs)  + '/dtifit/dtifit'
        mask_for_bdirs_file       = subs_path + '/' + subjects[sub] + '/diffusion/' + str(b_dirs)  + '/diffusion/nodif_brain_mask.nii.gz'
        dti_with_90dirs_path      = subs_path + '/' + subjects[sub] + '/diffusion/' + '90/'        + '/dtifit/dtifit'
        mask_for_training_file    = subs_path + '/' + subjects[sub] + '/masks/mask.nii.gz'  
        t1t2_path                 = subs_path + '/' + subjects[sub] + '/structural/'
        this_subject =  training_data(diffusion_with_bdirs_path,
                                      dti_with_bdirs_path,
                                      dti_with_90dirs_path,
                                      mask_for_bdirs_file,
                                      t1t2_path,
                                      mask_for_training_file,
                                      H, 
                                      N_train=N_patch,
                                      Nc=Nc)

        X.append(this_subject.X) #this is T1,T2, and bdirs diffusion data
        S0Y.append(this_subject.Y[0]) #dtifit S0 for labels
        Y.append(this_subject.Y[1]) #icosahedron signal from dtifit for labels
        mask_train.append(this_subject.mask_train) #training mask
        interp_matrix.append(torch.from_numpy(np.asarray(this_subject.diff_input.interpolation_matrices))) #interpolation matrices
        Xflat.append(this_subject.Xflat) #diffusion data projected on icosahedron for inputs
        S0X.append(this_subject.S0X) #Raw S0 from input diffusion data

        #this is to track with subjects interpolation matrices to use
        this_interp_matrix_inds = torch.ones([this_subject.X.shape[0]])
        this_interp_matrix_inds[this_interp_matrix_inds==1]=sub
        interp_matrix_ind.append(this_interp_matrix_inds)

    X = torch.cat(X)
    Y=torch.cat(Y)
    S0Y = torch.cat(S0Y)
    mask_train = torch.cat(mask_train)
    interp_matrix = torch.cat(interp_matrix)
    interp_matrix_ind=torch.cat(interp_matrix_ind).int()
    Xflat = torch.cat(Xflat)
    S0X = torch.cat(S0X)

    return X, Xflat, S0X, Y, S0Y, mask_train, interp_matrix, interp_matrix_ind

#grab training data
X, Xflat, S0X, Y, S0Y, mask_train, interp_matrix, interp_matrix_ind=data_grab(1,'/home/u2hussai/scratch/niceData/training/')

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
             'bvec_dirs': int(bdir),
             'batch_size': 1,
             'lr': 1e-4,
             'factor': 0.5,
             'Nepochs': 20,
             'patience': 7,
             'Ntrain': X.shape[0],
             'Ntest': 1,
             'Nvalid': 1,
             'interp': 'inverse_distance',
             'basepath': '/home/u2hussai/scratch/dtitraining/networks/',
             'type': 'V1-InternalPaddingON_reflectionfix_cornersNotZero',
             'misc':'residual5dscalar'
            }

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
trnr.save_modelParams()
trnr.train()