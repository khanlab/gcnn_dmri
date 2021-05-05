import diffusion
import icosahedron
import os
import sys
import nibabel as nib
import numpy as np
import nifti2traintest

N_train = sys.argv[1]
diffpath = sys.argv[2]
dtipath = sys.argv[3]
outpath = sys.argv[4]



X_train,Y_train, ico,diff=nifti2traintest.load(diffpath,dtipath,int(N_train))

if not os.path.exists(outpath):
    os.makedirs(outpath)

np.save(outpath+'/X_train_'+str(N_train)+'.npy',X_train)
np.save(outpath+'/Y_train_'+str(N_train)+'.npy',Y_train)



