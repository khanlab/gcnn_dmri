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
outpath = sys.argv[4] #this should be upto subject and then loop over each bvec downsample






Ndirs=90
cuts=np.linspace(6,Ndirs,10).astype(int)
cuts[-1]=Ndirs #incase this is different from rounding

for cut in cuts:
    thispath = outpath + '/' + str(cut) + '/' 
    print('path is ',thispath)
    X_train,Y_train, ico,diff=nifti2traintest.load(thispath,dtipath,int(N_train))

    np.save(thispath+'/X_train_'+str(N_train)+'.npy',X_train)
    np.save(thispath+'/Y_train_'+str(N_train)+'.npy',Y_train)



