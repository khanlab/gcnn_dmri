import diffusion
import icosahedron
import os
import sys
import nibabel as nib
import numpy as np
import nifti2traintest
import os

diffpath = sys.argv[1]
dtipath = sys.argv[2]
outpath = sys.argv[3]

if not os.path.exists(outpath):
    os.makedirs(outpath)

S0X,X_predict=nifti2traintest.loadDownUp(diffpath,None,dtipath,0,all=1)

np.save(outpath+'/X_predict.npy',X_predict)
np.save(outpath+'/S0X_predict.npy',S0X)

#np.save(outpath+'/Y_predict.npy',Y_predict)
#np.save(outpath+'/X_predict.npy',X_predict)



