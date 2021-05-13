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

X_predict,Y_predict, ico,diff=nifti2traintest.load(diffpath,dtipath,0,all=1)

np.save(outpath+'/X_predict.npy',X_predict)
np.save(outpath+'/Y_predict.npy',Y_predict)




