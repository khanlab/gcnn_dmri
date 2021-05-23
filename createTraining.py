import diffusion
import icosahedron
import os
import sys
import nibabel as nib
import numpy as np
import nifti2traintest

N_train = sys.argv[1]
diffpath = sys.argv[2]
#dtipath = sys.argv[3]
#outpath = sys.argv[4] #this should be upto subject and then loop over each bvec downsample
#downdatapath=sys.argv[2]
#updatapath=sys.argv[3]

#this is for signal
Ndirs=90
cuts=np.linspace(6,Ndirs,10).astype(int)
cuts[-1]=Ndirs #incase this is different from rounding

for cut in cuts:
    downpath = diffpath + '/' + str(cut) + '/'
    uppath = diffpath + '/' + str(cuts[-1]) + '/'
    print(' down path is ',downpath)
    print(' up path is ', uppath)
    S0X,X,S0Y,Y=nifti2traintest.loadDownUp(downpath,uppath,int(N_train))

    np.save(downpath + '/S0X_train_' + str(N_train) + '.npy', S0X)
    np.save(downpath+'/X_train_'+str(N_train)+'.npy',X)

    np.save(downpath + '/S0Y_train_' + str(N_train) + '.npy', S0Y)
    np.save(downpath+'/Y_train_'+str(N_train)+'.npy',Y)




#this is for dti
# Ndirs=90
# cuts=np.linspace(6,Ndirs,10).astype(int)
# cuts[-1]=Ndirs #incase this is different from rounding
#
# for cut in cuts:
#     thispath = outpath + '/' + str(cut) + '/'
#     print('path is ',thispath)
#     X_train,Y_train, ico,diff=nifti2traintest.load(thispath,dtipath,int(N_train))
#
#     np.save(thispath+'/X_train_'+str(N_train)+'.npy',X_train)
#     np.save(thispath+'/Y_train_'+str(N_train)+'.npy',Y_train)



