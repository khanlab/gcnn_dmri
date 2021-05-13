import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

matplotlib.use('Agg')


subnetpath=sys.argv[1]
subgrndpath=sys.argv[2]


print(os.listdir(sys.argv[1]))

V1_grnd=nib.load(subgrndpath + 'dtifit_V1.nii.gz')
FA_grnd=nib.load(subgrndpath + 'dtifit_FA.nii.gz')



bdir_mean=[]
bdir_std=[]
bdirs=np.asarray(os.listdir(subnetpath))
bdirs=bdirs.astype(int)
bdirs.sort()

for bdir in bdirs:
    mask=nib.load(subnetpath +'/' + str(bdir) +'/' + 'nodif_brain_mask.nii.gz')
    V1_network=nib.load(subnetpath +'/' + str(bdir) +'/' + 'dtifit_V1.nii.gz')
    dot = np.abs((V1_network.get_fdata()*V1_grnd.get_fdata()).sum(axis=-1))
    eps=1e-6
    dot[np.abs(dot-1)<eps]=1.0
    dotnii=dot
    dotnii=1-np.rad2deg(np.arccos(dotnii))/90
    dotnii=nib.Nifti1Image(dotnii,V1_network.affine)
    nib.save(dotnii,subnetpath +'/' + str(bdir) +'/' + 'V1_difference_dtifit.nii.gz')
    #dot=dot[mask.get_fdata()>0]
    dot=dot[FA_grnd.get_fdata()>0.3]
    dot= np.rad2deg(np.arccos(dot))
    bdir_mean.append(dot.mean())
    bdir_std.append(dot.std())

bdir_mean=np.asarray(bdir_mean)
bdir_std=np.asarray(bdir_std)
print(bdir_mean)
print(bdir_std)


plt.figure()
plt.plot(bdirs,bdir_mean,color='black')
plt.plot(bdirs,bdir_mean+bdir_std,':',color='black')
plt.plot(bdirs,bdir_mean-bdir_std,':',color='black')
plt.ylabel('degrees')
plt.xlabel('directions')
plt.savefig('plot')
