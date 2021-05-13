import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt 
import matplotlib 
import os
import sys

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

#make plots of FA versus difference 

FA_cut=np.linspace(0.01,0.9,10)
diff=np.zeros([2,10,10]) #[mean/std,dirs,FA_cut]
diff_dtifit=np.zeros([2,10,10]) #[mean/std,dirs,FA_cut]

fig,ax=plt.subplots(len(FA_cut),figsize=(3,20))
for iFA,FA in enumerate(FA_cut):
    for ibdir,bdir in enumerate(bdirs):
        V1_diff=nib.load(subnetpath+'/'+str(bdir) + '/' + 'V1_difference.nii.gz')
        V1_diff_dtifit=nib.load(subnetpath + '/' + str(bdir)+'/'+'V1_difference_dtifit.nii.gz')
        
        V1_diff_c=V1_diff.get_fdata()
        V1_diff_c=V1_diff_c[V1_diff_c>FA]
        diff[0,iFA,ibdir]=V1_diff_c.mean()
        diff[1,iFA,ibdir]=V1_diff_c.std()
        
        V1_diff_c=V1_diff_dtifit.get_fdata()
        V1_diff_c=V1_diff_c[V1_diff_c>FA]
        diff_dtifit[0,iFA,ibdir]=V1_diff_c.mean()
        diff_dtifit[1,iFA,ibdir]=V1_diff_c.std()
    
    ax[iFA].set_ylim([0.3,1.05])
    ax[iFA].plot(bdirs,diff_dtifit[0,iFA,:],color='black')
    ax[iFA].plot(bdirs,diff_dtifit[0,iFA,:]+diff_dtifit[1,iFA,:],':',color='black')
    ax[iFA].plot(bdirs,diff_dtifit[0,iFA,:]-diff_dtifit[1,iFA,:],':',color='black')
    ax[iFA].plot(bdirs,diff[0,iFA,:],color='orange')
    ax[iFA].plot(bdirs,diff[0,iFA,:]+diff[1,iFA,:],':',color='orange')
    ax[iFA].plot(bdirs,diff[0,iFA,:]-diff[1,iFA,:],':',color='orange')

plt.savefig('test.png')