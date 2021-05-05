import diffusion
import icosahedron
import os
import sys
import nibabel as nib
import numpy as np
import os

diffpath = sys.argv[1]
outpath = sys.argv[2]

#diffpath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
#dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"
#outpath='/home/u2hussai/scratch/'

if not os.path.exists(outpath):
    os.makedirs(outpath)
    


diff=diffusion.diffVolume()
diff.getVolume(diffpath)
diff.shells()
diff.makeBvecMeshes()

ico=icosahedron.icomesh()
ico.get_icomesh()
ico.vertices_to_matrix()
ico.getSixDirections()

i, j, k = np.where(diff.mask.get_fdata() == 1)
voxels = np.asarray([i, j, k]).T


diffdown=diffusion.diffDownsample(diff,ico)
s0,signal=diffdown.downSampleFromList(voxels)
s0 = np.asarray(s0)
s0 = s0.mean(axis=-1)
s0 = s0.reshape(len(s0))
signal = np.asarray(signal)
signal=signal.reshape(signal.shape[0],signal.shape[-1])


sz=[]
sz.extend(diff.vol.shape[0:3])
sz.extend([13])
diffout_nii= np.zeros(sz)
diffout_nii[i,j,k,0]=s0
diffout_nii[i,j,k,1:]=signal

diffout_nii=nib.Nifti1Image(diffout_nii,diff.vol.affine)
nib.save(diffout_nii,outpath + '/data.nii.gz')

nib.save(diff.mask,outpath+'/nodif_brain_mask.nii.gz')

fbval= open(outpath + '/bvals',"w")
fbval.write(str(0)+" ")
bval=1000
for i in range(1,diffout_nii.shape[-1]):
    print(i)
    fbval.write(str(bval) + " ")
fbval.close()

fbvec = open(outpath + '/bvecs',"w")
fbvec.write(str(0)+" ")
for i in range(1,diffout_nii.shape[-1]):
    fbvec.write(str(ico.six_direction_mesh.x[i-1])+' ')
fbvec.write("\n")

fbvec.write(str(0)+" ")
for i in range(1,diffout_nii.shape[-1]):
    fbvec.write(str(ico.six_direction_mesh.y[i-1])+' ')
fbvec.write("\n")

fbvec.write(str(0)+" ")
for i in range(1,diffout_nii.shape[-1]):
    fbvec.write(str(ico.six_direction_mesh.z[i-1])+' ')
fbvec.write("\n")
fbvec.close()

