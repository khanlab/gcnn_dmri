import torch
import nibabel as nib
import sys
import os

#the HCP diffusion niftiis have size 145,174,145
#this is divisible by 29 but that is a very memory heavy patch (have to reduce filters even with two gpus)
# 144,176,144 is divisible by 16 which more manageable

#we will take a folder as input and convert all the niftiis found in there to this shape
#we will remove a plane in the x and z directions and pad with zeros in the y direction

def cuts_and_pad(nii):
    A = torch.from_numpy(nii.get_fdata())
    A = A[0:-1,:,0:-1]
    shp = A.shape
    if len(A.shape)>3:
        A = A.reshape([A.shape[0],A.shape[1],A.shape[2]*A.shape[3]])
        A = torch.nn.functional.pad(A,(0,0,1,1,0,0))
        A = A.reshape(A.shape[0:2]+shp[-2:])
    else:
        A = torch.nn.functional.pad(A,(0,0,1,1,0,0))
    return nib.Nifti1Image(A.detach().numpy(),nii.affine)

inpath = sys.argv[1]
outpath = sys.argv[2]

niis=[]

for file in os.listdir(inpath):
    if file.endswith('.nii.gz'):
        niis.append(file)

#make outdir if does not exist
if not os.path.exists(outpath):
    os.makedirs(outpath)

for nii in niis:
    print('working on ' + nii)
    out_nii = nib.load(inpath + '/'+ nii )
    out_nii = cuts_and_pad(out_nii)
    nib.save(out_nii,outpath + '/' + nii)

