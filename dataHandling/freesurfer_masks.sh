#!/bin/bash

#sing=/project/6050199/akhanf/singularity/bids-apps/pwighton_freesurfer_7.1.0.sif

#export FC_LICENSE=export FS_LICENSE=/project/6050199/akhanf/opt/freesurfer/.license	
#singularity exec --bind /home/u2hussai/ $sing mri_binarize --i $1 --o $2/mask_all_wm.nii.gz --all-wm
mri_binarize --i $1 --o $2/mask_all_wm.nii.gz --all-wm
#singularity exec --bind /home/u2hussai/ $sing mri_binarize --i $1 --o $2/mask_gm.nii.gz --gm
mri_binarize --i $1 --o $2/mask_gm.nii.gz --gm
