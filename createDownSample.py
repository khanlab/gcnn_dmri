import diffusion
import icosahedron
import os
import sys
import nibabel as nib
import numpy as np
import os

diffpath = sys.argv[1]
outpath = sys.argv[2]
subjectid = sys.argv[3]


#diffpath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
#dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"
#outpath='/home/u2hussai/scratch/'

if not os.path.exists(outpath):
    os.makedirs(outpath)
  
diff=diffusion.diffVolume()
diff.getVolume(diffpath)
diff.shells()
diff.downSample(outpath,subjectid)