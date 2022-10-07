import os
import sys

import numpy as np
import torch

from .. import icosahedron, predictingScalar

# remove "net" at end
netpath_5 = "/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/bvec-dirs-6_type-ip-on_Ntrain-1498_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar"
netpath_10 = "/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/bvec-dirs-6_type-ip-on_Ntrain-3022_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar"
netpath_15 = "/home/u2hussai/projects/ctb-akhanf/u2hussai/networks/bvec-dirs-6_type-ip-on_Ntrain-4487_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar"


subs = np.array([5, 10, 15])
nsubs = int(sys.argv[1])
subs_ind = np.where(subs == nsubs)[0][0]
nets = [netpath_5, netpath_10, netpath_15]
netpath = nets[subs_ind]

print("Using network at path: ", netpath)

subjects = os.listdir("/home/u2hussai/project/u2hussai/niceData/testing/")

for i in range(0, 25):
    sub = subjects[i]
    out_dir = (
        "/home/u2hussai/projects/ctb-akhanf/u2hussai/predictions_"
        + str(nsubs)
        + "/"
        + sub
        + "/"
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("making prediction on " + out_dir)

    datapath = (
        "/home/u2hussai/project/u2hussai/niceData/testing/" + sub + "/diffusion/6/"
    )
    tpath = "/home/u2hussai/project/u2hussai/niceData/testing/" + sub + "/structural/"
    maskfile = (
        "/home/u2hussai/project/u2hussai/niceData/testing/" + sub + "/masks/mask.nii.gz"
    )
    print(datapath)
    ico = icosahedron.icomesh(m=4)

    predictor = predictingScalar.residual5dPredictorScalar(
        datapath + "diffusion/",
        datapath + "dtifit/dtifit",
        datapath + "dtifit/dtifit",
        tpath,
        maskfile,
        netpath,
        B=1,
        H=5,
        Nc=16,
        Ncore=100,
        core=ico.core_basis,
        core_inv=ico.core_basis_inv,
        zeros=ico.zeros,
        I=ico.I_internal,
        J=ico.J_internal,
    )

    if torch.cuda.is_available():
        predictor.net = predictor.net.cuda().eval()

    predictor.predict(out_dir)
