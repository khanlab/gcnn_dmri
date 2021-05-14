import icosahedron
#from mayavi import mlab
import stripy
import diffusion
from joblib import Parallel, delayed
import numpy as np
import time
import gPyTorch
import torch
import extract3dDiffusion
import os
import matplotlib.pyplot as plt

datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"
dtipath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/dti"
outpath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"

#ext=extract3dDiffusion.extractor3d(datapath,dtipath,outpath)

#ext.splitNsave()

chnk=extract3dDiffusion.chunk_loader(outpath)
X,Y=chnk.load(cut=40)

X=torch.from_numpy(X)
X=X.view((X.shape[0],1)+tuple(X.shape[1:]))
X=X.cuda()
g3d=gPyTorch.gConv3d(1,1,5,shells=1)
g3d=g3d.cuda()
# A=torch.rand([3,12,9,9,9,6,30]).cuda()
out=g3d.forward(X)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flat = 2160
        self.conv1 = gConv2d(3, 8, H, shells=3)
        self.conv2 = gConv2d(8, 4, H)

        self.pool = opool(last)


        self.fc1=Linear(int(last * h * w / 4),3)#,end - start-1)


    def forward(self, x):

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.pool(x)
        x=self.mx(x)
        x = x.view(-1, int(last * h * w / 4))
        x = self.fc1(x)

        return x



# datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
# dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"
#
# diff=diffusion.diffVolume()
# diff.getVolume(datapath)
# diff.shells()
# diff.makeBvecMeshes()
#
# ico=icosahedron.icomesh()
# ico.get_icomesh()
# ico.vertices_to_matrix()
# ico.getSixDirections()
#
# i, j, k = np.where(diff.mask.get_fdata() == 1)
# voxels = np.asarray([i, j, k]).T
#
# #compute time before/after "initialization"
# start=time.time()
# diffdown=diffusion.diffDownsample(diff,ico)
# test=diffdown.downSampleFromList(voxels[0:10000])
# end=time.time()
# print(end-start)
#
# start=time.time()
# diffdown=diffusion.diffDownsample(diff,ico)
# test=diffdown.downSampleFromList([voxels[0]])
# test=diffdown.downSampleFromList(voxels[1:10000])
# end=time.time()
# print(end-start)

#downsample
#for each subject,bvec create 10000 Xtrain
#combine these and train for each bvec
#test with completely unseen subject


#xyz=ico.getSixDirections()
#
#x=[]
#y=[]
#z=[]
#for vec in ico.vertices:
#    x.append(vec[0])
#    y.append(vec[1])
#    z.append(vec[2])
#
#mlab.points3d(x,y,z)
#mlab.points3d(xyz[:,0],xyz[:,1],xyz[:,2],color=(1,0,0),scale_factor=0.23)
#mlab.show()
