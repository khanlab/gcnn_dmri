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
from torch.nn.modules.module import Module
from torch.nn import Linear
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim



datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"
dtipath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/dti"
outpath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"

#ext=extract3dDiffusion.extractor3d(datapath,dtipath,outpath)
#ext.splitNsave(9)

chnk=extract3dDiffusion.chunk_loader(outpath)
X,Y=chnk.load(cut=100)

X=1-X.reshape((X.shape[0],1) + tuple(X.shape[1:]))
#Y=Y.reshape((Y.shape[0],1) + tuple(Y.shape[1:]))
Y=Y[:,:,:,:,4:7]
def zeropadder(input):
    sz=input.shape
    if len(sz) ==7:
        out = np.zeros([sz[0], sz[1], sz[2] + 2, sz[3] + 2, sz[4] + 2] + list(sz[5:]))
        out[:,:,1:-1,1:-1,1:-1,:,:]=input
        return out

    # if len(sz) == 5:
    #     out = np.zeros([sz[0], sz[1], sz[2], sz[3]] + list(sz[4:]))
    #     out[:, 1:-1, 1:-1, 1:-1, :] = input
    #     return out


X=zeropadder(X)
#Y=zeropadder(Y)

X=torch.from_numpy(X[0:2])
X[np.isnan(X)==1]=0
Y=torch.from_numpy(Y[0:2])
#Y=Y[:,:,:,:,:,4:7]
X=X.cuda()
Y=Y.cuda()

H=5
h= 5 * (H + 1)
w=H + 1
last=4
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = gPyTorch.gConv3d(1, 4, H, shells=1)
        self.conv2 = gPyTorch.gConv3d(4, 4, H)
        #self.conv3 = gPyTorch.gConv3d(1, 1, H)
        self.opool = gPyTorch.opool3d(last)
        self.lin3d = gPyTorch.linear3d(last,3,9,9,9,6,30)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = self.opool(x)
        x = self.lin3d(x)

        return x

net=Net().cuda()

out=net(X)

def Myloss(output,target):
    x=output
    y=target
    sz=output.shape
    loss_all=torch.zeros([sz[0],sz[1]*sz[2]*sz[3]]).cuda()
    l=0
    for i in range(0,output.shape[1]):
        for j in range(0, output.shape[2]):
            for k in range(0, output.shape[3]):
                x=output[:,i,j,k,:].cuda()
                y=target[:,i,j,k,:].cuda()
                #norm=x.norm(dim=-1)
                #norm=norm.view(-1,1)
                #norm=norm.expand(norm.shape[0],3)
                #if norm >0:
                #print(norm)
                x=F.normalize(x)
                loss=x*y
                loss=loss.sum(dim=-1).abs()
                #print(loss)
                eps = 1e-6
                loss[(loss - 1).abs() < eps] = 1.0
                loss_all[:,l]=torch.arccos(loss)
                #print(output)
                l+=1
    return loss_all.flatten().mean()

#criterion = nn.MSELoss()
#criterion=nn.SmoothL1Loss()
#criterion=nn.CosineSimilarity()
criterion=Myloss


optimizer = optim.Adamax(net.parameters(), lr=1e-2)#, weight_decay=0.001)
optimizer.zero_grad()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25, verbose=True)

running_loss = 0

train = torch.utils.data.TensorDataset(X, Y)
trainloader = DataLoader(train, batch_size=1)

for epoch in range(0, 20):
    print(epoch)
    for n, (inputs, targets) in enumerate(trainloader, 0):
        # print(n)

        optimizer.zero_grad()

        #print(inputs.shape)
        output = net(inputs.cuda())

        loss = criterion(output, targets)
        loss=loss.sum()
        print(loss)
        loss.backward()
        #print(net.lin3d.weight[2,2,4,4,4,3])
        #print(net.conv1.weight)
        optimizer.step()
        running_loss += loss.item()
    else:
        print(running_loss / len(trainloader))
    # if i%N_train==0:
    #    print('[%d, %5d] loss: %.3f' %
    #          ( 1, i + 1, running_loss / 100))
    scheduler.step(running_loss)
    running_loss = 0.0






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
