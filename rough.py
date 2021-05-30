import icosahedron
import diffusion
import numpy as np

diff=diffusion.diffVolume('./data/6')
ico=icosahedron.icomesh(m=10)
diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)

i,j,k = np.where(diff.mask.get_fdata()>0)
voxels=np.asarray([i,j,k]).T
#diff.makeFlat(voxels[0:1000],ico)

out=diff.makeFlat(voxels,ico)

# #import icosahedron
#
# #ico=icosahedron.icomesh(m=4)
# #ico.get_icomesh()
# #ico.vertices_to_matrix()
# #ico.grid2xyz()
#
# import residualPrediction
#
# inputpath='/home/u2hussai/scratch/dtitraining/prediction/sub-518746/6/'
# netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-residual_Ntrain-100000_Nepochs-200_patience-20_factor-0.65_lr-0.01_batch_size-16_interp-inverse_distance_glayers-1-16-16-16-16-16-16-1_gactivation0-relu_residual'
# #
# redPred=residualPrediction.resPredictor(inputpath,netpath)
# #
# redPred.predict()
# redPred.makeNifti(inputpath,11)

# import nifti2traintest
# import matplotlib.pyplot as plt


# downpath='./data/6/'
# uppath='./data/90/'

# Sd,X,Sup,Y= nifti2traintest.loadDownUp(downpath,uppath,20)

# i=15
# fig,ax=plt.subplots(2)
# ax[0].imshow(X[i,0])
# ax[1].imshow(Y[i,0])

# # import icosahedron
# # #from mayavi import mlab
# # import stripy
# # import diffusion
# # from joblib import Parallel, delayed
# # import numpy as np
# # import time
# # import gPyTorch
# # import torch
# # import extract3dDiffusion
# # import os
# # import matplotlib.pyplot as plt
# # from torch.nn.modules.module import Module
# # from torch.nn import Linear
# # from torch.nn import functional as F
# # from torch.nn import ELU
# # from torch.optim.lr_scheduler import ReduceLROnPlateau
# # from torch.utils.data import DataLoader
# # import torch.optim as optim
# # from torch.nn import Conv3d
# # import dihedral12 as d12
# # import torch.nn as nn
# #
# # from gPyTorch import (gConv5dFromList,opool5d, maxpool5d, lNet5dFromList)
# #
# #
# # class Net(Module):
# #     def __init__(self):
# #         super(Net,self).__init__()
# #         self.gconv=gConv5dFromList(11,[1,4,8],shells=1,activationlist=[ELU(),ELU(),ELU(),ELU()])
# #         self.opool=opool5d(8)
# #         self.mxpool=maxpool5d([2,2])
# #         self.lin1=lNet5dFromList([int(8*12*60/4),100,90,80,50,40],activationlist=[ELU(),ELU(),ELU(),ELU(),ELU()])
# #         self.conv3d1 = Conv3d(40, 8, [3, 3, 3], padding=[1, 1, 1])
# #         self.conv3d2 = Conv3d( 8,8, [3, 3, 3], padding=[1, 1, 1])
# #         self.conv3d3 = Conv3d( 8, 4, [3, 3, 3], padding=[1, 1, 1])
# #         self.conv3d4 = Conv3d(4, 3, [3, 3, 3], padding=[1, 1, 1])
# #
# #
# #     def forward(self,x):
# #         x=self.gconv(x)
# #         x=self.opool(x)
# #         x=self.mxpool(x)
# #         x=self.lin1(x)
# #         x=self.conv3d1(x)
# #         x=self.conv3d2(x)
# #         x=self.conv3d3(x)
# #         x=self.conv3d4(x)
# #         return(x)
# #
# #
# #
# #
# #
# # # datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"
# # # dtipath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/dti"
# # # outpath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"
# #
# #
# # # datapath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # dtipath="./data/sub-100206/dtifit"
# # # outpath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# #
# # datapath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # dtipath="./data/sub-100206/dtifit"
# # outpath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# #
# #
# # ext=extract3dDiffusion.extractor3d(datapath,dtipath,outpath)
# # ext.splitNsave(9)
# #
# # # I,J,T=d12.padding_basis(11)
# # #
# # # chnk=extract3dDiffusion.chunk_loader(outpath)
# # # X,Y=chnk.load(cut=100)
# # # X=X.reshape((X.shape[0],1) + tuple(X.shape[1:]))
# # #
# # # X=X[:,:,:,:,:,I[0,:,:],J[0,:,:]]
# # #
# # #
# # def plotter(X1,X2,i):
# #     fig,ax=plt.subplots(2,1)
# #     ax[0].imshow(1/X1[i, 0, 2, 2, 2, :, :])
# #     ax[1].imshow(1/X2[i, 0, 2, 2, 2, :, :])
# # #
# # #
# # #
# # #
# # # Y=Y[:,:,:,:,4:7]
# # #
# # #
# # #
# # # inputs= np.moveaxis(X,1,-3)
# # # inputs= torch.from_numpy(inputs[103:104]).contiguous().cuda().float()
# # #
# # # targets=np.moveaxis(Y,-1,1)
# # # targets=torch.from_numpy(targets[103:104]).contiguous().cuda().float()
# # #
# # # def Myloss(output,target):
# # #     x=output
# # #     y=target
# # #     sz=output.shape
# # #     loss_all=torch.zeros([sz[0],sz[-3]*sz[-2]*sz[-1]]).cuda()
# # #     l=0
# # #     for i in range(0,output.shape[-3]):
# # #         for j in range(0, output.shape[-2]):
# # #             for k in range(0, output.shape[-1]):
# # #                 x=output[:,:,i,j,k].cuda()
# # #                 y=target[:,4:7,i,j,k].cuda()
# # #                 FA=target[:,0,i,j,k].cuda().detach()
# # #                 #FA[torch.isnan(FA)]=0
# # #                 #norm=x.norm(dim=-1)
# # #                 #norm=norm.view(-1,1)
# # #                 #norm=norm.expand(norm.shape[0],3)
# # #                 #if norm >0:
# # #                 #print(norm)
# # #                 #print(x)
# # #                 x=F.normalize(x)
# # #                 loss=x-y
# # #                 loss=loss.sum(dim=-1).abs()
# # #                 #print(loss)
# # #                 #eps = 1e-6
# # #                 #loss[(loss - 1).abs() < eps] = 1.0
# # #                 #loss_all[:,l]=torch.arccos(loss)*(1-FA)
# # #                 loss_all[:, l]=loss
# # #                 l+=1
# # #     return loss_all.flatten().mean()
# # #
# # # net=Net().cuda()
# # #
# # #
# # # criterion = nn.MSELoss()
# # # #criterion=nn.SmoothL1Loss()
# # # #criterion=nn.CosineSimilarity()
# # # #criterion=Myloss
# # # #
# # # #
# # # optimizer = optim.Adamax(net.parameters(), lr=1e-3)#, weight_decay=0.001)
# # # optimizer.zero_grad()
# # # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
# # # #
# # # running_loss = 0
# # #
# # # train = torch.utils.data.TensorDataset(inputs, targets)
# # # trainloader = DataLoader(train, batch_size=2)
# # # #
# # # for epoch in range(0, 40):
# # #     print(epoch)
# # #     for n, (inputs, target) in enumerate(trainloader, 0):
# # #         # print(n)
# # #
# # #         optimizer.zero_grad()
# # #
# # #         #print(inputs.shape)
# # #         output = net(inputs.cuda())
# # #
# # #         loss = criterion(output, target)
# # #         loss=loss.sum()
# # #         print(loss)
# # #         loss.backward()
# # #         #print(net.lin3d.weight[2,2,4,4,4,3])
# # #         #print(net.conv1.weight)
# # #         optimizer.step()
# # #         running_loss += loss.item()
# # #     else:
# # #         print(running_loss / len(trainloader))
# # #     # if i%N_train==0:
# # #     #    print('[%d, %5d] loss: %.3f' %
# # #     #          ( 1, i + 1, running_loss / 100))
# # #     scheduler.step(running_loss)
# # #     running_loss = 0.0
# # # #
# # # #
# # #
# # #
# # # #use network to make prediction and put volume back together
# # # datapath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # dtipath="./data/sub-100206/dtifit"
# # # outpath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # #
# #
# # #ext=extract3dDiffusion.extractor3d(datapath,dtipath,outpath)
# # #ext.splitNsave(9)
# #
# # # chnk=extract3dDiffusion.chunk_loader(outpath)
# # # X,Y=chnk.load(cut=0)
# # # X=1-X.reshape((X.shape[0],1) + tuple(X.shape[1:]))
# # #
# # # inputs= np.moveaxis(X,1,-3)
# # # #inputs= torch.from_numpy(inputs).contiguous().cuda()
# # #
# # # net=torch.load('net')
# # #
# # # batch_size=2
# # # outp=np.zeros([len(inputs),3,9,9,9])
# # # for i in range(0,len(inputs),batch_size):
# # #     print(i)
# # #     thisinput=torch.from_numpy(inputs[i:i+batch_size]).contiguous().cuda()
# # #     outp[i:i+batch_size,:,:,:,:]=net(thisinput).detach().cpu()
# # #     #test = net(thisinput).detach().cpu()
# # #
# # # #normalize the outs
# # # diff=[]
# # # FA=[]
# # # for i in range(0,len(outp)):
# # #     for a in range(0,9):
# # #         for b in range(0,9):
# # #             for c in range(0,9):
# # #                 if Y[i,a,b,c,0]>0.1:
# # #                     vec1=outp[i,:,a,b,c]
# # #                     vec2=Y[i,a,b,c,4:7]
# # #                     vec1=vec1/np.sqrt((vec1*vec1).sum())
# # #                     diff.append(np.rad2deg( np.arccos(np.abs( (vec1*vec2).sum()) )))
# # #                     FA.append(Y[i,a,b,c,0])
# # #
# # # diff=np.asarray(diff)
# # # FA=np.asarray(FA)
# #
# # #def zeropadder(input):
# # #     sz=input.shape
# # #     if len(sz) ==7:
# # #         out = np.zeros([sz[0], sz[1], sz[2] + 2, sz[3] + 2, sz[4] + 2] + list(sz[5:]))
# # #         out[:,:,1:-1,1:-1,1:-1,:,:]=input
# # #         return out
# # #
# # #     # if len(sz) == 5:
# # #     #     out = np.zeros([sz[0], sz[1], sz[2], sz[3]] + list(sz[4:]))
# # #     #     out[:, 1:-1, 1:-1, 1:-1, :] = input
# # #     #     return out
# # #
# # #
# # # X=zeropadder(X)
# # # #Y=zeropadder(Y)
# # #
# # # X=torch.from_numpy(X[0:2])
# # # X[np.isnan(X)==1]=0
# # # Y=torch.from_numpy(Y[0:2])
# # # #Y=Y[:,:,:,:,:,4:7]
# # # X=X.cuda()
# # # Y=Y.cuda()
# # #
# # # H=5
# # # h= 5 * (H + 1)
# # # w=H + 1
# # # last=4
# # # class Net(Module):
# # #     def __init__(self):
# # #         super(Net, self).__init__()
# # #         self.conv1 = gPyTorch.gConv3d(1, 4, H, shells=1)
# # #         self.conv2 = gPyTorch.gConv3d(4, 4, H)
# # #         #self.conv3 = gPyTorch.gConv3d(1, 1, H)
# # #         self.opool = gPyTorch.opool3d(last)
# # #         self.lin3d = gPyTorch.linear3d(last,3,9,9,9,6,30)
# # #
# # #     def forward(self, x):
# # #         x = F.relu(self.conv1(x))
# # #         x = F.relu(self.conv2(x))
# # #         #x = F.relu(self.conv3(x))
# # #         x = self.opool(x)
# # #         x = self.lin3d(x)
# # #
# # #         return x
# # #
# # # net=Net().cuda()
# # #
# # # out=net(X)
# # #
# # # def Myloss(output,target):
# # #     x=output
# # #     y=target
# # #     sz=output.shape
# # #     loss_all=torch.zeros([sz[0],sz[1]*sz[2]*sz[3]]).cuda()
# # #     l=0
# # #     for i in range(0,output.shape[1]):
# # #         for j in range(0, output.shape[2]):
# # #             for k in range(0, output.shape[3]):
# # #                 x=output[:,i,j,k,:].cuda()
# # #                 y=target[:,i,j,k,:].cuda()
# # #                 #norm=x.norm(dim=-1)
# # #                 #norm=norm.view(-1,1)
# # #                 #norm=norm.expand(norm.shape[0],3)
# # #                 #if norm >0:
# # #                 #print(norm)
# # #                 x=F.normalize(x)
# # #                 loss=x*y
# # #                 loss=loss.sum(dim=-1).abs()
# # #                 #print(loss)
# # #                 eps = 1e-6
# # #                 loss[(loss - 1).abs() < eps] = 1.0
# # #                 loss_all[:,l]=torch.arccos(loss)
# # #                 #print(output)
# # #                 l+=1
# # #     return loss_all.flatten().mean()
# # #
# #
# #
# #
# #
# # # datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
# # # dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"
# # #
# # # diff=diffusion.diffVolume()
# # # diff.getVolume(datapath)
# # # diff.shells()
# # # diff.makeBvecMeshes()
# # #
# # # ico=icosahedron.icomesh()
# # # ico.get_icomesh()
# # # ico.vertices_to_matrix()
# # # ico.getSixDirections()
# # #
# # # i, j, k = np.where(diff.mask.get_fdata() == 1)
# # # voxels = np.asarray([i, j, k]).T
# # #
# # # #compute time before/after "initialization"
# # # start=time.time()
# # # diffdown=diffusion.diffDownsample(diff,ico)
# # # test=diffdown.downSampleFromList(voxels[0:10000])
# # # end=time.time()
# # # print(end-start)
# # #
# # # start=time.time()
# # # diffdown=diffusion.diffDownsample(diff,ico)
# # # test=diffdown.downSampleFromList([voxels[0]])
# # # test=diffdown.downSampleFromList(voxels[1:10000])
# # # end=time.time()
# # # print(end-start)
# #
# # #downsample
# # #for each subject,bvec create 10000 Xtrain
# # #combine these and train for each bvec
# # #test with completely unseen subject
# #
# #
# # #xyz=ico.getSixDirections()
# # #
# # #x=[]
# # #y=[]
# # #z=[]
# # #for vec in ico.vertices:
# # #    x.append(vec[0])
# # #    y.append(vec[1])
# # #    z.append(vec[2])
# # #
# # #mlab.points3d(x,y,z)
# # #mlab.points3d(xyz[:,0],xyz[:,1],xyz[:,2],color=(1,0,0),scale_factor=0.23)
# # #mlab.show()
