import torch
import gPyTorch
from gPyTorch import gConv2d
from gPyTorch import opool
from torch.nn import functional as F
from torch.nn.modules.module import Module
import dihedral12 as d12
import numpy as np
import diffusion
import icosahedron
from nibabel import load
import matplotlib.pyplot as plt
import random
from torch.nn import GroupNorm
from torch.nn import Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import dihedral12 as d12
from torch.nn import MaxPool2d
import copy
#from numpy import load
import time


######################################internal data grab###########################################3
def internal(basepath,N_train,N_test,start,end):
    # put X in array format
    def list_to_array_X(S, flats):
        # convert the lists to arrays and also renormalize the data to make attenutations brighter
        N = len(flats)
        shells = len(flats[0])
        h = len(flats[0][0])
        w = len(flats[0][0][0])
        out = np.zeros([N, shells, h, w])
        for p in range(0, N):
            #S_mean = S[p].mean()
            #if (np.isnan(S_mean) == 1) | (S_mean == 0):
            #    print('Nan! or zero')
            for s in range(0, shells):
                temp = copy.deepcopy(flats[p][s][I[0,:,:],J[0,:,:]])
                #temp= 1 - temp
                #temp[temp==1]=0
                out[p, s, :, :]=temp
                # out[p, s, :, :] = 1000/flats[p][s][I[:, :, 0], J[:, :, 0]]
                # out[p, s, :, :] = (flats[p][s][I[0, :, :], J[0, :, :]]/S_mean)
                #out[p, s, :, :] = (flats[p][s][I[0, :, :], J[0, :, :]])
                # for chart in range(0,5):
                #    top=chart*w
                #    bottom=top+w-1
                #    out[p,s,top,0]=0
                #    out[p, s, bottom, 0] = 0
                # out[out==S_mean]=0
        return out

    # put Y in array format
    def list_to_array_Y(L1, L2, L3):
        N = len(L1)
        out = np.zeros([N, 3])
        scale = 1
        for p in range(0, N):
            out[p, 0] = scale * L1[p]
            out[p, 1] = scale * L2[p]
            out[p, 2] = scale * L3[p]
        return out

    # load diffusion data
    diff = diffusion.diffVolume()
    diff.getVolume(basepath)
    diff.shells()
    diff.makeBvecMeshes()

    # get the dti data
    dti = diffusion.dti()
    dti.load(pathprefix=basepath + '/dti')

    # get the icosahedron ready
    ico = icosahedron.icomesh(m=4)
    ico.get_icomesh()
    ico.vertices_to_matrix()
    diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)

    # mask is available in diff but whatever
    mask = load(basepath + "/nodif_brain_mask.nii.gz")

    # these are all the voxels
    i, j, k = np.where(diff.mask.get_fdata() == 1)
    voxels = np.asarray([i, j, k]).T

    # pick 50000 random for training from first 500000 inds
    training_inds = random.sample(range(700000), N_train)
    test_inds = random.sample(range(500000, 700000), N_test)

    #pick straight voxels for testing
    N = N_train + N_test
    all_inds = np.arange(0, N)
    training_inds = all_inds[0:N_train]
    test_inds = all_inds[N_train:N]

    train_voxels = np.asarray([i[training_inds], j[training_inds], k[training_inds]]).T
    test_voxels = np.asarray([i[test_inds], j[test_inds], k[test_inds]]).T

    # this is for X_train and X_test
    S0_train, flat_train, signal_train = diff.makeFlat(train_voxels, ico)
    S0_test, flat_test, signal_test = diff.makeFlat(test_voxels, ico)

    I, J, T = d12.padding_basis(ico.m + 1)


    # this should be in correct format
    X_trainp = np.copy(list_to_array_X(S0_train, flat_train))
    X_testp = np.copy(list_to_array_X(S0_test, flat_test))

    X_trainp[np.isinf(X_trainp)] = 0
    X_testp[np.isinf(X_testp)] = 0
    X_trainp[np.isnan(X_trainp)] = 0
    X_testp[np.isnan(X_testp)] = 0

    # get L1,L2 and L3 (training)
    L1_train = []
    L2_train = []
    L3_train = []
    for p in train_voxels:
        L1_train.append(dti.L1.get_fdata()[tuple(p)])
        L2_train.append(dti.L2.get_fdata()[tuple(p)])
        L3_train.append(dti.L3.get_fdata()[tuple(p)])
    L1_test = []
    L2_test = []
    L3_test = []
    for p in test_voxels:
        L1_test.append(dti.L1.get_fdata()[tuple(p)])
        L2_test.append(dti.L2.get_fdata()[tuple(p)])
        L3_test.append(dti.L3.get_fdata()[tuple(p)])

    Y_trainp = list_to_array_Y(L1_train, L2_train, L3_train)
    Y_testp = list_to_array_Y(L1_test, L2_test, L3_test)

    return X_trainp,Y_trainp,X_testp,Y_testp,ico,diff

def external(N_train,N_test,start,end):

    ##Get data##
    X_train = np.load('./data/fromOldImplementation/X_train_podal.npy')
    Y_train = np.load('./data/fromOldImplementation/Y_train_podal.npy')
    X_test = np.load('./data/fromOldImplementation/X_test_podal.npy')
    Y_test = np.load('./data/fromOldImplementation/Y_test_podal.npy')


    X_train =np.moveaxis(X_train, -1, 1)
    X_test = np.moveaxis(X_test, -1, 1)

    return X_train[0:N_train],Y_train[0:N_train,start:end],X_test[0:N_test],Y_test[0:N_test,start:end]

def train(input,target,lr,H):
    #H = ico.m + 1
    h = 5 * (H + 1)
    w = H + 1
    last = 4

    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.flat = 2160
            self.conv1 = gConv2d(3, 16, H, shells=3)
            self.gn1 = GroupNorm(16, 16 * 12)
            self.conv2 = gConv2d(16, 4, H)
            self.gn2 = GroupNorm(4, 4 * 12)
            self.conv3 = gConv2d(4, 4, H)
            self.gn3 = GroupNorm(4, 4 * 12)
            self.pool = opool(last)
            self.mx = MaxPool2d([2, 2])
            self.fc1 = Linear(int(last * h * w / 4), 3)  # ,end - start-1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.gn1(x)
            x = F.relu(self.conv2(x))
            x = self.gn2(x)
            #x = self.conv3(x)
            #x = self.gn3(x)
            x = self.pool(x)
            x = self.mx(x)
            x = x.view(-1, int(last * h * w / 4))
            x = self.fc1(x)

            return x

    net = Net().cuda()

    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    optimizer = optim.Adamax(net.parameters(), lr=lr)  # , weight_decay=0.001)
    optimizer.zero_grad()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, verbose=True)

    running_loss = 0

    train = torch.utils.data.TensorDataset(input, target)
    trainloader = DataLoader(train, batch_size=16)

    for epoch in range(0, 150):
        print(epoch)
        for n, (inputs, targets) in enumerate(trainloader, 0):
            # print(n)

            optimizer.zero_grad()

            output = net(inputs.cuda())

            loss = criterion(output, targets)
            #print(loss)
            loss=loss.sum()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(running_loss / len(trainloader))
            if np.isnan(running_loss / len(trainloader))==1:
                break
        # if i%N_train==0:
        #    print('[%d, %5d] loss: %.3f' %
        #          ( 1, i + 1, running_loss / 100))
        scheduler.step(running_loss)
        running_loss = 0.0


    return net

def convert2cuda(X_train,Y_train,X_test=None,Y_test=None):
    #X_train_p = np.copy(0.1 / X_train)
    X_train_p = np.copy(X_train)
    Y_train_p = 1 * (np.copy(Y_train))
    X_train_p[np.isinf(X_train_p)] = 0

    inputs = X_train_p
    inputs = torch.from_numpy(inputs.astype(np.float32))
    input = inputs.detach()
    input = input.cuda()

    target = Y_train_p
    targets = torch.from_numpy(target.astype(np.float32))
    target = targets.detach()
    target = target.cuda()

    return input,target

N_train=1000
N_test=400
basepath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"

#internal
Xtrain_internal,Ytrain_internal,Xtest_internal,Ytest_internal,ico,diff=internal(basepath,N_train,N_test,4,7)
input_internal,target_internal=convert2cuda(Xtrain_internal,Ytrain_internal)
net_internal=train(input_internal,target_internal,1e-2,ico.m+1)

#external
Xtrain_external,Ytrain_external,Xtest_external,Ytest_enternal=external(N_train,N_test,4,7)
input_external,target_external=convert2cuda(Xtrain_external,Ytrain_external)
net_external=train(input_external,target_external,1e-2,ico.m+1)



#plotting
shell=0
for b in range(0,10):
    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(input_internal[b,shell,:,:].cpu().detach())
    ax[1].imshow(input_external[b,shell,:,:].cpu().detach())
    ax[2].imshow(input_external[b,shell, :, :].cpu().detach()/input_internal[b,shell,:,:].cpu().detach())

input_test_internal,targer_test_internal=convert2cuda(Xtest_external,Ytest_enternal)
pred_internal=net_internal(input_test_internal)

fig,ax=plt.subplots(3)
for l in range(0,3):
    ax[l].axis('equal')
    #ax[l].plot([0,8],[0,8])
    ax[l].scatter(pred_internal.cpu().detach()[:,0],targer_test_internal.cpu().detach()[:,0])




#time.sleep(5)
#plt.close()
# voxels=[[68,86,73],
#         [69,86,73],
#         [70,86,73],
#         [71,86,73],
#         [72,86,73]]
#
# voxels=[[63,86,107],
#         [63,86,106],
#         [63,86,105],
#         [63,86,104],
#         [63,86,103]]
#
#
# S0, flat,signal=diff.makeFlat(voxels,ico)
# flat = np.asarray(flat)
# #
# fig,ax=plt.subplots(1,5)
#
#
#
# for i in range(0,5):
#     ax[i].imshow(np.flip(np.log(S0[i].mean() - flat[i,1,:,:]),0))
#

#diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)

# signal=diff.makeFlat([[77,89,94]],ico)
# flat=diff.sphere_to_flat(signal[0][0],ico)
#
# voxels=[[77,88,94],
#  [77,89,93],
#  [77,89,92],
#  [77,89,91],
#  [77,89,90]]
#
# flats=[]
# for voxel in voxels:
#     signal=diff.makeFlat([voxel],ico)
#     flat=diff.sphere_to_flat(signal[0][0],ico)

















#
# Cin=8
# Cout=2
# ntheta=12
# nhex=7
# test_weight=torch.zeros([Cout,Cin,ntheta,nhex]).cuda()
# for i in range(0,Cout):
#     for j in range(0,Cin):
#         for k in range(0,ntheta):
#             for l in range(0,nhex):
#                 #test_weight[i,j,k,l]=(10000*i+1000*j+10*k+l)/10000
#                 net.conv2.weight[i, j, k, l] = (10000 * i + 1000 * j + 10 * k + l) / 10000
#
# #weights_conv1_e=d12.apply_weight_basis(net.conv1.weight, net.conv1.basis_e_h)
# #weights_conv2_e=d12.apply_weight_basis(net.conv2.weight, net.conv2.basis_e_t,net.conv2.basis_e_h)
# weights_conv2_e2=d12.apply_weight_basis(net.conv2.weight, net.conv2.basis_e_h,net.conv2.basis_e_t)
#
#
# h = 5 * (H + 1)
# w = H + 1
#
# I, J, T = np.meshgrid(np.arange(0, h), np.arange(0, w),np.arange(0, 12), indexing='ij')
# I_out, J_out, T_out = np.meshgrid(np.arange(0, h), np.arange(0, w),np.arange(0, 12), indexing='ij')
#
# I_out, J_out, T_out=d12.padding_basis(4)
#
#
#
#
#
#
#
