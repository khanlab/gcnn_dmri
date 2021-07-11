from numpy.random import shuffle
import gPyTorch
from torch.nn.modules.module import Module
import torch.optim as optim
from torch import nn
import os
from preprocessing import training_data
import numpy as np
from torch.utils.data import DataLoader
import torch
from dataLoader import dataLoader
from torch.nn import functional as F


##params for data grab
N_subjects =1#sys.argv[1]
N_per_sub=500
Nc=3
sub_path = '/home/u2hussai/scratch/dtitraining/downsample_cut_pad/' #sys.argv[2] #path for input subjects
#sub_path = './data/downsample_cut_pad/' #sys.argv[2] #path for input subjects
bdir = str(6) #sys.argv[3] #number of bvec directions
H=8 #lets keep this small for intial run
h=H+1
w=5*h

dl = dataLoader(sub_path,H,3,1,None,6)


class Net(Module):
    def __init__(self):
        super(Net,self).__init__()
        #params
        gfilterlist = [1,128,128,128,128,128,128,1]
        gactivationlist = [F.relu for i in range(0,len(gfilterlist)-1)]
        gactivationlist[-1]=None
        self.gconvs = gPyTorch.gNetFromList(H,gfilterlist,1,gactivationlist)
        self.pool = gPyTorch.opool(1)
    def forward(self,x):
        x=self.gconvs(x)
        x=self.pool(x)
        return x

net = Net().cuda()

# x = torch.rand(1,3,6,30).cuda()
# y = torch.rand(1,3,6,30).cuda()

optimizer = optim.Adamax(net.parameters(), lr=0.0001)  # , weight_decay=0.001)
criterion = nn.L1Loss()

h=H+1
w=5*h
X=dl.Xflat.view(-1,1,h,w)
Y=dl.Y.view(-1,1,h,w)
train = torch.utils.data.TensorDataset(X,Y-X)
trainloader = DataLoader(train, 64)

X=dl.Xflatv.view(-1,1,h,w)
Y=dl.Yv.view(-1,1,h,w)
valid = torch.utils.data.TensorDataset(X,Y-X)
validloader = DataLoader(train, 8, shuffle=True)
iterloader = iter(validloader)

running_loss = 0
     
for e in range(0,50):
    print('Epoch:',e)
    for n,(x,y) in enumerate(trainloader):
        optimizer.zero_grad()
        out=net(x.cuda())
        loss = criterion(out, y.cuda())
        print(loss)
        loss.backward()
        optimizer.step()
        #print(net.gc1.weight[0,0,0])
        running_loss += loss.item()
    else:
        print(running_loss / len(trainloader))
    running_loss = 0
    try:
        xv,yv = next(iterloader)
        outv = net(xv.cuda())
        lossv = criterion(outv,yv.cuda()).item()
        print('validationloss is:',lossv)
    except StopIteration:
        pass
    
    
