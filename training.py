import torch
from gPyTorch import opool
from torch.nn.modules.module import Module
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d
from gPyTorch import gNetFromList
import pickle
from torch.nn import InstanceNorm3d
from torch.nn import Conv3d
from torch.nn import ModuleList
from torch.nn import DataParallel


def path_from_modelParams(modelParams):
    def array2str(A):
        out=str(A[0])
        for i in range(1,len(A)):
            out=out + '-'+(str(A[i]))
        return out

    path = 'bvec-dirs-' + str(modelParams['bvec_dirs'])
    path = path + '_type-' + str(modelParams['type'])
    path = path + '_Ntrain-' + str(modelParams['Ntrain'])
    path = path + '_Nepochs-' + str(modelParams['Nepochs'])
    path = path + '_patience-' + str(modelParams['patience'])
    path = path + '_factor-' + str(modelParams['factor'])
    path = path + '_lr-' + str(modelParams['lr'])
    path = path + '_batch_size-'+ str(modelParams['batch_size'])
    path = path + '_interp-' + str(modelParams['interp'])
    path = path + '_glayers-'+ array2str(modelParams['gfilterlist'])
    try:
        path = path + '_gactivation0-' + str(modelParams['gactivationlist'][0].__str__()).split()[1]
    except:
        path = path + '_gactivation0-' + str(modelParams['gactivationlist'][0].__str__()).split()[0]
    if modelParams['linfilterlist'] != None:
        print(modelParams['linfilterlist'])
        path = path + '_linlayers-' + array2str(modelParams['linfilterlist'])
        try:
            path = path + '_lactivation0-' + str(modelParams['lactivationlist'][0].__str__()).split()[1]
        except:
            path = path + '_lactivation0-' + str(modelParams['lactivationlist'][0].__str__()).split()[0]
    path = path + '_' + str(modelParams['misc'])

    return modelParams['basepath']+ path


def get_accuracy(net,input_val,target_val):
    x=net(input_val)
    #accuracy =(1- ((pred - target_val) / target_val).abs()).abs()
    norm = x.norm(dim=-1)
    norm = norm.view(-1, 1)
    norm = norm.expand(norm.shape[0], 3)
    x = x / norm
    accuracy = x * target_val
    accuracy = torch.rad2deg( torch.arccos( accuracy.sum(dim=-1).abs()))
    return accuracy.mean().detach().cpu()


class in3d(Module):
    """
    Here we are just moving the 2d space into the batch dimension
    """
    def __init__(self,core):
        super(in3d, self).__init__()
        self.core = core

    def forward(self,x):
        #x will have shape [batch,Nc,Nc,Nc,C,h,w]
        B = x.shape[0]
        Nc = x.shape[1]
        C = x.shape[-3]
        x=x[:,:,:,:,:,self.core==1]
        Ncore = x.shape[-1]
        x = x.moveaxis((-1,-2),(1,2))
        x=x.contiguous()
        x = x.view([B*Ncore,C,Nc,Nc,Nc])
        return x

class out3d(Module):
    """
    Inverse of in3d
    """
    def __init__(self,B,Nc,Ncore,core_inv,I,J,zeros):
        super(out3d, self).__init__()
        self.B = B
        self.Nc = Nc
        self.Ncore = Ncore
        self.core_inv = core_inv
        self.I = I
        self.J = J
        self.zeros = zeros

    def forward(self,x):
        #x will have shape [B*Ncore, C, Nc, Nc, Nc]
        C = x.shape[1]
        x = x.view(self.B,self.Ncore,C,self.Nc,self.Nc,self.Nc)
        x = x[:, self.core_inv, :, :, :, :]  # shape is [B,h,w,C,Nc,Nc,Nc])
        x = x[:, self.I, self.J, :, :, :, :] #padding
        x = x.moveaxis((1, 2, 3), (-2, -1, -3))
        x[:, :, :, :, :, self.zeros == 1] = 0 #zeros

        return x

class in2d(Module):
    """
    Moving 3d dimensions to batch dimension
    """
    def __init__(self):
        super(in2d, self).__init__()

    def forward(self,x):
        #x has shape [batch, Nc, Nc, Nc, C, h, w]
        B = x.shape[0]
        Nc = x.shape[1]
        C = x.shape[-3]
        h = x.shape[-2]
        w = x.shape[-1]
        x = x.view((B*Nc*Nc*Nc,C,h,w)) #patch voxels go in as a batch
        return x

class out2d(Module):
    """
    inverse of in2d
    """
    def __init__(self,B,Nc,Cout):
        super(out2d, self).__init__()
        self.B = B
        self.Nc = Nc
        self.Cout= Cout

    def forward(self,x):
        h=x.shape[-2]
        w=x.shape[-1]
        x = x.view([self.B,self.Nc,self.Nc,self.Nc,self.Cout,h,w])
        return x


class gnet3d(Module): #this module (layer) takes in a 3d patch but only convolves in internal space
    def __init__(self,H,filterlist,shells=None,activationlist=None): #shells should be same as filterlist[0]
        super(gnet3d,self).__init__()
        self.filterlist = filterlist
        self.gconvs=gNetFromList(H,filterlist,shells,activationlist= activationlist)
        self.pool = opool(filterlist[-1])

    def forward(self,x):
        x = self.gconvs(x) #usual g conv
        x = self.pool(x) # shape [batch,Nc^3,filterlist[-1],h,w
        return x

#this is a class to make lists out of
class conv3d(Module):
    """
    This class combines conv3d and batch norm layers and applies a provided activation
    """
    def __init__(self,Cin,Cout,activation=None,norm=True):
        super(conv3d,self).__init__()
        self.activation= activation
        self.conv = Conv3d(Cin,Cout,3,padding=1)
        self.norm = InstanceNorm3d(Cout)
        self.batch_norm = norm

    def forward(self,x):
        if self.activation!=None:
            x=self.activation(self.conv(x))
        else:
            x=self.conv(x)

        if self.batch_norm:
            x = self.norm(x)

        return x

#this makes the list for 3d convs
class conv3dList(Module):
    def __init__(self,filterlist,activationlist=None):
        super(conv3dList,self).__init__()
        self.conv3ds=[]
        self.filterlist = filterlist
        self.activationlist = activationlist
        if activationlist is None:
            self.activationlist = [None for i in range(0,len(filterlist)-1)]
        for i in range(0,len(filterlist)-1):
            if i==0:
                self.conv3ds=[conv3d(filterlist[i],filterlist[i+1],self.activationlist[i])]
            else:
                norm = True
                if i == len(filterlist) - 2:
                    norm = False
                self.conv3ds.append(conv3d(filterlist[i],filterlist[i+1],self.activationlist[i],norm=norm))
        self.conv3ds = ModuleList(self.conv3ds)

    def forward(self,x):
        for i,conv in enumerate(self.conv3ds):
            x = conv(x)
        return x


class residualnet5d(Module):
    def __init__(self,filterlist3d,activationlist3d,filterlist2d,activationlist2d,H,shells,B,Nc,Ncore,core,core_inv,I,J,
                 zeros):
        super(residualnet5d,self).__init__()
        #params
        self.flist3d = filterlist3d
        self.alist3d = activationlist3d
        self.flist2d = filterlist2d
        self.alist2d = activationlist2d

        self.layer_in3d = in3d(core)
        self.layer_out3d = out3d(B,Nc,Ncore,core_inv,I,J,zeros)
        self.layer_in2d = in2d()
        self.layer_out2d = out2d(B,Nc,self.flist2d[-1])


        self.conv3ds = conv3dList(filterlist3d,activationlist3d)
        self.gconvs = gnet3d(H,filterlist2d,shells,activationlist2d)

        self.conv3ds = DataParallel(self.conv3ds)
        self.gconvs = DataParallel(self.gconvs)

    def forward(self,x):

        x=self.layer_in3d(x)
        x=self.conv3ds(x)
        x=self.layer_out3d(x)

        x=self.layer_in2d(x)
        x=self.gconvs(x)
        x=self.layer_out2d(x)

        return x


class trainer:
    def __init__(self,modelParams,Xtrain=None,Ytrain=None,FA=None,B=None,Nc=None,Ncore=None,core=None,core_inv=None,I=None,\
                                                                                                            J=None,\
                                                                                                        zeros=None):
        """
        Class to create and train networks
        :param modelParams: A dict with all network parameters
        :param Xtrain: Cuda Xtrain data
        :param Ytrain: Cuda Ytrain data
        """
        self.modelParams=modelParams
        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        self.net=[]
        self.B = B
        self.Nc = Nc
        self.Ncore = Ncore
        self.core = core
        self.core_inv = core_inv
        self.I = I
        self.J = J
        self.zeros = zeros
        self.FA = FA

    def mul_by_FA(inputs,targets,FA):
        h = inputs.shape[-2]
        w = inputs.shape[-1]
        inputs = inputs.view(-1,h*w)
        targets = targets.view(-1,h*w)
        FA = FA.view(-1)
        inputs= torch 

    def makeNetwork(self):
        if self.modelParams['misc']=='residual5d':
            self.net = residualnet5d(self.modelParams['filterlist3d'],
                                  self.modelParams['activationlist3d'],
                                  self.modelParams['gfilterlist'],
                                  self.modelParams['gactivationlist'],
                                  self.modelParams['H'],
                                  self.modelParams['shells'],
                                  self.B,
                                  self.Nc,
                                  self.Ncore,
                                  self.core,
                                  self.core_inv,
                                  self.I,
                                  self.J,
                                  self.zeros)

    def train(self):
        outpath = path_from_modelParams(self.modelParams)
        lossname = outpath + 'loss.png'
        netname = outpath + 'net'
        criterion = self.modelParams['loss']
        optimizer = optim.Adamax(self.net.parameters(), lr=self.modelParams['lr'])  # , weight_decay=0.001)
        optimizer.zero_grad()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.modelParams['factor'],
                                      patience=self.modelParams['patience'],
                                      verbose=True)
        running_loss = 0
        train = torch.utils.data.TensorDataset(self.Xtrain, self.Ytrain,self.FA)
        trainloader = DataLoader(train, batch_size=self.modelParams['batch_size'])

        epochs_list = []
        loss_list = []

        for epoch in range(0, self.modelParams['Nepochs']):
            print(epoch)
            for n, (inputs, targets,FA) in enumerate(trainloader, 0):
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                output = self.net(inputs.cuda())
                output = FA[:,:,:,:,None,None,None].to(output.device.type)*output
                targets = FA[:,:,:,:,None,None,None].to(targets.device.type)*targets
                loss = criterion(output, targets.cuda())
                print(loss.shape)
                loss = loss.sum()
                print(loss)
                #print(self.net.gconvs.gconvs.gConvs[-1].conv.weight[0, 0])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                print(running_loss / len(trainloader))
                loss_list.append(running_loss / len(trainloader))
                epochs_list.append(epoch)
                # print(output[0])
                if np.isnan(running_loss / len(trainloader)) == 1:
                    break

            scheduler.step(running_loss)
            running_loss = 0.0
            if (epoch % 1) == 0:
                fig_err, ax_err = plt.subplots()
                ax_err.plot(epochs_list, np.log10(loss_list))
                if lossname is None:
                    lossname = 'loss.png'
                plt.savefig(lossname)
                plt.close(fig_err)
                if netname is None:
                    netname = 'net'
                torch.save(self.net.state_dict(), netname)

    def save_modelParams(self):
        outpath = path_from_modelParams(self.modelParams)
        with open(outpath + 'modelParams.pkl','wb') as f:
            pickle.dump(self.modelParams,f,pickle.HIGHEST_PROTOCOL)




