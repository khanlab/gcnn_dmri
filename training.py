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

class lNetFromList(Module):
    """
    This class will give us a linear network from a list of filters
    """
    def __init__(self,filterlist,activationlist=None):
        super(lNetFromList,self).__init__()
        self.activationlist=activationlist
        self.lins=[]
        if activationlist is None: #if activation list is None turn it into list of nones to avoid error below
            self.activationlist=[None for i in range(0,len(filterlist)-1)]
        for i in range(0,len(filterlist)-1):
            self.lins.append(Linear(filterlist[i],filterlist[i+1]))
        self.lins=ModuleList(self.lins)

    def forward(self,x):
        for idx,lin in enumerate(self.lins):
            activation=self.activationlist[idx]
            if activation == None:
                x=lin(x)
            else:
                x=activation(lin(x))
        return x

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

class gnet(Module):
    """
    This will create the entire network
    """
    def __init__(self,linfilterlist,gconfilterlist,shells,H,lactivationlist=None,gactivationlist=None):
        super(gnet,self).__init__()
        self.input=input
        self.linfilterlist=linfilterlist
        self.gconfilterlist=gconfilterlist
        self.lactivationlist=lactivationlist
        self.gactivationlist=gactivationlist
        self.shells=shells
        self.H=H
        self.h = 5*(self.H+1)
        self.w = self.H+1
        self.last = self.gconfilterlist[-1]

        self.gConvs=gNetFromList(self.H,self.gconfilterlist,shells,activationlist=gactivationlist)
        self.pool = opool(self.last)
        self.mx = MaxPool2d([2,2])
        self.lins=lNetFromList(linfilterlist,activationlist=lactivationlist)

    def forward(self,x):
        x = self.gConvs(x)
        x = self.pool(x)
        x = self.mx(x)
        x = x.view(-1,int(self.last * self.h * self.w / 4))
        x = self.lins(x)

        return x


class gnet3d(Module): #this module (layer) takes in a 3d patch but only convolves in internal space
    def __init__(self,H,filterlist,shells=None,activationlist=None,multigpu=False): #shells should be same as filterlist[0]
        super(gnet3d,self).__init__()
        self.filterlist = filterlist
        self.gconvs=gNetFromList(H,filterlist,shells,activationlist= activationlist,multigpu=multigpu)
        self.pool = opool(filterlist[-1])

    def forward(self,x):
        #x will have shape [batch,Nc,Nc,Nc,C,h,w]
        B=x.shape[0]
        Nc = x.shape[1]
        C = x.shape[-3]
        h = x.shape[-2]
        w = x.shape[-1]
        # print('we are in gnet3d')
        # print(x.shape)
        x = x.view((B*Nc*Nc*Nc,C,h,w)) #patch voxels go in as a batch
        #x = x.reshape((B*Nc*Nc*Nc,C,h,w)) #patch voxels go in as a batch
        # print(x.shape)
        x = self.gconvs(x) #usual g conv
        x = self.pool(x) # shape [batch,Nc^3,filterlist[-1],h,w
        #x = x.view([B,Nc,Nc,Nc,self.filterlist[-1],h,w])
        x = x.view([B,Nc,Nc,Nc,h,w]) #this is under the assumption that last filter is 1 with None activation
        #x = x.reshape([B,Nc,Nc,Nc,h,w]) #this is under the assumption that last filter is 1 with None activation
        # print(x.shape)
        return x

#this is a class to make lists out of
class conv3d(Module):
    """
    This class combines conv3d and batch norm layers and applies a provided activation
    """
    def __init__(self,Cin,Cout,activation=None):
        super(conv3d,self).__init__()
        self.activation= activation
        self.conv = Conv3d(Cin,Cout,3,padding=1)
        self.norm = InstanceNorm3d(Cout)


    def forward(self,x):
        if self.activation!=None:
            x=self.activation(self.conv(x))
            x=self.norm(x)

        else:
            x=self.conv(x)
            x=self.norm(x)
        return x

#this makes the list for 3d convs
class conv3dList(Module):
    def __init__(self,filterlist,activationlist=None):
        super(conv3dList,self).__init__()
        self.conv3ds=[]
        self.filterlist = filterlist
        if activationlist is None:
            self.activationlist = [None for i in range(0,len(filterlist)-1)]
        for i in range(0,len(filterlist)-1):
            if i==0:
                self.conv3ds=[conv3d(filterlist[i],filterlist[i+1],activationlist[i])]
            else:
                self.conv3ds.append(conv3d(filterlist[i],filterlist[i+1],activationlist[i]))
        self.conv3ds = ModuleList(self.conv3ds)

    def forward(self,x):
        H=x.shape[-2]-1
        h = H + 1
        w = 5 * (H + 1)
        Nc = x.shape[1]
        B = x.shape[0]
        C=1#x.shape[-3]
        for i,conv in enumerate(self.conv3ds):
            if i==0:
                #input size is [B,Nc,Nc,Nc,h,w]
                
                ##-----directions as channels
                #x = x.view([B,Nc,Nc,Nc,C*h*w]).moveaxis(-1,1) # Instead of this, maybe a simpler approach is to put h*w in batch dimension
                
                ## -----directions as batch
                x = x.moveaxis((-2,-1),(1,2)).view([B*h*w,C,Nc,Nc,Nc])
                #x = x.moveaxis((-2,-1),(1,2)).reshape([B*h*w,C,Nc,Nc,Nc])
                
                x = conv(x)

            elif i==len(self.conv3ds)-1:
                x=conv(x)
                
                ## -------directions as channels
                #x = x.moveaxis(1,-1)
                #C = int(self.filterlist[-1]/(h*w)) #is this right?
                #x = x.view([B,Nc,Nc,Nc,C,h,w]) #this will be input for internal space convolutions
                
                ## ------directions as batch
                #incomping shape is [batch*h*w,C,Nc,Nc,Nc]
                C = x.shape[1]  
                x=x.view(B,h,w,C,Nc,Nc,Nc)
                #x=x.reshape(B,h,w,C,Nc,Nc,Nc)
                x = x.moveaxis((1,2,3),(-2,-1,-3))
            else:
                x = conv(x)
        return x



class residualnet(Module):
    def __init__(self,gfilterlist,shells,H,gactivationlist=None):
        super(residualnet,self).__init__()
        self.gfilterlist=gfilterlist
        self.gactivationlist=gactivationlist
        self.shells=shells
        self.H=H
        self.gConvs=gNetFromList(self.H,self.gfilterlist,shells,activationlist=self.gactivationlist)
        self.opool = opool(self.gfilterlist[-1])

    def forward(self,x):
        x=self.gConvs(x)
        x=self.opool(x)

        return x

class residualnet5d(Module):
    def __init__(self,filterlist3d,activationlist3d,filterlist2d,activationlist2d,H,shells,multigpu=False):
        super(residualnet5d,self).__init__()
        #params
        self.flist3d = filterlist3d
        self.alist3d = activationlist3d
        self.flist2d = filterlist2d
        self.alist2d = activationlist2d
        self.H = H 
        self.h = H+1
        self.w = 5*(H+1)
        self.shells =shells
        self.multigpu = multigpu

        # #network layers 
        # if multigpu:
        #     print('Using multi gpus')
        #     self.conv3ds = conv3dList(filterlist3d,activationlist3d).cuda(0)
        #     self.gconvs = gnet3d(H,filterlist2d,shells,activationlist2d).cuda(1)
        # else:
        self.conv3ds = conv3dList(filterlist3d,activationlist3d).cuda(0)
        self.gconvs = gnet3d(H,filterlist2d,shells,activationlist2d,multigpu=multigpu)
        

    def forward(self,x):
        
        #if self.multigpu:
        #    x = x.cuda(0)

        x=self.conv3ds(x)
        
        #if self.multigpu:
        #    x = x.cuda(1)
          
    
        x=self.gconvs(x)
        
        return x


class trainer:
    def __init__(self,modelParams,Xtrain=None,Ytrain=None,multigpu=False):
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
        self.multigpu = multigpu

    def makeNetwork(self):
        if self.modelParams['misc']=='residual':
            self.net = residualnet(self.modelParams['gfilterlist'],self.modelParams['shells'],self.modelParams['H'],
                                   self.modelParams['gactivationlist'])
        if self.modelParams['misc']=='residual5d':
            self.net = residualnet5d(self.modelParams['filterlist3d'],
                                  self.modelParams['activationlist3d'],
                                  self.modelParams['gfilterlist'],
                                  self.modelParams['gactivationlist'],
                                  self.modelParams['H'],
                                  self.modelParams['shells'],
                                  multigpu=self.multigpu)
        else:
            self.net = gnet(self.modelParams['linfilterlist'],self.modelParams['gfilterlist'] ,
                            self.modelParams['shells'],self.modelParams['H'],
                            self.modelParams['lactivationlist'],
                            self.modelParams['gactivationlist'])
        #self.net = self.net.cuda()
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
        train = torch.utils.data.TensorDataset(self.Xtrain, self.Ytrain)
        trainloader = DataLoader(train, batch_size=self.modelParams['batch_size'])

        epochs_list = []
        loss_list = []

        for epoch in range(0, self.modelParams['Nepochs']):
            print(epoch)
            for n, (inputs, targets) in enumerate(trainloader, 0):
                optimizer.zero_grad()
                if self.multigpu:
                    torch.cuda.empty_cache()
                    output = self.net(inputs.cuda(0))#.cpu()
                else:
                    torch.cuda.empty_cache()
                    output = self.net(inputs.cuda()).cpu()
                loss = criterion(output, targets)
                loss = loss.sum()
                print(loss)
                #loss.cuda(0).backward()
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




