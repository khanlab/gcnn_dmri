from torch.nn.modules.module import Module
import dihedral12 as d12
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init, GroupNorm, ModuleList
from torch.nn import functional as F
import torch
import math
import numpy as np
from torch.nn import MaxPool2d
from torch.nn import GroupNorm, Linear, ModuleList
from torch.nn import Conv3d



class gConv2d(Module):
    def __init__(self, in_channels, out_channels, H, shells=None):
        super(gConv2d,self).__init__()
        self.shells=shells
        self.kernel_size=7

        self.out_channels=out_channels
        self.in_channels= in_channels

        self.kernel_size_e=[]
        self.kernel_e = [] #expanded kernel
        self.bias_e=[] #expanded bias
        self.H=H
        self.deep=[]

        #Condition for detecting a scalar layer or regular layer, (this needs to be improved)
        if self.in_channels == self.shells:
            self.deep=0
        else:
            self.deep=1

        #initialize padding basis
        self.I,self.J,self.T= d12.padding_basis(self.H)


        #initialize weight basis if not deep
        if self.deep==0:
            self.weight = Parameter(Tensor(out_channels, in_channels, self.kernel_size))
            self.basis_e_h = d12.basis_expansion(self.deep)#.detach() #this can be at net level also
            self.basis_e_t = None

        # initialize weight basis if deep
        elif self.deep == 1:  # and (in_channels % 12)==0:
            self.weight = Parameter(Tensor(out_channels, in_channels, 12, self.kernel_size))
            self.basis_e_h, self.basis_e_t = d12.basis_expansion(self.deep) #this can be at the net level also
            self.basis_e_h=self.basis_e_h#.detach()
            self.basis_e_t=self.basis_e_t#.detach()

        self.bias = Parameter(Tensor(out_channels))
        self.bias_basis = d12.bias_basis(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self,input):
        #use the bases to expand the kernel and the bias
        self.bias_e=self.bias[self.bias_basis]#.detach()
        self.kernel_e=d12.apply_weight_basis(self.weight,self.basis_e_h,self.basis_e_t)#.detach()
        out=F.conv2d(input.float(),self.kernel_e.float(),bias=self.bias_e.float())
        out=F.pad(out,(1,1,1,1))
        #return out
        return d12.pad(out,self.I,self.J,self.T)


class opool(Module):
    #layer for orientation pooling
    def __init__(self,in_channels):
        super(opool,self).__init__()
        self.in_channels=in_channels

    def forward(self,input):
        batch_size=input.shape[0]
        h=input.shape[-2]
        w=input.shape[-1]
        D=12
        input_pool=torch.zeros(batch_size,self.in_channels,h,w, requires_grad=False)
        pooled, ind = F.max_pool3d(input, [12, h, w], return_indices=True)
        inds = np.asarray(ind.detach().reshape(batch_size * self.in_channels).cpu().numpy())
        subs = np.asarray(np.unravel_index(inds, [D * self.in_channels, h, w]))
        subs = subs.reshape(3, batch_size, self.in_channels)
        for b in range(0, batch_size):
            input_pool[b, :, :, :] = input[b, subs[0, b, :], :, :].clone()
        return input_pool#.cuda()

class gConv_gNorm(Module):
    """
    This class combines the gConv and gNorm layers with a provided activaition
    """
    def __init__(self,Cin,Cout,H,shells=None,activation=None,norm=True):
        super(gConv_gNorm, self).__init__()
        self.activation=activation
        self.conv = gConv2d(Cin, Cout, H, shells=shells)
        self.gn = GroupNorm(Cout, Cout * 12)
        self.norm =norm

    def forward(self, x):
        if self.activation != None:
            x=self.activation(self.conv(x))
        else:
            x=self.conv(x)
        if self.norm:
            x = self.gn(x)
        return x

class gNetFromList(Module):
    """
    This class will give us a gConv network from a list of filters
    """
    def __init__(self,H,filterlist,shells,activationlist=None,multigpu=False):
        super(gNetFromList, self).__init__()
        self.multigpu = multigpu

        print("Length of activation list is", len(activationlist))
        if activationlist is None: #if activation list is None turn it into list of nones to avoid error below
                activationlist=[None for i in range(0,len(filterlist)-1)]

        if ((multigpu) & (len(activationlist)>2)): #we want to shard this network over two gpu
            print("Using 2 gpus in gNetFromList")
            N_layers = len(filterlist)
            N_half = int(N_layers/2)
        
            filterlist_1 = filterlist[0:N_half]
            filterlist_2 = filterlist[N_half:]
            
            filterlist_2.insert(0,filterlist_1[-1])  #since the first element in the filter list is the dimension of the input, we need to insert that here
            
            N_1 = len(filterlist_1)
            N_2 = len(filterlist_2)

            activationlist_1 = activationlist[0:N_1-1]
            activationlist_2 = activationlist[N_1-1:]

            self.gConvs1 = []
            self.gConvs2 = []

            for i in range(0,len(filterlist_1)-1): #first gpu
                if i==0:
                    self.gConvs1 = [gConv_gNorm(filterlist_1[i], filterlist_1[i+1], H, shells=shells,activation=activationlist_1[i])]  # this is the initilization 
                else:
                    norm = True
                    if (i==len(filterlist_1)-2):
                        norm =False
                    self.gConvs1.append(gConv_gNorm(filterlist_1[i], filterlist_1[i+1], H, shells=0,
                                                    activation=activationlist_1[i],norm=norm))
                self.gConvs1 = ModuleList(self.gConvs1).cuda(0)

            for i in range(0,len(filterlist_2)-1): #second gpu
                if i==0:                                                           #notice shells=0
                    self.gConvs2 = [gConv_gNorm(filterlist_2[i], filterlist_2[i+1], H, shells=0,activation=activationlist_2[i])]  # this is the initilization 
                else:
                    norm = True
                    if (i == len(filterlist_2) - 2):
                        norm = False
                    self.gConvs2.append(gConv_gNorm(filterlist_2[i], filterlist_2[i+1], H, shells=0,
                                                    activation=activationlist_2[i],norm=norm))
                self.gConvs2 = ModuleList(self.gConvs2).cuda(1)

        else:
            self.gConvs=[]
            self.gNorms=[]
            
            for i in range(0,len(filterlist)-1):
                if i ==0:
                    self.gConvs = [gConv_gNorm(filterlist[i], filterlist[i+1], H, shells=shells,activation=activationlist[i])]  # this is the initilization
                else:
                    norm = True
                    if (i == len(filterlist) - 2):
                        norm = False
                    self.gConvs.append(gConv_gNorm(filterlist[i],filterlist[i+1],H,shells=0,
                                                   activation=activationlist[i],norm=norm))
            self.gConvs=ModuleList(self.gConvs)

    def forward(self,x):
        if self.multigpu:
            x = x.cuda(0)
            for gConv in self.gConvs1:
                x = gConv(x)
            x = x.cuda(1)
            for gConv in self.gConvs2:
                x = gConv(x)

        else:
            for gConv in self.gConvs:
                x=gConv(x)
            
        return x



