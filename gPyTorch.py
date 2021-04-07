from torch.nn.modules.module import Module
import dihedral12 as d12
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F
import torch
import math
import numpy as np


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
            self.basis_e_h = d12.basis_expansion(self.deep) #this can be at net level also
            self.basis_e_t = None

        # initialize weight basis if deep
        elif self.deep == 1:  # and (in_channels % 12)==0:
            self.weight = Parameter(Tensor(out_channels, in_channels, 12, self.kernel_size))
            self.basis_e_h, self.basis_e_t = d12.basis_expansion(self.deep) #this can be at the net level also

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
        self.bias_e=self.bias[self.bias_basis]
        self.kernel_e=d12.apply_weight_basis(self.weight,self.basis_e_h,self.basis_e_t)
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
        return input_pool.cuda()