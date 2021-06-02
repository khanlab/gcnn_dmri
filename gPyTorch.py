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


#need something that slides in 3d but hen takes 2d convolutions at each point

#input is i,j,k,l,m where l,m is the 2d space to take convutions on
#the filter will be (Cin,Cout,3,3,3,12,7) the 12 is for theta dimension and 7 is for hex filter
#lets say we start at one corner of the 3d image we get 3,3,3,l,m input signal
    # for each (l,m) at a we have a corresponding filter
    # we compute 'internal' 2d convolutions and sum them up (just like scalar convolution)
        #this gives us a (l,m) output for each 3,3,3 filter.
    #now we have to slide the filter and extract the next 3,3,3 part of the input signal
        #this is the part where some clever falttening needs to used?
        #can go with simple loop







class gConv3d(Module):
    def __init__(self, in_channels, out_channels, H, shells=None):
        super(gConv3d, self).__init__()
        self.shells=shells
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.H=H

        #Condition for detecting a scalar layer or regular layer, (this needs to be improved)
        if self.in_channels == self.shells:
            self.deep=0
        else:
            self.deep=1

        #initialize padding basis
        self.I,self.J,self.T= d12.padding_basis(self.H)

        # initialize weight basis if not deep
        if self.deep == 0:
            self.weight = Parameter(Tensor(out_channels, in_channels, 3,3,3,7))
            #self.weight = Parameter(Tensor(out_channels, in_channels, 7))
            self.basis_e_h = d12.basis_expansion(self.deep).detach()  # this can be at net level also
            self.basis_e_t = None

        # initialize weight basis if deep
        elif self.deep == 1:  # and (in_channels % 12)==0:
            self.weight = Parameter(Tensor(out_channels, in_channels, 3,3,3,12, 7))
            #self.weight = Parameter(Tensor(out_channels, in_channels, 12, 7))
            self.basis_e_h, self.basis_e_t = d12.basis_expansion(self.deep) # this can be at the net level also
            self.basis_e_h = self.basis_e_h.detach()
            self.basis_e_t = self.basis_e_t.detach()

        self.bias = Parameter(Tensor(out_channels,3,3,3))
        self.bias_basis = d12.bias_basis(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self,input):
        #use the bases to expand the kernel and the bias
        #input shape is [minibatch,in_channels,i,j,k,l,m]
        kernelsize=3

        sz=np.asarray(input.shape)
        sz[1]=12*int(self.weight.shape[0])

        out=torch.zeros(tuple(sz),requires_grad=True).cuda()
        for i in range(0,input.shape[2]-kernelsize):
            for j in range(0, input.shape[3] - kernelsize):
                for k in range(0, input.shape[4] - kernelsize):
                    chunk=input[:,:,i:i+kernelsize,j:j+kernelsize,k:k+kernelsize,:,:].cuda()
                    out[:,:,i+1,j+1,k+1,:,:]=self.g2dconvolution(chunk)

        return out #+ self.bias[self.bias_basis]#.detach()



    def g2dconvolution(self,chunk):
        #out=torch.zeros([self.weight[0],3,3,3,self.chunk.shape[-2],self.chunk.shape[-1]])
        out=torch.zeros([int(12*self.weight.shape[0]),chunk.shape[-2],chunk.shape[-1]],requires_grad=True).cuda()
        for fi in range(0,3):
            for fj in range(0,3):
                for fk in range(0,3):
                    if self.deep==0:
                        weight=self.weight[:,:,fi,fj,fk,:].cuda()
                    if self.deep == 1:
                        weight = self.weight[:, :, fi, fj, fk, :,:].cuda()

                    #kernel_e = d12.apply_weight_basis(self.weight, self.basis_e_h,self.basis_e_t)#.detach()
                    kernel_e = d12.apply_weight_basis(weight, self.basis_e_h, self.basis_e_t)  # .detach()
                    bias=self.bias[:,fi,fj,fk]
                    bias_e = bias[self.bias_basis]#.detach()
                    temp =F.pad( F.conv2d(chunk[:,:,fi,fj,fk,:,:].float(), kernel_e.float(),bias=bias_e.float()),(1,1,1,1))
                    out = out + d12.pad(temp, self.I, self.J, self.T)
        return out

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


class opool3d(Module):
    def __init__(self,in_channels):
        super(opool3d, self).__init__()
        self.in_channels = in_channels
        self.opool2d=opool(self.in_channels)

    def forward(self,input):
        #size of the input is [mini,in_channels,i,j,k,h,w]
        sz=input.shape
        out=torch.zeros([sz[0],self.in_channels,sz[2],sz[3],sz[4],sz[5],sz[6]])
        for i in range(0,input.shape[2]):
            for j in range(0, input.shape[3]):
                for k in range(0, input.shape[4]):
                    this_input=input[:,:,i,j,k,:,:]
                    out[:,:,i,j,k,:,:]=self.opool2d.forward(this_input)
        return out

class linear3d(Module):
    def __init__(self,in_channels,out_channels,h3d,w3d,d3d,h,w):
        super(linear3d,self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.h3d=h3d
        self.w3d=w3d
        self.d3d=d3d
        self.h=h
        self.w=w
        self.weight = Parameter(Tensor(out_channels, in_channels, h3d,w3d,d3d,h*w))
        self.bias = Parameter(Tensor(out_channels, h3d,w3d,d3d))
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self,input):
        out=torch.zeros([input.shape[0],self.h3d,self.w3d,self.d3d,self.out_channels])
        for i in range(0, self.h3d):
            for j in range(0, self.w3d):
                for k in range(0, self.d3d):
                    vector=input[:,:,i+1,j+1,k+1,:,:] #batch,Cin,h,w
                    vector=vector.reshape(input.shape[0],-1).cuda()
                    thisweight=self.weight[:,:,i,j,k,:].reshape(-1,self.in_channels*self.h*self.w).cuda()
                    bias=self.bias[:,i,j,k].cuda()
                    out[:,i,j,k,:]=F.linear(vector,thisweight,bias)#torch.matmul(vector,thisweight.T)
        return out


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

class gConv_gNorm(Module):
    """
    This class combines the gConv and gNorm layers with a provided activaition
    """
    def __init__(self,Cin,Cout,H,shells=None,activation=None):
        super(gConv_gNorm, self).__init__()
        self.activation=activation
        self.conv = gConv2d(Cin, Cout, H, shells=shells)
        self.gn = GroupNorm(Cout, Cout * 12)

    def forward(self, x):
        if self.activation != None:
            x=self.activation(self.conv(x))
        else:
            x=self.conv(x)
        x = self.gn(x)
        return x

class gNetFromList(Module):
    """
    This class will give us a gConv network from a list of filters
    """
    def __init__(self,H,filterlist,shells,activationlist=None):
        super(gNetFromList, self).__init__()
        self.gConvs=[]
        self.gNorms=[]
        if activationlist is None: #if activation list is None turn it into list of nones to avoid error below
            activationlist=[None for i in range(0,len(filterlist)-1)]
        for i in range(0,len(filterlist)-1):
            if i ==0:
                self.gConvs = [gConv_gNorm(filterlist[i], filterlist[i+1], H, shells=shells,activation=activationlist[i])]  # this is the initilization
            else:
                self.gConvs.append(gConv_gNorm(filterlist[i],filterlist[i+1],H,shells=shells,activation=activationlist[i]))
        self.gConvs=ModuleList(self.gConvs)

    def forward(self,x):
        for gConv in self.gConvs:
            x=gConv(x)
        return x



class gConv5dFromList(gNetFromList):
    def __init__(self,H,filterlist,shells,activationlist=None):
        super(gConv5dFromList,self).__init__(H,filterlist,shells,activationlist=activationlist)

    def forward(self,x):
        #here x has shape [batch,Nin,Nin,Nin,Cin,h,w]
        #this needs to flattened to [batch x Nin^3, Cin,h, w] and then fed through the net work
        sz=x.shape
        x=x.view([sz[0]*sz[1]*sz[2]*sz[3], sz[4],sz[5],sz[6]])
        for gConv in self.gConvs:
            x = gConv(x)
        nsz=x.shape
        return x.view(sz[0],sz[1],sz[2],sz[3],nsz[-3],nsz[-2],nsz[-1])

class opool5d(Module):
    def __init__(self,in_channels):
        super(opool5d,self).__init__()
        self.in_channels=in_channels
        self.opool=opool(self.in_channels)

    def forward(self,input):
        sz=input.shape
        input=input.view([sz[0]*sz[1]*sz[2]*sz[3], sz[4],sz[5],sz[6]])
        input=self.opool.forward(input)
        nsz=input.shape
        return input.view([sz[0],sz[1],sz[2],sz[3],nsz[-3],nsz[-2],nsz[-1]])

class maxpool5d(Module):
    def __init__(self,kernel):
        super(maxpool5d,self).__init__()
        self.kernel=kernel
        self.maxpool2d=MaxPool2d(kernel)

    def forward(self,input):
        sz=input.shape
        input=input.view([sz[0]*sz[1]*sz[2]*sz[3], sz[4],sz[5],sz[6]])
        input=self.maxpool2d(input)
        nsz=input.shape
        return input.view([sz[0],sz[1],sz[2],sz[3],nsz[-3],nsz[-2],nsz[-1]])

class lNet5dFromList(Module):

    def __init__(self,filterlist, activationlist=None):
        super(lNet5dFromList,self).__init__()
        self.filterlist = filterlist
        self.activationlist = activationlist
        self.lNet=lNetFromList(filterlist, activationlist)

    def forward(self,input):
        sz = input.shape
        input = input.view([sz[0] * sz[1] * sz[2] * sz[3], sz[4]* sz[5]* sz[6]])
        input=self.lNet(input)
        nsz=input.shape
        input=input.view([sz[0], sz[1], sz[2], sz[3], nsz[-1]])
        input=input.permute(0,-1,1,2,3)
        return input

#need something