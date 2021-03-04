from torch.nn.modules.module import Module
import dihedral12 as d12
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
import math


class gConv2d(Module):
    def __init__(self, in_channels, out_channels, H, shells=None):
        super(gConv2d,self).__init__()
        self.shells=shells
        self.kernel_size=7

        self.out_channels=out_channels
        self.in_channels= in_channels

        self.kernel_size_e=[]
        self.bias_e=[]
        self.H=H
        self.deep=[]

        #Condition for detecting a scalar layer or regular layer, (this needs to be improved)
        if self.in_channels == self.shells:
            self.deep=0
        else:
            self.deep=1

        self.I,self.J,self.T= d12.padding_basis(self.H)

        if self.deep==0:
            self.weight = Parameter(Tensor(out_channels, in_channels, self.kernel_size))
            self.basis_e_h = d12.basis_expansion(self.deep) #this can be at net level also
            self.basis_e_t = 0

        elif self.deep == 1:  # and (in_channels % 12)==0:
            self.weight = Parameter(Tensor(out_channels, in_channels, 12, self.kernel_size))
            self.basis_e_t, self.basis_e_h = d12.basis_expansion(self.deep) #this can be at the net level also

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self,input):
        pass

