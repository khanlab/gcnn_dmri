import gPyTorch
from torch.nn.modules.module import Module
import dihedral12 as d12
import numpy as np

class Net(Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1=gPyTorch.gConv2d(2,4,4,shells=2)
        self.conv2=gPyTorch.gConv2d(5,2,4)


    def forward(self,x):
        return x

net=Net().cuda()

for i in range(0,2):
    for j in range(0,5):
        for k in range(0,12):
            for l in range(0,7):
                net.conv2.weight[i,j,k,l]=(10000*i+1000*j+10*k+l)/10000

weights_conv1_e=d12.apply_weight_basis(net.conv1.weight, net.conv1.basis_e_h)
weights_conv2_e=d12.apply_weight_basis(net.conv2.weight, net.conv2.basis_e_h,net.conv2.basis_e_t)


h = 5 * (H + 1)
w = H + 1

I, J, T = np.meshgrid(np.arange(0, h), np.arange(0, w),np.arange(0, 12), indexing='ij')
I_out, J_out, T_out = np.meshgrid(np.arange(0, h), np.arange(0, w),np.arange(0, 12), indexing='ij')

I_out, J_out, T_out=d12.padding_basis(4)







