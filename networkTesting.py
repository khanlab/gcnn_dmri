import gPyTorch
from torch.nn.modules.module import Module
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.optim as optim


"""
Here we want to test aspects of the network. Important things to check are:
1) Do the weights update
2) Does weight sharing in the orinentation work?
"""



H=5
h=H+1
w=5*h

X=torch.rand([h,w])
X=X.view([1,1,h,w]).cuda()

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=gPyTorch.gConv2d(1,2,H,shells=1)
        self.conv2=gPyTorch.gConv2d(2,3,H)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)

        return x


net=Net().cuda()

#some loss
def testLoss(output):
    return output.flatten().mean().abs()

#training
optimizer = optim.Adamax(net.parameters(), lr=1e-2)#, weight_decay=0.001)
optimizer.zero_grad()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25, verbose=True)
optimizer.zero_grad

#criterion
criterion=testLoss


out=net(X)
loss=criterion(out)
#weights before update
print('conv1 weight')
print(net.conv1.weight[0,0,:])
print('conv1 expanded weights')
print(net.conv1.kernel_e[0,0,:,:])
print(net.conv1.kernel_e[1,0,:,:])

print("\n")
print("post step")
loss.backward()
optimizer.step()
out=net(X)
print('conv1 weight')
print(net.conv1.weight[0,0,:])
print('conv1 expanded weights')
print(net.conv1.kernel_e[0,0,:,:])
print(net.conv1.kernel_e[1,0,:,:])


#same for conv2
out=net(X)
loss=criterion(out)
#weights before update
print('conv2 weight')
print(net.conv2.weight[0,0,:,:])
print('conv2 expanded weights')
print(net.conv2.kernel_e[0,0,:,:])
print(net.conv2.kernel_e[1,1,:,:])

print("\n")
print("post step")
loss.backward()
optimizer.step()
out=net(X)
print('conv2 weight')
print(net.conv2.weight[0,0,:,:])
print('conv2 expanded weights')
print(net.conv2.kernel_e[0,0,:,:])
print(net.conv2.kernel_e[1,1,:,:])
