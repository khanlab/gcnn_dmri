import torch
from gPyTorch import opool
from torch.nn.modules.module import Module
import numpy as np
from torch.nn import functional as F
from gPyTorch import opool
from torch.nn import ModuleList
from torch.nn import Conv3d
from torch import nn

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d
from gPyTorch import gNetFromList
import pickle
#c
import preprocessing
import nibabel as nib
import torch
import numpy as np
import diffusion
import icosahedron
import dihedral12 as d12
from torch.nn import InstanceNorm3d


############################################## DATA ###################################################
##load sources to generate training data later
#input
diff6 = diffusion.diffVolume('/home/u2hussai/scratch/tempdata/6')
in_shp = diff6.vol.shape
#output
dti=diffusion.dti('/home/u2hussai/scratch/tempdata/90/dti','/home/u2hussai/scratch/tempdata/90/nodif_brain_mask.nii.gz')

##use the size of input diffusion volume to get patches
#get coordinates
x=np.arange(0,in_shp[0])
y=np.arange(0,in_shp[1])
z=np.arange(0,in_shp[2])
x,y,z = np.meshgrid(x,y,z,indexing='ij')
x=torch.from_numpy(x)
y=torch.from_numpy(y)
z=torch.from_numpy(z)
#unfold coordinates
Nc = 10 #patch size
x=x.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
y=y.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
z=z.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
coarse_shp = x.shape #save patch labels for later
#collapse patch indices
x = x.reshape((-1,)+tuple(x.shape[-3:]))
y = y.reshape((-1,)+tuple(y.shape[-3:]))
z = z.reshape((-1,)+tuple(z.shape[-3:]))

##mask and FA
#unfold
mask = torch.from_numpy(dti.mask.get_fdata()).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
FA = torch.from_numpy(dti.FA.get_fdata()).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
#flatten patch labels (indices for each patch) and patch indices (indices for voxels in patch)
mask = mask.reshape((-1,)+ tuple(mask.shape[-3:])) #flatten patch labels
FA = FA.reshape((-1,)+ tuple(FA.shape[-3:])) #flatten patch labels
mask = mask.reshape(mask.shape[0],-1) #flatten patch voxels
FA = FA.reshape(FA.shape[0],-1) #flatten patch voxels
#take mean for each patch
mask = mask.mean(dim=-1)
FA = FA.mean(dim=-1)

##get coordinates for data extraction
N=20 #---------------number of training patches (3D images)-----------------------#
occ_inds = np.where(FA>0.2)[0] #extract patches based on mean FA or mask
oi=np.random.randint(0,len(occ_inds)-1,N) #get random patches
xpp=x[occ_inds[oi]] #extract coordinates
ypp=y[occ_inds[oi]]
zpp=z[occ_inds[oi]]

## generate training data
H = 7 #----------------dimensions of icosahedron internal space-------------------#
h = H+1
w = 5*h
ico = icosahedron.icomesh(m=H-1) 
I, J, T = d12.padding_basis(H=H) #for padding
xp= xpp.reshape(-1).numpy() #fully flattened coordinates
yp= ypp.reshape(-1).numpy()
zp= zpp.reshape(-1).numpy()
voxels=np.asarray([xp,yp,zp]).T #putting them in one array
#inputs
diff6.makeInverseDistInterpMatrix(ico.interpolation_mesh) #interpolation initiation
S0X, X = diff6.makeFlat(voxels,ico) #interpolate
X = X[:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
shp=tuple(xpp.shape) + (h,w) #we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
X = X.reshape(shp)
#output (labels)
Y=dti.icoSignalFromDti(ico)
Y = Y[:,:,:,I[0,:,:],J[0,:,:]]
Y=Y[xp,yp,zp]
Y=Y.reshape(shp) #same shape for outputs
print('that shape for outputs is ', Y.shape)
# diff90 = diffusion.diffVolume('/home/u2hussai/scratch/tempdata/90')
# diff90.makeInverseDistInterpMatrix(ico.interpolation_mesh)
# S0Y, Y = diff90.makeFlat(voxels,ico) #interpolate
# Y = Y[:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
# shp=tuple(xpp.shape) + (h,w) #we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
# Y = Y.reshape(shp)
#standardize both
X = (X - X.mean())/X.std()
X = torch.from_numpy(X).contiguous().float()
Y = (Y - Y.mean())/Y.std()
Y = torch.from_numpy(Y).contiguous().float()


############################################ NETWORK ####################################################

class gnet3d(Module): #this module (layer) takes in a 3d patch but only convolves in internal space
    def __init__(self,H,filterlist,shells=None,activationlist=None): #shells should be same as filterlist[0]
        super(gnet3d,self).__init__()
        self.filterlist = filterlist
        self.gconvs=gNetFromList(H,filterlist,shells,activationlist= activationlist)
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
        # print(x.shape)
        x = self.gconvs(x) #usual g conv
        x = self.pool(x) # shape [batch,Nc^3,filterlist[-1],h,w
        #x = x.view([B,Nc,Nc,Nc,self.filterlist[-1],h,w])
        x = x.view([B,Nc,Nc,Nc,h,w]) #this is under the assumption that last filter is 1 with None activation
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
            x = self.norm(x)
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
                x = x.moveaxis((1,2,3),(-2,-1,-3))
            else:
                x = conv(x)
        return x

#the actual network
class Net(Module):
    def __init__(self,H):
        super(Net,self).__init__()
        self.H=H
        #self.shp = shp
        self.h = H+1
        self.w = 5*(H+1)
        #self.Nc = shp[1]
        #self.Nc3 = shp[1]*shp[1]*shp[1]
        self.conv3ds=conv3dList([1,4,4,4],
                                [F.relu, F.relu,F.relu])
        self.gConvs= gnet3d(H,[4,4,4,1],shells=4,
                            activationlist= [F.relu,F.relu,None])
        
    def forward(self,x):
        x=self.conv3ds(x)
        x=self.gConvs(x)
        return x

net = Net(H).float().cuda() #instance of network
#net.load_state_dict(torch.load('./net'))

############################################### TRAINING ######################################################### 
criterion = nn.MSELoss()
#criterion=nn.SmoothL1Loss()
#criterion=nn.CosineSimilarity()
#criterion=Myloss
#
#
optimizer = optim.Adamax(net.parameters(), lr=1e-1)#, weight_decay=0.001)
optimizer.zero_grad()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
#
running_loss = 0

train = torch.utils.data.TensorDataset(X, Y-X)
trainloader = DataLoader(train, batch_size=1)

for epoch in range(0, 20):
    print(epoch)
    for n, (inputs, target) in enumerate(trainloader, 0):
        # print(n)

        optimizer.zero_grad()

        #print(inputs.shape)
        output = net(inputs.cuda()).cpu()

        loss = criterion(output, target)
        loss=loss.sum()
        print(loss)
        loss.cuda().backward()
        #print(net.lin3d.weight[2,2,4,4,4,3])
        #print(net.conv1.weight)
        optimizer.step()
        running_loss += loss.item()
    else:
        print(running_loss / len(trainloader))
    # if i%N_train==0:
    #    print('[%d, %5d] loss: %.3f' %
    #          ( 1, i + 1, running_loss / 100))
    scheduler.step(running_loss)
    running_loss = 0.0

torch.save(net.state_dict(),'./net')


###################################### PREDICTING ############################################

#get inputs on whole volume
occ_inds = np.where(mask>0)[0] 
xpp = x[occ_inds]
ypp = y[occ_inds]
zpp = z[occ_inds]
xp= xpp.reshape(-1).numpy() #fully flattened coordinates
yp= ypp.reshape(-1).numpy()
zp= zpp.reshape(-1).numpy()
voxels=np.asarray([xp,yp,zp]).T #putting them in one array
S0X, X = diff6.makeFlat(voxels,ico) #interpolate
X = X[:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
shp=tuple(xpp.shape) + (h,w) #we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
X = X.reshape(shp)
#standardize inputs
mean= X.mean() #need mean and std for later
std=X.std()
X = (X - X.mean())/X.std()
X = torch.from_numpy(X).contiguous().float()

#save the output also to check that label signal is correct
Y=dti.icoSignalFromDti(ico)

#predict
pred=torch.zeros_like(X)
batch_size=1
for i in range(0,X.shape[0],batch_size):
    print(i)
    pred[i:i+batch_size]=(net(X[i:i+batch_size].cuda()).cpu() + X[i:i+batch_size]).detach()

out = np.zeros(diff6.vol.shape[0:3]+(h,w))
oldshp=pred.shape
pred = pred.view(-1,h,w)
pred = pred*std + mean

out[xp,yp,zp]=pred

#make nifti
basis=np.zeros([h,w])
for c in range(0,5):
    basis[1:H,c*h+1:(c+1)*h-1]=1
N=len(basis[basis==1])+1

print('Number of bdirs is: ', N)

N_random=2*w
rng=np.random.default_rng()
inds=rng.choice(N-2,size=N_random,replace=False)+1
inds[0]=0

bvals=np.zeros(N_random)
x_bvecs=np.zeros(N_random)
y_bvecs=np.zeros(N_random)
z_bvecs=np.zeros(N_random)

x_bvecs[1:]=ico.X_in_grid[basis==1].flatten()[inds[1:]]
y_bvecs[1:]=ico.Y_in_grid[basis==1].flatten()[inds[1:]]
z_bvecs[1:]=ico.Z_in_grid[basis==1].flatten()[inds[1:]]

bvals[1:]=1000

sz=out.shape
diff_out=np.zeros([sz[0],sz[1],sz[2],N_random])
diff_out[:,:,:,0]=diff6.vol.get_fdata()[:,:,:,diff6.inds[0]].mean(-1)
i, j, k = np.where(diff6.mask.get_fdata() == 1)

#for checking labels
labels_out=np.zeros([sz[0],sz[1],sz[2],N_random])
labels_out[:,:,:,0]=diff6.vol.get_fdata()[:,:,:,diff6.inds[0]].mean(-1)


for p in range(0,len(i)):
    signal =out[i[p],j[p],k[p]]
    signal = signal[basis==1].flatten()
    diff_out[i[p],j[p],k[p],1:] = signal[inds[1:]]
    label_signal = Y[i[p],j[p],k[p]]
    label_signal = label_signal[basis==1].flatten()
    labels_out[i[p],j[p],k[p],1:] = label_signal[inds[1:]]

diff_out=nib.Nifti1Image(diff_out,diff6.vol.affine)
nib.save(diff_out,'./data_network.nii.gz')

labels_out = nib.Nifti1Image(labels_out,diff6.vol.affine)
nib.save(labels_out, './data_labels.nii.gz')

#write the bvecs and bvals
fbval = open('./bvals_network', "w")
for bval in bvals:
    fbval.write(str(bval)+" ")
fbval.close()

fbvecs = open('./bvecs_network',"w")
for x in x_bvecs:
    fbvecs.write(str(x)+ ' ')
fbvecs.write('\n')
for y in y_bvecs:
    fbvecs.write(str(y)+ ' ')
fbvecs.write('\n')
for z in z_bvecs:
    fbvecs.write(str(z)+ ' ')
fbvecs.write('\n')
fbvecs.close()










# # #for debugging isoSignalFromDti
# # diff=diffusion.diffVolume('/home/u2hussai/scratch/tempdata/90')
# # dti=diffusion.dti('/home/u2hussai/scratch/tempdata/90/dti','/home/u2hussai/scratch/tempdata/90/nodif_brain_mask.nii.gz')
# H=5
# # ico = icosahedron.icomesh(m=H - 1)
# # Y=dti.icoSignalFromDti(ico)


# X_nii=nib.load('/home/u2hussai/scratch/tempdata/6/data.nii.gz')
# #Y_nii=nib.load('./data/90/data_cut_flat.nii.gz')

# # #get patches from matrices of coordintes
# # x = np.arange(0,X_nii.shape[0])
# # y = np.arange(0,X_nii.shape[1])
# # z = np.arange(0,X_nii.shape[2])
# # x,y,z=np.meshgrid(x,y,z,indexing='ij')
# # x=torch.from_numpy(x)
# # y=torch.from_numpy(y)
# # z=torch.from_numpy(z)

# # Nc=16 #patch size
# # #unfold the coordinates
# # x=x.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
# # y=y.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
# # z=z.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)

# # #do same for mask
# # mask = nib.load('/home/u2hussai/scratch/tempdata/6/nodif_brain_mask.nii.gz')
# # mask= mask.get_fdata()
# # mask = torch.from_numpy(mask)
# # mask = mask.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)

# # #flatten the corse direction
# # coarse_sz=mask.shape[0:3]
# # mask = mask.reshape(((-1,) + tuple(mask.shape[-3:])))
# # x = x.reshape(((-1,) + tuple(x.shape[-3:])))
# # y = y.reshape(((-1,) + tuple(y.shape[-3:])))
# # z = z.reshape(((-1,) + tuple(z.shape[-3:])))

# # #get occupancy from mask
# # mask = mask.reshape(mask.shape[0],-1)
# # mask = mask.mean(dim=-1)
# # occupation_theshold=0.8
# # occ_inds= np.where(mask>0.99)[0]

# # #maybe get occupancy from FA
# # FA = nib.load('/home/u2hussai/scratch/tempdata/90/dti_FA.nii.gz')
# # FA = FA.get_fdata()
# # FA = torch.from_numpy(FA)
# # FA = FA.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)

# # #flatten the corse direction
# # coarse_sz=FA.shape[0:3]
# # FA = FA.reshape(((-1,) + tuple(FA.shape[-3:])))


# # #get occupancy from mask
# # FA = FA.reshape(FA.shape[0],-1)
# # FA = FA.mean(dim=-1)
# # occupation_theshold=0.8
# # occ_inds= np.where(FA>0.2)[0]


# # N=20#len(occ_inds)
# # oi=np.random.randint(0,len(occ_inds),N)
# # x=x[occ_inds[oi]]
# # y=y[occ_inds[oi]]
# # z=z[occ_inds[oi]]

# # #extract actual data now
# # #X=X_nii.get_fdata()[x,y,z] this runs out of memory
# # #Y=Y_nii.get_fdata()[x,y,z] this runs out of memory

# # #approch is to generate flat data in the spot
# h=H+1
# w=5*h
# # diff6 = diffusion.diffVolume('/home/u2hussai/scratch/tempdata/6')
# # #diff90 = diffusion.diffVolume('./data/90')
# # ico = icosahedron.icomesh(m=H - 1)
# # I, J, T = d12.padding_basis(H=H)

# # xp= x.reshape(-1).numpy()
# # yp= y.reshape(-1).numpy()
# # zp= z.reshape(-1).numpy()

# # voxels=np.asarray([xp,yp,zp]).T
# # diff6.makeInverseDistInterpMatrix(ico.interpolation_mesh)
# # S0X, X = diff6.makeFlat(voxels,ico)
# # X = X[:,:,I[0,:,:],J[0,:,:]]
# # #X = X.reshape([-1,h*w])/S0X
# # shp=tuple(x.shape) + (h,w)
# # X = X.reshape(shp)

# # # diff90.makeInverseDistInterpMatrix(ico.interpolation_mesh)
# # # S0Y, Y = diff90.makeFlat(voxels,ico)
# # # S0Y = S0Y.mean(-1)
# # # S0Y = S0Y.reshape([len(S0Y),1])
# # # Y = Y[:,:,I[0,:,:],J[0,:,:]]
# # # #Y = Y.reshape([-1,h*w])/S0Y
# # # Y=Y.reshape(shp)

# # Y = Y[xp,yp,zp]
# # Y=Y.reshape(shp)

# # #make training data from dti
# # #dti = diffusion.dti('./data/90/dti')


# # X[np.isnan(X)]=0
# # X[np.isinf(X)]=0
# # Y[np.isnan(X)]=0
# # Y[np.isinf(X)]=0

# # X[np.isnan(Y)]=0
# # X[np.isinf(Y)]=0
# # Y[np.isnan(Y)]=0
# # Y[np.isinf(Y)]=0


# # X = (X - X.mean())/X.std()
# # X = torch.from_numpy(X).contiguous().float()

# # Y = (Y - Y.mean())/Y.std()
# # Y = torch.from_numpy(Y).contiguous().float()


# # #this is for 3d inputs but takes convolutions on the internal space
# # class gnet3d(Module):
# #     def __init__(self,H,filterlist,shells=1,activationlist=None):
# #         super(gnet3d,self).__init__()
# #         self.filterlist = filterlist
# #         self.gconvs=gNetFromList(H,filterlist,shells,activationlist= activationlist)
# #         self.pool = opool(filterlist[-1])

# #     def forward(self,x):
# #         #x will have shape [batch,Nc,Nc,Nc,h,w]
# #         B=x.shape[0]
# #         Nc = x.shape[1]
# #         h = x.shape[-2]
# #         w = x.shape[-1]
# #         x = x.view((B*Nc*Nc*Nc,1,h,w))
# #         x = self.gconvs(x)
# #         x = self.pool(x) # shape [batch,Nc^3,filterlist[-1],h,w
# #         #x = x.view([B,Nc,Nc,Nc,self.filterlist[-1],h,w])
# #         x = x.view([B,Nc,Nc,Nc,h,w])
# #         return x

# # #this is a class to make lists out of
# # class conv3d(Module):
# #     """
# #     This class combines conv3d and batch norm layers and applies a provided activation
# #     """
# #     def __init__(self,Cin,Cout,activation=None):
# #         super(conv3d,self).__init__()
# #         self.activation= activation
# #         self.conv = Conv3d(Cin,Cout,3,padding=1)

# #     def forward(self,x):
# #         if self.activation!=None:
# #             x=self.activation(self.conv(x))
# #         else:
# #             x=self.conv(x)
# #         return x

# # #this makes the list for 3d convs
# # class conv3dList(Module):
# #     def __init__(self,filterlist,activationlist=None):
# #         super(conv3dList,self).__init__()
# #         self.conv3ds=[]
# #         self.filterlist = filterlist
# #         if activationlist is None:
# #             self.activationlist = [None for i in range(0,len(filterlist)-1)]
# #         for i in range(0,len(filterlist)-1):
# #             if i==0:
# #                 self.conv3ds=[conv3d(filterlist[i],filterlist[i+1],activationlist[i])]
# #             else:
# #                 self.conv3ds.append(conv3d(filterlist[i],filterlist[i+1],activationlist[i]))
# #         self.conv3ds = ModuleList(self.conv3ds)

# #     def forward(self,x):
# #         H=x.shape[-2]-1
# #         h = H + 1
# #         w = 5 * (H + 1)
# #         Nc = x.shape[1]
# #         B = x.shape[0]
# #         C=1#x.shape[-3]
# #         for i,conv in enumerate(self.conv3ds):
# #             if i==0:
# #                 #input size is [B,Nc,Nc,Nc,C,h,w]
# #                 x = x.view([B,Nc,Nc,Nc,C*h*w]).moveaxis(-1,1)
# #                 x = conv(x)
# #             elif i==len(self.conv3ds)-1:
# #                 x=conv(x)
# #                 C=self.filterlist[-1]
# #                 x = x.moveaxis(1,-1)
# #                 x = x.view([B,Nc,Nc,Nc,h,w])
# #             else:
# #                 x = conv(x)
# #         return x

# # #the actual network
# # class Net(Module):
# #     def __init__(self,H,shells=1):
# #         super(Net,self).__init__()
# #         self.H=H
# #         #self.shp = shp
# #         self.h = H+1
# #         self.w = 5*(H+1)
# #         #self.Nc = shp[1]
# #         #self.Nc3 = shp[1]*shp[1]*shp[1]
# #         self.gConvs= gnet3d(H,[1,4,4,4,4,4,1],shells,[F.relu,F.relu,F.relu,F.relu,F.relu,None])
# #         self.conv3ds=conv3dList([1*self.h*self.w,4,4,4,4,1*self.h*self.w],[F.relu,F.relu,F.relu,F.relu,None])

# #     def forward(self,x):
# #         x=self.conv3ds(x)
# #         x=self.gConvs(x)
# #         return x


# # net = Net(H).cuda()
# # #out=net(X)

# # net = Net(H).float().cuda()

# # #net.load_state_dict(torch.load('./net'))

# # criterion = nn.MSELoss()
# # #criterion=nn.SmoothL1Loss()
# # #criterion=nn.CosineSimilarity()
# # #criterion=Myloss
# # #
# # #
# # optimizer = optim.Adamax(net.parameters(), lr=1e-1)#, weight_decay=0.001)
# # optimizer.zero_grad()
# # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
# # #
# # running_loss = 0

# # train = torch.utils.data.TensorDataset(X, Y-X)
# # trainloader = DataLoader(train, batch_size=1)

# # for epoch in range(0, 10):
# #     print(epoch)
# #     for n, (inputs, target) in enumerate(trainloader, 0):
# #         # print(n)

# #         optimizer.zero_grad()

# #         #print(inputs.shape)
# #         output = net(inputs.cuda())

# #         loss = criterion(output, target.cuda())
# #         loss=loss.sum()
# #         print(loss)
# #         loss.backward()
# #         #print(net.lin3d.weight[2,2,4,4,4,3])
# #         #print(net.conv1.weight)
# #         optimizer.step()
# #         running_loss += loss.item()
# #     else:
# #         print(running_loss / len(trainloader))
# #     # if i%N_train==0:
# #     #    print('[%d, %5d] loss: %.3f' %
# #     #          ( 1, i + 1, running_loss / 100))
# #     scheduler.step(running_loss)
# #     running_loss = 0.0

# # torch.save(net.state_dict(),'./net')

# ##predict
# #how do we do this quickly
# #get coordinates over whole image
# #flatten the corse direction
# #get patches from matrices of coordintes
# x = np.arange(0,X_nii.shape[0])
# y = np.arange(0,X_nii.shape[1])
# z = np.arange(0,X_nii.shape[2])
# x,y,z=np.meshgrid(x,y,z,indexing='ij')
# x=torch.from_numpy(x)
# y=torch.from_numpy(y)
# z=torch.from_numpy(z)

# Nc=16 #patch size
# #unfold the coordinates
# x=x.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
# y=y.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
# z=z.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)

# #do same for mask
# mask = nib.load('/home/u2hussai/scratch/tempdata/6/nodif_brain_mask.nii.gz')
# mask= mask.get_fdata()
# mask = torch.from_numpy(mask)
# mask = mask.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
# coarse_sz=mask.shape[0:3]
# mask = mask.reshape(((-1,) + tuple(mask.shape[-3:])))
# x = x.reshape(((-1,) + tuple(x.shape[-3:])))
# y = y.reshape(((-1,) + tuple(y.shape[-3:])))
# z = z.reshape(((-1,) + tuple(z.shape[-3:])))

# #get occupancy from mask
# mask = mask.reshape(mask.shape[0],-1)
# mask = mask.mean(dim=-1)
# occupation_theshold=0.5
# occ_inds= np.where(mask>0)[0]

# x=x[occ_inds]
# y=y[occ_inds]
# z=z[occ_inds]

# # #extract actual data now
# # #X=X_nii.get_fdata()[x,y,z] this runs out of memory
# # #Y=Y_nii.get_fdata()[x,y,z] this runs out of memory

# # #approch is to generate flat data in the spot
# # #H=9
# # diff6 = diffusion.diffVolume('/home/u2hussai/scratch/tempdata/6')
# diff90 = diffusion.diffVolume('/home/u2hussai/scratch/tempdata/90')
# # ico = icosahedron.icomesh(m=H - 1)
# # I, J, T = d12.padding_basis(H=H)

# xp= x.reshape(-1).numpy()
# yp= y.reshape(-1).numpy()
# zp= z.reshape(-1).numpy()

# # voxels=np.asarray([xp,yp,zp]).T
# # diff6.makeInverseDistInterpMatrix(ico.interpolation_mesh)
# # S0X, X = diff6.makeFlat(voxels,ico)

# # X = X[:,:,I[0,:,:],J[0,:,:]]
# # #X = X.reshape([-1,h*w])/S0X
# # shp=tuple(x.shape) + (h,w)
# # X = X.reshape(shp)

# # X[np.isnan(X)]=0
# # X[np.isinf(X)]=0

# # mean= X.mean()
# # std=X.std()
# # X = (X - X.mean())/X.std()
# # X = torch.from_numpy(X).contiguous().float()

# # pred=torch.zeros_like(X)

# # batch_size=1
# # for i in range(0,X.shape[0],batch_size):
# #     print(i)
# #     pred[i:i+batch_size]=(net(X[i:i+batch_size].cuda()).cpu() + X[i:i+batch_size]).detach()

# #out = np.zeros(diff6.vol.shape[0:3]+(h,w))

# out = np.zeros(X_nii.shape[0:3]+(h,w))

# # oldshp=pred.shape
# # pred = pred.view(-1,h,w)
# # pred = pred*std + mean

# # #out[xp,yp,zp]=pred

# #for debugging isoSignalFromDti
# diff=diffusion.diffVolume('/home/u2hussai/scratch/tempdata/90')
# dti=diffusion.dti('/home/u2hussai/scratch/tempdata/90/dti','/home/u2hussai/scratch/tempdata/90/nodif_brain_mask.nii.gz')
# ico = icosahedron.icomesh(m=H - 1)
# Y=dti.icoSignalFromDti(ico)


# out[xp,yp,zp] = Y[xp,yp,zp]
# ############## try to make a nifi from out #######################


# #diff = diff90 i think
# w=5*(H+1)
# h=H+1
# ico.grid2xyz()

# #how many unique directions are there? (!)

# basis=np.zeros([h,w])
# for c in range(0,5):
#     basis[1:H,c*h+1:(c+1)*h-1]=1

# N=len(basis[basis==1])+1
# print('Number of bdirs is: ', N)

# N_random=w
# rng=np.random.default_rng()
# inds=rng.choice(N-2,size=N_random,replace=False)+1
# inds[0]=0

# bvals=np.zeros(N_random)
# x_bvecs=np.zeros(N_random)
# y_bvecs=np.zeros(N_random)
# z_bvecs=np.zeros(N_random)

# x_bvecs[1:]=ico.X_in_grid[basis==1].flatten()[inds[1:]]
# y_bvecs[1:]=ico.Y_in_grid[basis==1].flatten()[inds[1:]]
# z_bvecs[1:]=ico.Z_in_grid[basis==1].flatten()[inds[1:]]

# bvals[1:]=1000



# #sz=diff90.vol.get_fdata().shape
# sz=out.shape
# diff_out=np.zeros([sz[0],sz[1],sz[2],N_random])

# diff_out[:,:,:,0]=diff90.vol.get_fdata()[:,:,:,diff90.inds[0]].mean(-1)
# i, j, k = np.where(diff90.mask.get_fdata() == 1)
# for p in range(0,len(i)):
#     #signal=self.Ypredict[p,0,:,:] #it is assumed that post processing has been done on this and it is the final signal
#     signal =out[i[p],j[p],k[p]]
#     signal = signal[basis==1].flatten()
#     # print(diff_out.shape)
#     # print(signal.shape)
#     diff_out[i[p],j[p],k[p],1:] = signal[inds[1:]]

# # for i in range(1,diff_out.shape[-1]):
# #     diff_out[:,:,:,i] = diff_out[:,:,:,i]*diff_out[:,:,:,0]

# diff_out=nib.Nifti1Image(diff_out,diff90.vol.affine)
# nib.save(diff_out,'./data_network.nii.gz')

# #write the bvecs and bvals
# fbval = open('./bvals_network', "w")
# for bval in bvals:
#     fbval.write(str(bval)+" ")
# fbval.close()

# fbvecs = open('./bvecs_network',"w")
# for x in x_bvecs:
#     fbvecs.write(str(x)+ ' ')
# fbvecs.write('\n')
# for y in y_bvecs:
#     fbvecs.write(str(y)+ ' ')
# fbvecs.write('\n')
# for z in z_bvecs:
#     fbvecs.write(str(z)+ ' ')
# fbvecs.write('\n')
# fbvecs.close()

# # import icosahedron
# # import diffusion
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import dihedral12 as d12
# # import torch
# # import residualPrediction
# #
# # respred=residualPrediction.resPredictor('./data/6',
# #                                         './databvec-dirs-6_type-residual_Ntrain-500_Nepochs-200_patience'
# #                                         '-20_factor-0.65_lr-0.01_batch_size-16_interp-inverse_distance_glayers-27-8-8'
# #                                         '-8-8-8-8-27_gactivation0-relu_residual',11)
# #
# #
# # # def loader_3d(diffpath,H):
# # #     diff=diffusion.diffVolume(diffpath)
# # #     diff.cut_ik()
# # #     ico = icosahedron.icomesh(m=H - 1)
# # #     diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)
# # #     I, J, T = d12.padding_basis(H=H)
# # #     i, j, k = np.where(diff.mask.get_fdata() > 0)
# # #     voxels = np.asarray([i, j, k]).T
# # #
# # #     S0,out=diff.makeFlat(voxels,ico)
# # #     out= out[:,:,I[0,:,:],J[0,:,:]]
# # #
# # #     S0_flat=np.zeros((diff.mask.get_fdata().shape[0:3]+(S0.shape[-1],)))
# # #     S0_flat[i,j,k]=S0
# # #     S0_flat=torch.from_numpy(S0_flat).contiguous()
# # #
# # #     diff_flat=np.zeros((diff.mask.get_fdata().shape[0:3]+out.shape[1:]))
# # #     diff_flat[i,j,k,:,:,:]=out
# # #     diff_flat=torch.from_numpy(diff_flat).contiguous()
# # #
# # #     return S0_flat.unfold(0,3,3).unfold(1,3,3).unfold(2,3,3), diff_flat.unfold(0,3,3).unfold(1,3,3).unfold(2,3,3)
# # #
# # # S06, diff6=loader_3d('./data/6',11)
# #
# #
# # # fig,ax=plt.subplots(3,9)
# # # c=0
# # # for r in range(0,3):
# # #     for c in range(0,9):
# # #         test=diff_flat[20,20,20,0,:,:].reshape([12,60,-1])
# # #         ax[r,c].imshow(test[:,:,c])
# # #         c+=c
# #
# # # #import icosahedron
# # #
# # # #ico=icosahedron.icomesh(m=4)
# # # #ico.get_icomesh()
# # # #ico.vertices_to_matrix()
# # # #ico.grid2xyz()
# # #
# # # import residualPrediction
# # #
# # # inputpath='/home/u2hussai/scratch/dtitraining/prediction/sub-518746/6/'
# # # netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-residual_Ntrain-100000_Nepochs-200_patience-20_factor-0.65_lr-0.01_batch_size-16_interp-inverse_distance_glayers-1-16-16-16-16-16-16-1_gactivation0-relu_residual'
# # # #
# # # redPred=residualPrediction.resPredictor(inputpath,netpath)
# # # #
# # # redPred.predict()
# # # redPred.makeNifti(inputpath,11)
# #
# # # import nifti2traintest
# # # import matplotlib.pyplot as plt
# #
# #
# # # downpath='./data/6/'
# # # uppath='./data/90/'
# #
# # # Sd,X,Sup,Y= nifti2traintest.loadDownUp(downpath,uppath,20)
# #
# # # i=15
# # # fig,ax=plt.subplots(2)
# # # ax[0].imshow(X[i,0])
# # # ax[1].imshow(Y[i,0])
# #
# # # # import icosahedron
# # # # #from mayavi import mlab
# # # # import stripy
# # # # import diffusion
# # # # from joblib import Parallel, delayed
# # # # import numpy as np
# # # # import time
# # # # import gPyTorch
# # # # import torch
# # # # import extract3dDiffusion
# # # # import os
# # # # import matplotlib.pyplot as plt
# # # # from torch.nn.modules.module import Module
# # # # from torch.nn import Linear
# # # # from torch.nn import functional as F
# # # # from torch.nn import ELU
# # # # from torch.optim.lr_scheduler import ReduceLROnPlateau
# # # # from torch.utils.data import DataLoader
# # # # import torch.optim as optim
# # # # from torch.nn import Conv3d
# # # # import dihedral12 as d12
# # # # import torch.nn as nn
# # # #
# # # # from gPyTorch import (gConv5dFromList,opool5d, maxpool5d, lNet5dFromList)
# # # #
# # # #
# # # # class Net(Module):
# # # #     def __init__(self):
# # # #         super(Net,self).__init__()
# # # #         self.gconv=gConv5dFromList(11,[1,4,8],shells=1,activationlist=[ELU(),ELU(),ELU(),ELU()])
# # # #         self.opool=opool5d(8)
# # # #         self.mxpool=maxpool5d([2,2])
# # # #         self.lin1=lNet5dFromList([int(8*12*60/4),100,90,80,50,40],activationlist=[ELU(),ELU(),ELU(),ELU(),ELU()])
# # # #         self.conv3d1 = Conv3d(40, 8, [3, 3, 3], padding=[1, 1, 1])
# # # #         self.conv3d2 = Conv3d( 8,8, [3, 3, 3], padding=[1, 1, 1])
# # # #         self.conv3d3 = Conv3d( 8, 4, [3, 3, 3], padding=[1, 1, 1])
# # # #         self.conv3d4 = Conv3d(4, 3, [3, 3, 3], padding=[1, 1, 1])
# # # #
# # # #
# # # #     def forward(self,x):
# # # #         x=self.gconv(x)
# # # #         x=self.opool(x)
# # # #         x=self.mxpool(x)
# # # #         x=self.lin1(x)
# # # #         x=self.conv3d1(x)
# # # #         x=self.conv3d2(x)
# # # #         x=self.conv3d3(x)
# # # #         x=self.conv3d4(x)
# # # #         return(x)
# # # #
# # # #
# # # #
# # # #
# # # #
# # # # # datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"
# # # # # dtipath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/dti"
# # # # # outpath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/"
# # # #
# # # #
# # # # # datapath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # # # dtipath="./data/sub-100206/dtifit"
# # # # # outpath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # #
# # # # datapath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # # dtipath="./data/sub-100206/dtifit"
# # # # outpath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # #
# # # #
# # # # ext=extract3dDiffusion.extractor3d(datapath,dtipath,outpath)
# # # # ext.splitNsave(9)
# # # #
# # # # # I,J,T=d12.padding_basis(11)
# # # # #
# # # # # chnk=extract3dDiffusion.chunk_loader(outpath)
# # # # # X,Y=chnk.load(cut=100)
# # # # # X=X.reshape((X.shape[0],1) + tuple(X.shape[1:]))
# # # # #
# # # # # X=X[:,:,:,:,:,I[0,:,:],J[0,:,:]]
# # # # #
# # # # #
# # # # def plotter(X1,X2,i):
# # # #     fig,ax=plt.subplots(2,1)
# # # #     ax[0].imshow(1/X1[i, 0, 2, 2, 2, :, :])
# # # #     ax[1].imshow(1/X2[i, 0, 2, 2, 2, :, :])
# # # # #
# # # # #
# # # # #
# # # # #
# # # # # Y=Y[:,:,:,:,4:7]
# # # # #
# # # # #
# # # # #
# # # # # inputs= np.moveaxis(X,1,-3)
# # # # # inputs= torch.from_numpy(inputs[103:104]).contiguous().cuda().float()
# # # # #
# # # # # targets=np.moveaxis(Y,-1,1)
# # # # # targets=torch.from_numpy(targets[103:104]).contiguous().cuda().float()
# # # # #
# # # # # def Myloss(output,target):
# # # # #     x=output
# # # # #     y=target
# # # # #     sz=output.shape
# # # # #     loss_all=torch.zeros([sz[0],sz[-3]*sz[-2]*sz[-1]]).cuda()
# # # # #     l=0
# # # # #     for i in range(0,output.shape[-3]):
# # # # #         for j in range(0, output.shape[-2]):
# # # # #             for k in range(0, output.shape[-1]):
# # # # #                 x=output[:,:,i,j,k].cuda()
# # # # #                 y=target[:,4:7,i,j,k].cuda()
# # # # #                 FA=target[:,0,i,j,k].cuda().detach()
# # # # #                 #FA[torch.isnan(FA)]=0
# # # # #                 #norm=x.norm(dim=-1)
# # # # #                 #norm=norm.view(-1,1)
# # # # #                 #norm=norm.expand(norm.shape[0],3)
# # # # #                 #if norm >0:
# # # # #                 #print(norm)
# # # # #                 #print(x)
# # # # #                 x=F.normalize(x)
# # # # #                 loss=x-y
# # # # #                 loss=loss.sum(dim=-1).abs()
# # # # #                 #print(loss)
# # # # #                 #eps = 1e-6
# # # # #                 #loss[(loss - 1).abs() < eps] = 1.0
# # # # #                 #loss_all[:,l]=torch.arccos(loss)*(1-FA)
# # # # #                 loss_all[:, l]=loss
# # # # #                 l+=1
# # # # #     return loss_all.flatten().mean()
# # # # #
# # # # # net=Net().cuda()
# # # # #
# # # # #
# # # # # criterion = nn.MSELoss()
# # # # # #criterion=nn.SmoothL1Loss()
# # # # # #criterion=nn.CosineSimilarity()
# # # # # #criterion=Myloss
# # # # # #
# # # # # #
# # # # # optimizer = optim.Adamax(net.parameters(), lr=1e-3)#, weight_decay=0.001)
# # # # # optimizer.zero_grad()
# # # # # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
# # # # # #
# # # # # running_loss = 0
# # # # #
# # # # # train = torch.utils.data.TensorDataset(inputs, targets)
# # # # # trainloader = DataLoader(train, batch_size=2)
# # # # # #
# # # # # for epoch in range(0, 40):
# # # # #     print(epoch)
# # # # #     for n, (inputs, target) in enumerate(trainloader, 0):
# # # # #         # print(n)
# # # # #
# # # # #         optimizer.zero_grad()
# # # # #
# # # # #         #print(inputs.shape)
# # # # #         output = net(inputs.cuda())
# # # # #
# # # # #         loss = criterion(output, target)
# # # # #         loss=loss.sum()
# # # # #         print(loss)
# # # # #         loss.backward()
# # # # #         #print(net.lin3d.weight[2,2,4,4,4,3])
# # # # #         #print(net.conv1.weight)
# # # # #         optimizer.step()
# # # # #         running_loss += loss.item()
# # # # #     else:
# # # # #         print(running_loss / len(trainloader))
# # # # #     # if i%N_train==0:
# # # # #     #    print('[%d, %5d] loss: %.3f' %
# # # # #     #          ( 1, i + 1, running_loss / 100))
# # # # #     scheduler.step(running_loss)
# # # # #     running_loss = 0.0
# # # # # #
# # # # # #
# # # # #
# # # # #
# # # # # #use network to make prediction and put volume back together
# # # # # datapath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # # # dtipath="./data/sub-100206/dtifit"
# # # # # outpath="/home/uzair/PycharmProjects/dgcnn/data/6/"
# # # # #
# # # #
# # # # #ext=extract3dDiffusion.extractor3d(datapath,dtipath,outpath)
# # # # #ext.splitNsave(9)
# # # #
# # # # # chnk=extract3dDiffusion.chunk_loader(outpath)
# # # # # X,Y=chnk.load(cut=0)
# # # # # X=1-X.reshape((X.shape[0],1) + tuple(X.shape[1:]))
# # # # #
# # # # # inputs= np.moveaxis(X,1,-3)
# # # # # #inputs= torch.from_numpy(inputs).contiguous().cuda()
# # # # #
# # # # # net=torch.load('net')
# # # # #
# # # # # batch_size=2
# # # # # outp=np.zeros([len(inputs),3,9,9,9])
# # # # # for i in range(0,len(inputs),batch_size):
# # # # #     print(i)
# # # # #     thisinput=torch.from_numpy(inputs[i:i+batch_size]).contiguous().cuda()
# # # # #     outp[i:i+batch_size,:,:,:,:]=net(thisinput).detach().cpu()
# # # # #     #test = net(thisinput).detach().cpu()
# # # # #
# # # # # #normalize the outs
# # # # # diff=[]
# # # # # FA=[]
# # # # # for i in range(0,len(outp)):
# # # # #     for a in range(0,9):
# # # # #         for b in range(0,9):
# # # # #             for c in range(0,9):
# # # # #                 if Y[i,a,b,c,0]>0.1:
# # # # #                     vec1=outp[i,:,a,b,c]
# # # # #                     vec2=Y[i,a,b,c,4:7]
# # # # #                     vec1=vec1/np.sqrt((vec1*vec1).sum())
# # # # #                     diff.append(np.rad2deg( np.arccos(np.abs( (vec1*vec2).sum()) )))
# # # # #                     FA.append(Y[i,a,b,c,0])
# # # # #
# # # # # diff=np.asarray(diff)
# # # # # FA=np.asarray(FA)
# # # #
# # # # #def zeropadder(input):
# # # # #     sz=input.shape
# # # # #     if len(sz) ==7:
# # # # #         out = np.zeros([sz[0], sz[1], sz[2] + 2, sz[3] + 2, sz[4] + 2] + list(sz[5:]))
# # # # #         out[:,:,1:-1,1:-1,1:-1,:,:]=input
# # # # #         return out
# # # # #
# # # # #     # if len(sz) == 5:
# # # # #     #     out = np.zeros([sz[0], sz[1], sz[2], sz[3]] + list(sz[4:]))
# # # # #     #     out[:, 1:-1, 1:-1, 1:-1, :] = input
# # # # #     #     return out
# # # # #
# # # # #
# # # # # X=zeropadder(X)
# # # # # #Y=zeropadder(Y)
# # # # #
# # # # # X=torch.from_numpy(X[0:2])
# # # # # X[np.isnan(X)==1]=0
# # # # # Y=torch.from_numpy(Y[0:2])
# # # # # #Y=Y[:,:,:,:,:,4:7]
# # # # # X=X.cuda()
# # # # # Y=Y.cuda()
# # # # #
# # # # # H=5
# # # # # h= 5 * (H + 1)
# # # # # w=H + 1
# # # # # last=4
# # # # # class Net(Module):
# # # # #     def __init__(self):
# # # # #         super(Net, self).__init__()
# # # # #         self.conv1 = gPyTorch.gConv3d(1, 4, H, shells=1)
# # # # #         self.conv2 = gPyTorch.gConv3d(4, 4, H)
# # # # #         #self.conv3 = gPyTorch.gConv3d(1, 1, H)
# # # # #         self.opool = gPyTorch.opool3d(last)
# # # # #         self.lin3d = gPyTorch.linear3d(last,3,9,9,9,6,30)
# # # # #
# # # # #     def forward(self, x):
# # # # #         x = F.relu(self.conv1(x))
# # # # #         x = F.relu(self.conv2(x))
# # # # #         #x = F.relu(self.conv3(x))
# # # # #         x = self.opool(x)
# # # # #         x = self.lin3d(x)
# # # # #
# # # # #         return x
# # # # #
# # # # # net=Net().cuda()
# # # # #
# # # # # out=net(X)
# # # # #
# # # # # def Myloss(output,target):
# # # # #     x=output
# # # # #     y=target
# # # # #     sz=output.shape
# # # # #     loss_all=torch.zeros([sz[0],sz[1]*sz[2]*sz[3]]).cuda()
# # # # #     l=0
# # # # #     for i in range(0,output.shape[1]):
# # # # #         for j in range(0, output.shape[2]):
# # # # #             for k in range(0, output.shape[3]):
# # # # #                 x=output[:,i,j,k,:].cuda()
# # # # #                 y=target[:,i,j,k,:].cuda()
# # # # #                 #norm=x.norm(dim=-1)
# # # # #                 #norm=norm.view(-1,1)
# # # # #                 #norm=norm.expand(norm.shape[0],3)
# # # # #                 #if norm >0:
# # # # #                 #print(norm)
# # # # #                 x=F.normalize(x)
# # # # #                 loss=x*y
# # # # #                 loss=loss.sum(dim=-1).abs()
# # # # #                 #print(loss)
# # # # #                 eps = 1e-6
# # # # #                 loss[(loss - 1).abs() < eps] = 1.0
# # # # #                 loss_all[:,l]=torch.arccos(loss)
# # # # #                 #print(output)
# # # # #                 l+=1
# # # # #     return loss_all.flatten().mean()
# # # # #
# # # #
# # # #
# # # #
# # # #
# # # # # datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
# # # # # dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"
# # # # #
# # # # # diff=diffusion.diffVolume()
# # # # # diff.getVolume(datapath)
# # # # # diff.shells()
# # # # # diff.makeBvecMeshes()
# # # # #
# # # # # ico=icosahedron.icomesh()
# # # # # ico.get_icomesh()
# # # # # ico.vertices_to_matrix()
# # # # # ico.getSixDirections()
# # # # #
# # # # # i, j, k = np.where(diff.mask.get_fdata() == 1)
# # # # # voxels = np.asarray([i, j, k]).T
# # # # #
# # # # # #compute time before/after "initialization"
# # # # # start=time.time()
# # # # # diffdown=diffusion.diffDownsample(diff,ico)
# # # # # test=diffdown.downSampleFromList(voxels[0:10000])
# # # # # end=time.time()
# # # # # print(end-start)
# # # # #
# # # # # start=time.time()
# # # # # diffdown=diffusion.diffDownsample(diff,ico)
# # # # # test=diffdown.downSampleFromList([voxels[0]])
# # # # # test=diffdown.downSampleFromList(voxels[1:10000])
# # # # # end=time.time()
# # # # # print(end-start)
# # # #
# # # # #downsample
# # # # #for each subject,bvec create 10000 Xtrain
# # # # #combine these and train for each bvec
# # # # #test with completely unseen subject
# # # #
# # # #
# # # # #xyz=ico.getSixDirections()
# # # # #
# # # # #x=[]
# # # # #y=[]
# # # # #z=[]
# # # # #for vec in ico.vertices:
# # # # #    x.append(vec[0])
# # # # #    y.append(vec[1])
# # # # #    z.append(vec[2])
# # # # #
# # # # #mlab.points3d(x,y,z)
# # # # #mlab.points3d(xyz[:,0],xyz[:,1],xyz[:,2],color=(1,0,0),scale_factor=0.23)
# # # # #mlab.show()
