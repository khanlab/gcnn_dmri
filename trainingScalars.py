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
from torch.nn import Linear
from icosahedron import sphere_to_flat_basis

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
    path = path + '_3dlayers-'+array2str(modelParams['filterlist3d'])
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

class to_icosahedron(Module):
    """
    This module moves from N-directions to the 2d icosahedron
    """
    def __init__(self,ico_mesh,subject_interp_matrices):
        #subject_interp_matrices is the subject specific interpolation matrix [sub_id,...]
        super(to_icosahedron,self).__init__()
        self.icomesh=ico_mesh
        self.subject_interp_matrices=subject_interp_matrices

    def forward(self,x,sub_id):
        #x will have shape [batch,C,Nc,Nc,Nc]
        B=x.shape[0]
        C=x.shape[1]
        N=x.shape[2]
        x=x.moveaxis(1,-1)
        x=x.view([-1,C])
        
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
        x = self.pool(x) # output shape is [batch*Nc^3,filterlist[-1],h,w]
        return x

#this module takes in scalar and 3d input with shapes [B Nc^3,Cinp,Nscalar] and [B Nc^3,Cinp Nshell,h,w]
class gnet3dScalar(Module):
    def __init__(self,H,LCinp,LCout,filterlist,shells=None,activationlist=None): #shells should be same as filterlist[0]
        super(gnet3dScalar,self).__init__()

        self.LCinp = LCinp
        self.LCout = LCout
        self.filterlist = filterlist
        self.gconvs=gNetFromList(H,filterlist,self.LCinp*shells,activationlist= activationlist)
        self.pool = opool(filterlist[-1])

        self.lin = Linear(LCinp,LCout)


    def forward(self,x):
        #linear part
        x0 = x[0] #assuming x0 has shape [BNc^3,Cinp,Nscalar]
        x0 = x0.moveaxis(1,-1) #shape is [BNc^3,Nscalar, Cin]
        x0 = self.lin(x0)

        #gConv2d part
        x = x[1]
        x = self.gconvs(x) #usual g conv
        x = self.pool(x)
        return x0,x

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

class threeTotwo(Module):
    def __init__(self,Nshells,Nscalar,Ndir,ico,I,J):
        super(threeTotwo,self).__init__()
        self.ico = ico
        self.Nshells = Nshells
        self.Nscalar = Nscalar
        self.Ndir = Ndir
        self.I = I
        self.J = J


    def forward(self,x,w):
        #x will have have shape [B,Cin,Nc,Nc,Nc]
        B=x.shape[0]
        Cin = x.shape[1]
        Nc = x.shape[-1]
        Nscalar = self.Nscalar
        Ndir = self.Ndir
        Nshells = self.Nshells

        #we assume that x is flattened as [B,Cin=Cinp(Nscalar + NshellNdir),Nc,Nc,Nc]
        x = x.moveaxis(1,-1)#shape is now [B,Nc,Nc,Nc,Cin]
        x = x.view([-1,Cin])

        Cinp = int(Cin / (Nscalar + Nshells * Ndir))
        x = x.view([x.shape[0],Cinp,Nscalar + Nshells*Ndir])
        x_0 = x[:,:,0:Nscalar] #this is the scalar part
        x = x[:,:,Nscalar:] #this is the part to project to 2d has shape [x.shape[0],Cinp,Nshells*Ndir]
        x = x.view([x.shape[0],Cinp,Nshells,Ndir])
        #per shell matrix multiplication w should have shape [Nshells,N2d,2Ndir]
        N2d = w.shape[1]
        x_out = torch.empty([x.shape[0],Cinp,Nshells,N2d]).to(x.device.type).float()
        for m in range(0,Nshells):
            x_out[:,:,m,:] = torch.matmul(torch.cat([x[:,:,m,:],
                                                     x[:,:,m,:]],-1).reshape(x.shape[0],1,Cinp,2*Ndir),
                                                     w[m,:,:].to(x.device.type).T).view([x.shape[0],Cinp,N2d]).float()

            x_out[:,:,m,:] = 0.5*(x_out[:,:,m,:]+x_out[:,:,m,self.ico.antipodals])

        #move to icosahedron
        del x
        basis = sphere_to_flat_basis(self.ico)
        h = basis.shape[-2]
        w = basis.shape[-1]
        out = torch.empty([x_out.shape[0],Cinp,Nshells,h,w])

        i_nan, j_nan = np.where(np.isnan(basis))
        basis[i_nan, j_nan] = 0
        basis = basis.astype(int)

        out = x_out[:,:,:,basis]
        del x_out
        out = out[:,:,:,self.I[0,:,:],self.J[0,:,:]]
        out =out.view([out.shape[0],Cinp*Nshells,h,w])

        return x_0,out

class twoToThree(Module):
    def __init__(self,Nc,Nshell,Nscalar):
        super(twoToThree,self).__init__()
        self.Nc= Nc
        self.Nshell =Nshell
        self.Nscalar = Nscalar

    def forward(self,x):
        x0 = x[0]
        x = x[1]


        if (self.Nshell ==1 and self.Nscalar==1):
            B=int(x0.shape[0]/self.Nc**3)
            x0 = x0.view([B,self.Nc,self.Nc,self.Nc]) # only works with Nshell =1 and Nscalar=1
        else:
            x0 = x0.view([self.B, self.Nc, self.Nc, self.Nc,self.Nshell,self.Nscalar])  # only works with Nshell =1 and
            # Nscalar=1

        B = int(x.shape[0] / self.Nc ** 3)
        Cout = x.shape[1]
        h=x.shape[-2]
        w = x.shape[-1]

        if(Cout==1):
            x = x.view([B, self.Nc, self.Nc, self.Nc, h, w])
        else:
            x = x.view([B,self.Nc,self.Nc,self.Nc,Cout,h,w])

        return x0,x

class residualnet(Module):
    def __init__(self,filterlist3d,activationlist3d,filterlist2d,activationlist2d,H,shells,B,Nc,ico):
        super(residualnet,self).__init__()
        self.flist3d = filterlist3d
        self.alist3d = activationlist3d
        self.flist2d = filterlist2d
        self.alist2d = activationlist2d

        self.conv3ds = conv3dList(filterlist3d, activationlist3d)

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

class residualnetScalars(Module):
    def __init__(self, filterlist3d, activationlist3d, filterlist2d, activationlist2d, H, Nshells, Nc,I, J,Nscalar,Ndir,ico):
        super(residualnetScalars, self).__init__()
        # params
        self.Nshells = Nshells
        self.Nscalar = Nscalar
        self.Ndir = Ndir
        self.Cin = filterlist3d[-1]
        self.Cinp = int(self.Cin / (Nscalar + Nshells * Ndir))
        print('Cin is',self.Cin)
        print('Nscalar is',Nscalar)
        print('Nshells',Nshells)
        print('Ndir',Ndir)
        print('cINP IS', self.Cinp)
        self.ico = ico

        self.flist3d = filterlist3d
        self.alist3d = activationlist3d
        self.flist2d = filterlist2d
        self.alist2d = activationlist2d

        self.three2t = threeTotwo(Nshells,Nscalar,Ndir,ico,I,J)
        self.two2t = twoToThree(Nc,Nshells,Nscalar)

        self.conv3ds = conv3dList(filterlist3d, activationlist3d)
        self.gconvs = gnet3dScalar(H,self.Cinp,Nscalar, filterlist2d, Nshells, activationlist2d)

        self.conv3ds = DataParallel(self.conv3ds)
        self.gconvs = DataParallel(self.gconvs)

    def forward(self,x,w):
        x= self.conv3ds(x)
        x= self.three2t(x,w)
        x= self.gconvs(x)
        x= self.two2t(x)
        return x

class trainer:
    def __init__(self,modelParams,
                 Xtrain=None,Ytrain=None,S0Ytrain=None,interp_matrix_train=None,interp_matrix_ind_train=None,mask=None,
                 Xvalid=None,Yvalid=None,S0Yvalid=None,interp_matrix_valid=None,interp_matrix_ind_valid=None,maskvalid=None,
                 Nscalars=None,Ndir=None,ico=None,
                 B=None,Nc=None,Ncore=None,core=None,core_inv=None,I=None,J=None,zeros=None):
        """
        Class to create and train networks
        :param modelParams: A dict with all network parameters
        :param Xtrain: Cuda Xtrain data
        :param Ytrain: Cuda Ytrain data
        """
        self.modelParams=modelParams

        self.Xtrain=Xtrain
        self.S0Ytrain=S0Ytrain
        self.Ytrain=Ytrain
        self.w_train=interp_matrix_train
        self.w_ind_train = interp_matrix_ind_train


        self.Xvalid=Xvalid
        self.S0Yvalid = S0Yvalid
        self.Yvalid=Yvalid
        self.maskvalid = maskvalid
        self.w_valid = interp_matrix_valid
        self.w_ind_valid = interp_matrix_ind_valid

        self.net=[]
        self.B = B
        self.Nc = Nc
        self.Ncore = Ncore
        self.core = core
        self.core_inv = core_inv
        self.I = I
        self.J = J
        self.zeros = zeros
        self.mask = mask
        self.Nscalar=Nscalars
        self.Ndir=Ndir
        self.ico=ico

    def mul_by_mask(inputs,targets,mask):
        h = inputs.shape[-2]
        w = inputs.shape[-1]
        inputs = inputs.view(-1,h*w)
        targets = targets.view(-1,h*w)
        mask = mask.view(-1)
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
        if self.modelParams['misc']=='residual5dscalar':
            self.net = residualnetScalars (self.modelParams['filterlist3d'],
                                  self.modelParams['activationlist3d'],
                                  self.modelParams['gfilterlist'],
                                  self.modelParams['gactivationlist'],
                                  self.modelParams['H'],
                                  self.modelParams['shells'],
                                  self.Nc,
                                  self.I,
                                  self.J,
                                  self.Nscalar,
                                  self.Ndir,
                                  self.ico)
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
        train = torch.utils.data.TensorDataset(self.Xtrain, self.S0Ytrain,
                                               self.Ytrain, self.w_ind_train,
                                               self.mask)
        trainloader = DataLoader(train, batch_size=self.modelParams['batch_size'])
        
        running_loss_valid = 0
        valid = torch.utils.data.TensorDataset(self.Xvalid, self.S0Yvalid,
                                               self.Yvalid, self.w_ind_valid,
                                               self.maskvalid)

        validloader = DataLoader(valid,batch_size=self.modelParams['batch_size'])

        epochs_list = []
        loss_list = []
        loss_valid_list = []

        for epoch in range(0, self.modelParams['Nepochs']):
            print(epoch)
            for n, (X,S0Y,Y,w_ind,mask) in enumerate(trainloader, 0):
                optimizer.zero_grad()
                #if torch.cuda.is_available():
                torch.cuda.empty_cache()

                out_S0Y, out_Y = self.net(X.float().cuda(), self.w_train[w_ind.long(),:,:].float())
                out_S0Y = out_S0Y[mask==1].flatten()
                out_Y = out_Y[mask==1,:,:].flatten()
                S0Y = S0Y[mask==1].flatten()
                Y = Y[mask==1,:,:].flatten()

                outputs = torch.cat([out_S0Y,out_Y])
                targets = torch.cat([S0Y,Y])
                #outputs = torch.cat([out_Y])
                #targets = torch.cat([Y])
                loss = criterion(outputs.cuda(), targets.cuda())
                del outputs, targets, out_S0Y, out_Y
                torch.cuda.empty_cache()
                # else:
                #     torch.cuda.empty_cache()
                #     output = self.net(inputs)
                #     output = output[mask==1,:,:,:]
                #     targets = targets[mask==1,:,:,:]
                #     loss = criterion(output, targets)
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

                running_loss_valid = 0
                #compute validation loss
                for nn, (X_v,S0Y_v,Y_v,w_ind_v,mask_v) in enumerate(validloader, 0):
                    torch.cuda.empty_cache()
                    out_S0Y_v, out_Y_v = self.net(X_v.float().cuda(), self.w_valid[w_ind_v.long(),:,:].float())
                #     print('completed prediction and validation ind is',nn)

                    out_S0Y_v = out_S0Y_v.cpu()
                    out_Y_v = out_Y_v.cpu()
                    out_S0Y_v = out_S0Y_v[mask_v==1].flatten().cpu()
                    out_Y_v = out_Y_v[mask_v==1,:,:].flatten().cpu()
                    S0Y_v = S0Y_v[mask_v==1].flatten().cpu()
                    Y_v = Y_v[mask_v==1,:,:].flatten().cpu()
                    outputs = torch.cat([out_S0Y_v,out_Y_v]).detach().cpu()
                    targets = torch.cat([S0Y_v,Y_v]).detach().cpu()

                    loss_v = criterion(outputs, targets).cpu()
                    del outputs, targets, out_S0Y_v, out_Y_v
                    torch.cuda.empty_cache()
                    running_loss_valid += loss_v#.item()

                    
                else:
                    print('Validation loss',running_loss_valid / len(validloader))
                    loss_valid_list.append(running_loss_valid / len(validloader))

            scheduler.step(running_loss)
            running_loss = 0.0
            if (epoch % 1) == 0:
                fig_err, ax_err = plt.subplots()
                ax_err.plot(epochs_list, np.log10(loss_list))
                ax_err.plot(epochs_list, np.log10(loss_valid_list))
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




