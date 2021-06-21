import nibabel as nib
import diffusion
import icosahedron
import dihedral12 as d12
import torch
import numpy as np


class training_data:
    def __init__(self,inputpath,dtipath, maskpath, H, N_train, Nc=16):
        self.inputpath = inputpath
        self.dtipath = dtipath 
        self.maskpath = maskpath

        self.diff_input = diffusion.diffVolume(inputpath)
        self.dti=diffusion.dti(self.dtipath,self.maskpath)

        self.in_shp = self.diff_input.vol.shape
        self.N_patch= Nc #this is multiple of 144 and 176 (so have to pad HCP diffusion)
        self.H = H

        self.N_train=N_train

        self.make_coords()
        self.generate_training_data(N_train)

    def make_coords(self):
        #we make patches with meshgrids of coords and then extraxt data using those coords
        #we also make FA and mask list here, this is to extract patches based on FA or occupancy
        print("Making coords for patch extraction")
        
        self.x = np.arange(0,self.in_shp[0])
        self.y = np.arange(0,self.in_shp[1])
        self.z = np.arange(0,self.in_shp[2])
        
        self.x,self.y,self.z = np.meshgrid(self.x,self.y,self.z,indexing='ij')

        self.x=torch.from_numpy(self.x)
        self.y=torch.from_numpy(self.y)
        self.z=torch.from_numpy(self.z)

        Nc=self.N_patch

        self.x=self.x.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
        self.y=self.y.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
        self.z=self.z.unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)

        self.x = self.x.reshape((-1,)+tuple(self.x.shape[-3:]))
        self.y = self.y.reshape((-1,)+tuple(self.y.shape[-3:]))
        self.z = self.z.reshape((-1,)+tuple(self.z.shape[-3:]))

        mask = torch.from_numpy(self.dti.mask.get_fdata()).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
        FA = torch.from_numpy(self.dti.FA.get_fdata()).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
        #flatten patch labels (indices for each patch) and patch indices (indices for voxels in patch)
        mask = mask.reshape((-1,)+ tuple(mask.shape[-3:])) #flatten patch labels
        FA = FA.reshape((-1,)+ tuple(FA.shape[-3:])) #flatten patch labels
        mask = mask.reshape(mask.shape[0],-1) #flatten patch voxels
        FA = FA.reshape(FA.shape[0],-1) #flatten patch voxels
        #take mean for each patch
        self.mask = mask.mean(dim=-1)
        self.FA = FA.mean(dim=-1)

    def generate_training_data(self,N_train):
        #this will generate the inputs and outputs
        N=N_train

        ##get coordinates for data extraction
        occ_inds = np.where(((self.mask>0.1) & (self.FA>0.0)))[0] #extract patches based on mean FA
        #occ_inds = np.where(self.mask>0.3)[0] #extract patches based on mean FA
        print('Max patches available are ', len(occ_inds))
        #occ_inds = np.where(self.mask>0.5)[0]
        if N >= len(occ_inds):
            #raise ValueError('requested patches exceed those availale.')
            print('requested patches exceed those availale')
            N = len(occ_inds)
        oi=np.random.randint(0,len(occ_inds),N) #get random patches
        print('using %d patches' % len(oi))
        xpp=self.x[occ_inds[oi]] #extract coordinates
        ypp=self.y[occ_inds[oi]]
        zpp=self.z[occ_inds[oi]]

        ## generate training data
        H = self.H #----------------dimensions of icosahedron internal space-------------------#
        h = H+1
        w = 5*h
        self.ico = icosahedron.icomesh(m=H-1) 
        I, J, T = d12.padding_basis(H=H) #for padding
        xp= xpp.reshape(-1).numpy() #fully flattened coordinates
        yp= ypp.reshape(-1).numpy()
        zp= zpp.reshape(-1).numpy()
        self.xp = xp
        self.yp = yp
        self.zp = zp
        voxels=np.asarray([xp,yp,zp]).T #putting them in one array
        #inputs
        self.FA_on_points = self.dti.FA.get_fdata()[self.xp,self.yp,self.zp]
        self.diff_input.makeInverseDistInterpMatrix(self.ico.interpolation_mesh) #interpolation initiation
        S0X, X = self.diff_input.makeFlat(voxels,self.ico) #interpolate
        X = X[:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
        shp=tuple(xpp.shape) + (h,w) #we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
        self.FA_on_points = torch.from_numpy(self.FA_on_points).reshape(xpp.shape)
        X = X.reshape(shp)
        #output (labels)
        Y=self.dti.icoSignalFromDti(self.ico)
        Y = Y[:,:,:,I[0,:,:],J[0,:,:]]
        Y=Y[xp,yp,zp]
        Y=Y.reshape(shp) #same shape for outputs
        #standardize both
        X = (X - X.mean())/X.std()
        X = torch.from_numpy(X).contiguous().float()
        Y = (Y - Y.mean())/Y.std()
        Y = torch.from_numpy(Y).contiguous().float()

        # X = (X - np.nanmin(X))/(np.nanmax(X)-np.nanmin(X))
        # X = torch.from_numpy(X).contiguous().float()
        #
        # Y = (Y - np.nanmin(Y))/(np.nanmax(Y)-np.nanmin(Y))
        # Y = torch.from_numpy(Y).contiguous().float()


        self.X = X
        self.Y = Y



def makeFlat(diffpath,outpath,H):
    """
    This will make a nifti in icosahedron representation of first shell
    :param diffpath: path for diffusion data
    :param outpath: path for output
    :return: S0 nifti and nifti with shape [H,W,D,h,w] where h,w are dimensions of icosahedron space. H and D are cut by one for
    divisibility by 3.
    """

    #initialize everything
    print('loading everything')
    diff = diffusion.diffVolume(diffpath)
    diff.cut_ijk()
    ico = icosahedron.icomesh(m=H - 1)
    diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)
    I, J, T = d12.padding_basis(H=H)
    i, j, k = np.where(diff.mask.get_fdata() > 0)
    voxels = np.asarray([i, j, k]).T

    #get icosahedron output
    print('pushing to icosahedron')
    S0, out = diff.makeFlat(voxels, ico)
    print('padding')
    out = out[:, :, I[0, :, :], J[0, :, :]]

    print('saving')
    S0_flat = np.zeros((diff.mask.get_fdata().shape[0:3] + (S0.shape[-1],)))
    S0_flat[i, j, k] = S0

    diff_flat = np.zeros((diff.mask.get_fdata().shape[0:3] + out.shape[1:]))
    diff_flat[i, j, k, :, :, :] = out

    #save everthing (even the cut diffusion)
    nib.save(diff.vol,diffpath + '/data_cut.nii.gz')
    nib.save(diff.mask,diffpath + '/mask_cut.nii.gz')
    diff_flat = nib.Nifti1Image(diff_flat,diff.vol.affine)
    nib.save(diff_flat,outpath+ '/data_cut_flat.nii.gz')
    S0_flat = nib.Nifti1Image(S0_flat, diff.vol.affine)
    nib.save(S0_flat, outpath + '/S0_cut_flat.nii.gz')

    return S0_flat, diff_flat

def flatten_for_2dconv(input):
    """
    Flatten for 2d convolutions
    :param input: tensor of shape [batch, Nc, Nc, Nc, shells, h, w]
    :return: tensor of shape [batch,Nc^3*shells,h,w]
    """
    Nc=input.shape[1]
    h=input.shape[-2]
    w=input.shape[-1]








