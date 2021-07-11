import nibabel as nib
import diffusion
import icosahedron
import dihedral12 as d12
import torch
import numpy as np


class training_data:
    def __init__(self,inputpath,dtipath_in,dtipath, maskpath, tpath, H, N_train=None, Nc=16):
        self.inputpath = inputpath
        self.dtipath = dtipath 
        self.dtipath_in = dtipath_in
        self.maskpath = maskpath
        self.tpath = tpath

        self.diff_input = diffusion.diffVolume(inputpath)
        self.dti=diffusion.dti(self.dtipath,self.maskpath)
        self.dti_in=diffusion.dti(self.dtipath_in,self.maskpath)

        self.t1 = nib.load(tpath + '/T1_cut_pad.nii.gz')
        self.t2 = nib.load(tpath + '/T2_cut_pad.nii.gz')


        self.interp_matrix = []

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

        #mask = torch.from_numpy(self.dti.mask.get_fdata()).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
        mask = nib.load(self.inputpath+'/mask.nii.gz')
        mask = torch.from_numpy(mask.get_fdata()).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
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
        occ_inds = np.where(((self.mask>0.00) & (self.FA>0.0)))[0] #extract patches based on mean FA
        #occ_inds = np.where(self.mask>0.3)[0] #extract patches based on mean FA
        print('Max patches available are ', len(occ_inds))
        #occ_inds = np.where(self.mask>0.5)[0]
        print('N is:', N_train)
        if N is not None:
            if (N >= len(occ_inds)):
                #raise ValueError('requested patches exceed those availale.')
                print('requested patches exceed those availale')
                N = len(occ_inds)
        if N_train is None:
            oi=np.arange(0,len(occ_inds))
        else:
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
        self.mask_train = nib.load(self.inputpath + 'mask.nii.gz').get_fdata()[self.xp,self.yp,self.zp] #this is the freesurfer mask
        self.diff_input.makeInverseDistInterpMatrix(self.ico.interpolation_mesh) #interpolation initiation
        
        #shp=tuple(xpp.shape) + (h,w) #we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
        self.FA_on_points = torch.from_numpy(self.FA_on_points).reshape(xpp.shape)
        self.mask_train = torch.from_numpy(self.mask_train).reshape(xpp.shape)
        
        #S0X, X = self.diff_input.makeFlat(voxels,self.ico) #interpolate
        
        
        # S0X, X = self.diff_input.makeFlat(voxels,self.ico) #interpolate
        # X = X[:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
        # X = X.reshape(shp)


        #X = self.dti_in.icoSignalFromDti(self.ico)
        #X = X[:,:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
        shp = tuple(xpp.shape) + (self.diff_input.vol.shape[-1],)  # we are putting this in the shape of list [
                                                                 # patch_label_list,Nc,Nc,Nc,Ndir]

        #T1 and T2 stuff
        self.XT1 = self.t1.get_fdata()[xp, yp, zp].reshape(xpp.shape)
        self.XT2 = self.t2.get_fdata()[xp, yp, zp].reshape(xpp.shape)

        self.XT1 = (self.XT1 - self.XT1.mean()) / self.XT1.std()
        self.XT2 = (self.XT2 - self.XT2.mean()) / self.XT2.std()

        self.XT1 = torch.from_numpy(self.XT1).contiguous().float()
        self.XT2 = torch.from_numpy(self.XT2).contiguous().float()

        self.XT1 = self.XT1.view(self.XT1.shape + (1,))
        self.XT2 = self.XT2.view(self.XT2.shape + (1,))


        X = self.diff_input.vol.get_fdata()[xp,yp,zp,:]
        #X = X[xp,yp,zp]
        X = X.reshape(shp)


        S0X, Xflat = self.diff_input.makeFlat(voxels,self.ico) #interpolate
        Xflat = Xflat[:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
        shp = tuple(xpp.shape) + (h, w)
        Xflat = Xflat.reshape(shp)
        S0X = S0X.reshape(xpp.shape)

        #we want also Xflat_dti
        Xflat_dti = self.dti_in.icoSignalFromDti(self.ico)
        Xflat_dti = Xflat_dti/self.dti.S0[:,:,:,None,None]
        Xflat_dti = Xflat_dti[:,:,:,I[0,:,:],J[0,:,:]]
        Xflat_dti = Xflat_dti[xp,yp,zp]
        Xflat_dti = Xflat_dti.reshape(tuple(xpp.shape) + (h,w))

        #output (labels)
        shp = tuple(xpp.shape) + (h, w)  # we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
        Y=self.dti.icoSignalFromDti(self.ico)/self.dti.S0[:,:,:,None,None]
        Y = Y[:,:,:,I[0,:,:],J[0,:,:]]
        Y = Y[xp,yp,zp]
        Y = Y.reshape(shp) #same shape for outputs
        S0Y = self.dti.S0.get_fdata()[xp,yp,zp]
        S0Y = S0Y.reshape(xpp.shape)

        S0Y = (S0Y - S0Y.mean())/S0Y.std()
        S0Y = torch.from_numpy(S0Y).contiguous().float()

        #standardize both
        #save mean for predictions
        self.Xmean = X.mean()
        self.Xstd = X.std()
        self.S0Xmean = S0X.mean()
        self.S0Xstd = S0X.std()
        self.Xflatmean =Xflat.mean()
        self.Xflatstd = Xflat.std()

        X = (X - X.mean())/X.std()
        X = torch.from_numpy(X).contiguous().float()
        Y = (Y - Y.mean())/Y.std()
        Y = torch.from_numpy(Y).contiguous().float()

        # X = (X - np.nanmin(X))/(np.nanmax(X)-np.nanmin(X))
        # X = torch.from_numpy(X).contiguous().float()
        #
        # Y = (Y - np.nanmin(Y))/(np.nanmax(Y)-np.nanmin(Y))
        # Y = torch.from_numpy(Y).contiguous().float()

        self.Xflat =torch.from_numpy( (Xflat - Xflat.mean())/Xflat.std()).contiguous().float()
        self.S0X = torch.from_numpy((S0X - S0X.mean())/S0X.std()).contiguous().float()
        self.Xflat_dti =torch.from_numpy( (Xflat_dti - Xflat_dti.mean())/Xflat_dti.std()).contiguous().float()

        self.X = torch.cat([self.XT1,self.XT2,X],dim=-1)
        self.X = self.X.moveaxis(-1,1)
        self.Y = [S0Y,Y]



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
    #S0_flat = S0_flat[:, :, :,0]

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








