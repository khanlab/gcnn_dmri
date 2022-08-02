import nibabel as nib
import diffusion
import icosahedron
import dihedral12 as d12
import torch
import numpy as np


class training_data:
    def __init__(self,inputpath,dtipath_in,dtipath, maskpath, tpath,train_mask_path, H, N_train=None, Nc=16):
        self.inputpath = inputpath
        self.dtipath_in = dtipath_in
        self.dtipath = dtipath 
        self.maskpath = maskpath
        self.tpath = tpath
        self.train_mask_path=train_mask_path

        self.diff_input = diffusion.diffVolume(inputpath)
        self.dti=diffusion.dti(self.dtipath,self.maskpath)
        self.dti_in=diffusion.dti(self.dtipath_in,self.maskpath)

        self.t1 = nib.load(tpath + '/T1.nii.gz')
        self.t2 = nib.load(tpath + '/T2.nii.gz')


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
        #mask = nib.load(self.inputpath+'/mask.nii.gz')

        mask1=nib.load(self.maskpath).get_fdata()
        #mask2=nib.load(self.tpath + '/masks/mask.nii.gz').get_fdata()
        mask2=nib.load(self.train_mask_path).get_fdata()
        mask=np.zeros_like(mask1)
        mask[ (mask1==1) & (mask2==1) ]=1
        mask = torch.from_numpy(mask).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)

        #mask = nib.load(self.maskpath)#+'/mask.nii.gz')
        #mask = torch.from_numpy(mask.get_fdata()).unfold(0,Nc,Nc).unfold(1,Nc,Nc).unfold(2,Nc,Nc)
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
        #self.mask_train = nib.load(self.maskpath).get_fdata()[self.xp,self.yp,self.zp]#nib.load(self.inputpath + 'mask.nii.gz').get_fdata()[self.xp,self.yp,self.zp] #this is the freesurfer mask
        mask1=nib.load(self.maskpath).get_fdata()
        mask2=nib.load(self.train_mask_path).get_fdata()
        self.mask_train=np.zeros_like(mask1)
        self.mask_train[ (mask1==1) & (mask2==1) ]=1
        self.mask_train = self.mask_train[self.xp,self.yp,self.zp] #this is the freesurfer mask
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

        self.XT1[np.isnan(self.XT1)]=0
        self.XT2[np.isnan(self.XT2)]=0

        self.XT1[np.isinf(self.XT1)]=0
        self.XT2[np.isinf(self.XT2)]=0


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
        #Xflat[:,:,self.ico.corners_in_grid==1]=0
        shp = tuple(xpp.shape) + (h, w)
        Xflat = Xflat.reshape(shp)
        S0X = S0X.reshape(xpp.shape)

        S0X[np.isnan(S0X)]=0
        X[np.isnan(X)]=0
        Xflat[np.isnan(Xflat)]=0
        S0X[np.isinf(S0X)]=0
        X[np.isinf(X)]=0
        Xflat[np.isinf(Xflat)]=0

        #we want also Xflat_dti
        Xflat_dti=None
        #Xflat_dti = self.dti_in.icoSignalFromDti(self.ico)
        #Xflat_dti = Xflat_dti/self.dti.S0[:,:,:,None,None]
        #Xflat_dti = Xflat_dti[:,:,:,I[0,:,:],J[0,:,:]]
        #Xflat_dti = Xflat_dti[xp,yp,zp]
        #Xflat_dti = Xflat_dti.reshape(tuple(xpp.shape) + (h,w))

        #output (labels)
        shp = tuple(xpp.shape) + (h, w)  # we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
        Y=self.dti.icoSignalFromDti(self.ico)#/self.dti.S0[:,:,:,None,None]
        Y = Y[:,:,:,I[0,:,:],J[0,:,:]]
        #Y[:,:,:,self.ico.corners_in_grid==1]=0
        Y = Y[xp,yp,zp]
        Y = Y.reshape(shp) #same shape for outputs
        S0Y = self.dti.S0.get_fdata()[xp,yp,zp]
        S0Y = S0Y.reshape(xpp.shape)

        S0Y[np.isnan(S0Y)]=0
        Y[np.isnan(Y)]=0
        S0Y[np.isinf(S0Y)]=0
        Y[np.isinf(Y)]=0
        

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
        #self.Xflat_dti =torch.from_numpy( (Xflat_dti - Xflat_dti.mean())/Xflat_dti.std()).contiguous().float()

        self.X = torch.cat([self.XT1,self.XT2,X],dim=-1)
        self.X = self.X.moveaxis(-1,1)
        self.Y = [S0Y,Y]








