import diffusion
import numpy as np
import icosahedron
import copy
import dihedral12 as d12
import pickle
import training
import torch
import nibabel as nib
import preprocessing


"""
Fuctions and classes to make predictions
"""


def load_obj(path):
    with open(path + 'modelParams.pkl', 'rb') as f:
        return pickle.load(f)

def convert2cuda(X_train):
    X_train_p = np.copy(X_train)
    #X_train_p = np.copy(X_train)
    X_train_p[np.isinf(X_train_p)] = 0
    X_train_p[np.isnan(X_train_p)] = 0
    inputs = X_train_p
    inputs = torch.from_numpy(inputs.astype(np.float32))
    input = inputs.detach()
    input = input.cuda()
    return input

def normalize(Ypredict):
    out=Ypredict
    norm=np.sqrt(np.sum(out*out,-1))
    return out/norm[:,None]


class predicting_data:
    def __init__(self, inputpath, H,N_patch=16):
        
        self.inputpath = inputpath
        self.diff_input = diffusion.diffVolume(inputpath)
        self.in_shp = self.diff_input.vol.shape


        self.N_patch= N_patch
        self.H = H

        self.make_coords()
        self.generate_predicting_data()

    def make_coords(self):
        #we make patches with meshgrids of coords and then extraxt data using those coords
        #we also make mask list here
        print("Making coords for patch extraction")
        
        self.x = np.arange(0,self.in_shp[0])
        self.y = np.arange(0,self.in_shp[1])
        self.z = np.arange(0,self.in_shp[2])
        
        self.x,self.y,self.z = np.meshgrid(self.x,self.y,self.z,indexing='ij')

        self.x=torch.from_numpy(self.x)
        self.y=torch.from_numpy(self.y)
        self.z=torch.from_numpy(self.z)

        Nc=self.N_patch
        
        Nx = Nc
        Ny = Nc
        Nz = Nc
        
        if Nx >= self.x.shape[0]: Nx = self.x.shape[0]
        if Ny >= self.y.shape[0]: Ny = self.y.shape[0]
        if Nz >= self.z.shape[0]: Nz = self.z.shape[0]


        self.x=self.x.unfold(0,Nx,Nx).unfold(1,Nx,Nx).unfold(2,Nx,Nx)
        self.y=self.y.unfold(0,Ny,Ny).unfold(1,Ny,Ny).unfold(2,Ny,Ny)
        self.z=self.z.unfold(0,Nz,Nz).unfold(1,Nz,Nz).unfold(2,Nz,Nz)

        self.x = self.x.reshape((-1,)+tuple(self.x.shape[-3:]))
        self.y = self.y.reshape((-1,)+tuple(self.y.shape[-3:]))
        self.z = self.z.reshape((-1,)+tuple(self.z.shape[-3:]))

        mask = torch.from_numpy(self.diff_input.mask.get_fdata()).unfold(0,Nx,Nx).unfold(1,Ny,Ny).unfold(2,Nz,Nz)
        #flatten patch labels (indices for each patch) and patch indices (indices for voxels in patch)
        mask = mask.reshape((-1,)+ tuple(mask.shape[-3:])) #flatten patch labels
        mask = mask.reshape(mask.shape[0],-1) #flatten patch voxels
        #take mean for each patch
        self.mask = mask.mean(dim=-1)

    def generate_predicting_data(self):
            #this will generate the inputs and outputs
            
            ##get coordinates for data extraction
            occ_inds = np.where(self.mask>0)[0] #extract patches based on mean FA
            #occ_inds = np.where(self.mask>0.5)[0]
            xpp=self.x[occ_inds] #extract coordinates
            ypp=self.y[occ_inds]
            zpp=self.z[occ_inds]

            ## generate training data
            H = self.H #----------------dimensions of icosahedron internal space-------------------#
            h = H+1
            w = 5*h
            self.ico = icosahedron.icomesh(m=H-1) 
            I, J, T = d12.padding_basis(H=H) #for padding
            xp= xpp.reshape(-1).numpy() #fully flattened coordinates
            yp= ypp.reshape(-1).numpy()
            zp= zpp.reshape(-1).numpy()
            voxels=np.asarray([xp,yp,zp]).T #putting them in one array
            #inputs
            self.diff_input.makeInverseDistInterpMatrix(self.ico.interpolation_mesh) #interpolation initiation
            S0X, X = self.diff_input.makeFlat(voxels,self.ico) #interpolate
            X = X[:,:,I[0,:,:],J[0,:,:]] #pad (this is input data)
            shp=tuple(xpp.shape) + (h,w) #we are putting this in the shape of list [patch_label_list,Nc,Nc,Nc,h,w]
            X = X.reshape(shp)
            #standardize inputs
            # self.Xmean = X.mean()
            # self.Xstd = X.std()
            # X = (X - self.Xmean)/self.Xstd
            # X = torch.from_numpy(X).contiguous().float()
            self.Xmax = np.nanmax(X)
            self.Xmin = np.nanmin(X)
            X = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
            X = torch.from_numpy(X).contiguous().float()

            shp = X.shape
            X = X.view(shp[0:4] + (1,) + shp[-2:])

            self.X = X
            self.xp = xp
            self.yp = yp
            self.zp = zp 
        
class residual5dPredictor:
    def __init__(self,datapath,netpath,B=None,Nc=None,Ncore=None,core=None,core_inv=None,
                 I=None,J=None,zeros=None):
        self.datapath = datapath
        self.netpath = netpath
        self.modelParams = []
        self.pred_data = []
        self.B = B
        self.Nc = Nc
        self.Ncore = Ncore
        self.core = core
        self.core_inv = core_inv
        self.I = I
        self.J = J
        self.zeros = zeros

        self.loadNetwork()
        self.generate_predicting_data()


    def loadNetwork(self):
        #load pkl file for model params and initiate network
        path=self.netpath
        self.modelParams=load_obj(path)
        trnr=training.trainer(self.modelParams,0,0,
                              B=self.B,
                              Nc = self.Nc,
                              Ncore = self.Ncore,
                              core=self.core,
                              core_inv=self.core_inv,
                              I=self.I,
                              J=self.J,
                              zeros=self.zeros)
        trnr.makeNetwork()
        self.net=trnr.net
        self.net.load_state_dict(torch.load(path+ 'net',map_location=torch.device('cpu')))

    def generate_predicting_data(self):
        self.H = self.modelParams['H']
        self.pred_data=predicting_data(self.datapath,self.H,N_patch=self.Nc)

    def predict(self, outpath,batch_size=1):
        H = self.H
        h = self.H +1
        w = 5*h
        pred = torch.zeros_like(self.pred_data.X)
        batch_size=1
        #device = self.net.device.type
        for i in range(0,self.pred_data.X.shape[0],batch_size):
            print(i)
            pred[i:i+batch_size]=(self.net(self.pred_data.X[i:i+batch_size].cuda() ).cpu() + self.pred_data.X[i:i+batch_size]).detach()

        out = np.zeros((self.pred_data.diff_input.vol.shape[0:3] + (h,w)))
        oldshp = pred.shape
        pred = pred.view(-1,h,w)
        #pred = pred*self.pred_data.Xstd + self.pred_data.Xmean
        pred = pred * (self.pred_data.Xmax - self.pred_data.Xmin) + self.pred_data.Xmin

        out[self.pred_data.xp,self.pred_data.yp,self.pred_data.zp] = pred

        #make nifti
        # basis=np.zeros([h,w])
        # for c in range(0,5):
        #     basis[1:H,c*h+1:(c+1)*h-1]=1
        # N=len(basis[basis==1])+1

        N=self.Ncore

        print('Number of bdirs is: ', N)

        #N_random=2*w
        N_random=N-2
        rng=np.random.default_rng()
        inds=rng.choice(N-2,size=N_random,replace=False)+1
        inds[0]=0

        bvals=np.zeros(N_random)
        x_bvecs=np.zeros(N_random)
        y_bvecs=np.zeros(N_random)
        z_bvecs=np.zeros(N_random)

        x_bvecs[1:]=self.pred_data.ico.X_in_grid[self.core==1].flatten()[inds[1:]]
        y_bvecs[1:]=self.pred_data.ico.Y_in_grid[self.core==1].flatten()[inds[1:]]
        z_bvecs[1:]=self.pred_data.ico.Z_in_grid[self.core==1].flatten()[inds[1:]]

        bvals[1:]=1000

        sz=out.shape
        diff_out=np.zeros([sz[0],sz[1],sz[2],N_random])
        diff_out[:,:,:,0]=self.pred_data.diff_input.vol.get_fdata()[:,:,:,self.pred_data.diff_input.inds[0]].mean(-1)
        i, j, k = np.where(self.pred_data.diff_input.mask.get_fdata() == 1)

        for p in range(0,len(i)):
            signal =out[i[p],j[p],k[p]]
            signal = signal[self.core==1].flatten()
            diff_out[i[p],j[p],k[p],1:] = signal[inds[1:]]

        diff_out=nib.Nifti1Image(diff_out,self.pred_data.diff_input.vol.affine)
        nib.save(diff_out,outpath+'./data_network.nii.gz')

        #write the bvecs and bvals
        fbval = open(outpath+'./bvals_network', "w")
        for bval in bvals:
            fbval.write(str(bval)+" ")
        fbval.close()

        fbvecs = open(outpath+'./bvecs_network',"w")
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


       


class predictor:
    def __init__(self, datapath, dtipath, netpath):
        self.datapath=datapath
        self.dtipath=dtipath
        self.modelParams=[]
        self.diff=[]
        self.dti=[]
        self.Xpredict=[]
        self.Ypredict=[]
        self.net=[]
        self.netpath = netpath

        # load diffusion data
        self.diff = diffusion.diffVolume()
        self.diff.getVolume(datapath)
        self.diff.shells()
        self.diff.makeBvecMeshes()

        # get the dti data
        self.dti = diffusion.dti()
        self.dti.load(pathprefix=dtipath)
        self.i, self.j, self.k = np.where(self.diff.mask.get_fdata() == 1)

        self.ico=[]
        self.m=4

    def nii2Xpredict(self,path,m=4):
        """
        This function will convert whole volume into Xpredict, this is time consuming
        :return: populate Xpredict also save Xpredict in path
        """
        self.m=m
        self.ico = icosahedron.icomesh(m=self.m)
        self.ico.get_icomesh()
        self.ico.vertices_to_matrix()
        self.diff.makeInverseDistInterpMatrix(self.ico.interpolation_mesh)
        voxels= np.asarray([self.i, self.j, self.k]).T
        S0,flat,signal=self.diff.makeFlat(voxels,self.ico,interp=self.interp)

        self.Xpredict = np.copy(self.list_to_array_X(S0, flat))
        np.save(path,self.Xpredict)

    def loadXpredict(self,path):
        """
        This will load Xpredict if it exists in path
        :param path:
        :return: populate self.Xpredict
        """
        self.Xpredict=1-np.load(path)
        #print(self.Xpredict.shape)
        #Xpredict_mean = np.mean(self.Xpredict,axis=0)
        #Xpredict_std = np.std(self.Xpredict,axis=0)
        #self.Xpredict = (self.Xpredict - Xpredict_mean)/Xpredict_std
        

    def list_to_array_X(self,S, flats):
        # convert the lists to arrays and also normalize the data to make attenutations brighter
        I, J, T = d12.padding_basis(self.ico.m + 1)
        N = len(flats)
        shells = len(flats[0])
        h = len(flats[0][0])
        w = len(flats[0][0][0])
        out = np.zeros([N, shells, h, w])
        for p in range(0, N):
            for s in range(0, shells):
                temp = copy.deepcopy(flats[p][s][I[0,:,:],J[0,:,:]])
                out[p, s, :, :]=temp #notice no normalizaton applied for now
        return out

    def loadNetwork(self):
        #load pkl file for model params and initiate network
        path=self.netpath
        self.modelParams=load_obj(path)
        trnr=training.trainer(self.modelParams,0,0)
        trnr.makeNetwork()
        self.net=trnr.net
        self.net.load_state_dict(torch.load(path+ 'net'))

    def predict(self,Nout=3,batch_size=1000):
        self.Xpredict=convert2cuda(self.Xpredict)
        self.Ypredict=np.zeros([len(self.Xpredict),Nout])
        print('making predictions for a total of '+ str(len(self.Ypredict))+ 'inputs')
        for p in range(0,len(self.Ypredict),batch_size):
            print(p)
            self.Ypredict[p:p+batch_size,:]=self.net(self.Xpredict[p:p+batch_size,:]).cpu().detach()




    def savePredictions(self,path):
        
        Y=normalize(self.Ypredict)
        
        Nout=self.Ypredict.shape[-1]
        
        dtiout=np.zeros_like(self.dti.V1.get_fdata())
        dtiout[self.i,self.j,self.k,:]=Y[:,:]

        dtiout=nib.Nifti1Image(dtiout,self.dti.V1.affine)
        nib.save(dtiout,path)


        pass
