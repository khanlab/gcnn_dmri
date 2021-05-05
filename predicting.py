import diffusion
import numpy as np
import icosahedron
import copy
import dihedral12 as d12
import pickle
import training
import torch
import nibabel as nib


"""
Fuctions and classes to make predictions
"""


def load_obj(path):
    with open(path + 'modelParams.pkl', 'rb') as f:
        return pickle.load(f)

def convert2cuda(X_train):
    X_train_p = np.copy(1/X_train)
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
        self.Xpredict=np.load(path)

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

    def loadNetwork(self,path):
        #load pkl file for model params and initiate network
        self.modelParams=load_obj(path)
        trnr=training.trainer(self.modelParams,0,0)
        trnr.makeNetwork()
        self.net=trnr.net
        self.net.load_state_dict(torch.load(path+ 'net'))

    def predict(self,Nout=3,batch_size=1000):
        self.Xpredict=convert2cuda(self.Xpredict)
        self.Ypredict=np.zeros([len(self.Xpredict),Nout])
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