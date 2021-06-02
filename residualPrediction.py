import diffusion
import numpy as np
import icosahedron
import copy
import dihedral12 as d12
import pickle
import training
import torch
import nibabel as nib
import extract3dDiffusion

"""
Fuctions and classes to make predictions
"""

def convert(S,X):
    S=S.reshape(-1,S.shape[3],3*3*3)
    X=X.reshape(-1,12,60,3*3*3)

    S=S.moveaxis((0,1,2),(0,2,1))
    X=X.moveaxis(3,1)

    return S.float(),X.float()

def load_obj(path):
    with open(path + 'modelParams.pkl', 'rb') as f:
        return pickle.load(f)

def preproc(X,S0):
    for i in range(0,X.shape[0]):
        for s in range(0,S0.shape[1]):
            Smean=S0[i,s,:].mean()
            if Smean ==0: 
                #print(S0[i,s,:])
                #print('zero mean encountered')
                Smean = X[i,s,:,:].max() # not sure if the this is the best appraoch
                if Smean ==0:
                    Smean =1
    X[i,s,:,:]=X[i,s,:,:]/Smean
    mean = X[X>0].mean()
    std = X[X>0].std()
    X = (X-mean)/std
    return X,mean,std


class resPredictor:
    def __init__(self,inputpath,netpath,H):
        self.inputpath=inputpath
        self.netpath=netpath
        self.modelParams=[]
        #self.Xpredict = []
        #self.S0Xpredict = []
        #self.net =[]
        self.H=H
        
        #load the training data
        #self.Xpredict = np.load(inputpath + '/X_predict.npy') #TODO:put in the name of the file you want here
        #self.S0_predict = np.load(inputpath + '/S0X_predict.npy') #TODO: put in the name of the file
        #self.Xpredict,self.mean,self.std = preproc(self.Xpredict,self.S0_predict) #TODO: need to figure out what
        self.S0_predict, self.Xpredict = extract3dDiffusion.loader_3d(inputpath, self.H)
        self.S0_predict, self.Xpredict = convert(self.S0_predict, self.Xpredict)
        self.Xpredict, self.mean, self.std = preproc(self.Xpredict, self.S0_predict)

        # this function is
        self.S0_predict =None

        #load the network
        path=self.netpath
        self.modelParams=load_obj(path)
        trnr=training.trainer(self.modelParams)
        trnr.makeNetwork()
        self.net=trnr.net
        self.net.load_state_dict(torch.load(path+ 'net'))

    def predict(self,batch_size=1000):
        self.Ypredict=np.zeros_like(self.Xpredict)
        print('making predictions for a total of '+ str(len(self.Ypredict))+ 'inputs')
        for p in range(0,len(self.Ypredict),batch_size):
        #for p in range(0,2000,batch_size):
            print(p)
            self.Ypredict[p:p+batch_size,:,:,:]=self.Xpredict[p:p+batch_size,:,:,:]+self.net(self.Xpredict[
                                                                                             p:p+batch_size,:,:,:].cuda().float()).cpu().detach().numpy()
        self.Ypredict = self.std*self.Ypredict + self.mean

    def back23d(self):
        self.Ypredict=self.Ypredict.moveaxis(1,-1)
        self.Ypredict=self.Ypredict.view(48,58,48,12,60,3,3,3)
        self.Ypredict=self.Ypredict.moveaxis((0,1,2,3,4,5,6,7),(2,4,6,0,1,3,5,7))
        self.Ypredict=self.Ypredict.reshape(12,60,144,174,144)


    # def makeNifti(self,path,H):
    #     self.diff = diffusion.diffVolume()
    #     self.diff.getVolume(path)
    #     self.diff.shells()
    #
    #     w=5*(H+1)
    #     h=H+1
    #     self.ico=icosahedron.icomesh(H-1)
    #     self.ico.get_icomesh()
    #     self.ico.vertices_to_matrix()
    #     self.ico.grid2xyz()
    #
    #
    #     #how many unique directions are there? (!)
    #
    #     basis=np.zeros([h,w])
    #     for c in range(0,5):
    #         basis[1:H,c*h+1:(c+1)*h-1]=1
    #
    #     N=len(basis[basis==1])+1
    #     print('Number of bdirs is: ', N)
    #
    #     N_random=101
    #     rng=np.random.default_rng()
    #     inds=rng.choice(N-2,size=N_random,replace=False)+1
    #     inds[0]=0
    #
    #     bvals=np.zeros(N_random)
    #     x_bvecs=np.zeros(N_random)
    #     y_bvecs=np.zeros(N_random)
    #     z_bvecs=np.zeros(N_random)
    #
    #     x_bvecs[1:]=self.ico.X_in_grid[basis==1].flatten()[inds[1:]]
    #     y_bvecs[1:]=self.ico.Y_in_grid[basis==1].flatten()[inds[1:]]
    #     z_bvecs[1:]=self.ico.Z_in_grid[basis==1].flatten()[inds[1:]]
    #
    #     bvals[1:]=1000
    #
    #
    #
    #     sz=self.diff.vol.get_fdata().shape
    #     diff_out=np.zeros([sz[0],sz[1],sz[2],N_random])
    #
    #     diff_out[:,:,:,0]=1 #b0
    #     i, j, k = np.where(self.diff.mask.get_fdata() == 1)
    #     for p in range(0,len(i)):
    #         signal=self.Ypredict[p,0,:,:] #it is assumed that post processing has been done on this and it is the final signal
    #         signal = signal[basis==1].flatten()
    #         # print(diff_out.shape)
    #         # print(signal.shape)
    #         diff_out[i[p],j[p],k[p],1:] = signal[inds[1:]]
    #
    #     diff_out=nib.Nifti1Image(diff_out,self.diff.vol.affine)
    #     nib.save(diff_out,path+'/data_network.nii.gz')
    #
    #     #write the bvecs and bvals
    #     fbval = open(path + '/bvals_network', "w")
    #     for bval in bvals:
    #         fbval.write(str(bval)+" ")
    #
    #     fbvecs = open(path + '/bvecs_network',"w")
    #     for x in x_bvecs:
    #         fbvecs.write(str(x)+ ' ')
    #     fbvecs.write('\n')
    #     for y in y_bvecs:
    #         fbvecs.write(str(y)+ ' ')
    #     fbvecs.write('\n')
    #     for z in z_bvecs:
    #         fbvecs.write(str(z)+ ' ')
    #     fbvecs.write('\n')
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
