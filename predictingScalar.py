import preprocessing
from preprocessing import training_data as predicting_data
from predicting import load_obj
import trainingScalars as training
import torch
import numpy as np
import icosahedron
import dihedral12 as d12
import nibabel as nib

class residual5dPredictorScalar:
    def __init__(self,inputpath,dtipath_in,
                 dtipath, tpath, netpath, H, Nc=None,
                 B=None,Ncore=None,
                 core=None,core_inv=None,
                 I=None,J=None,zeros=None):
        

        
        self.inputpath = inputpath
        self.dtipath = dtipath
        self.dtipath_in= dtipath_in
        self.netpath = netpath 
        self.tpath =  tpath
        self.H = H
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



        self.X =[]
        self.Xmean = []
        self.Xstd = []
        self.S0X=[]
        self.Xflat =[]
        self.interp_matrix = []
        self.interp_matrix_ind = []

        H = self.H #----------------dimensions of icosahedron internal space-------------------#
        h = H+1
        w = 5*h
        self.ico = icosahedron.icomesh(m=H-1) 
        self.pI, self.pJ, self.pT = d12.padding_basis(H=H) #for padding
       
       
        self.generate_predicting_data()
        self.loadNetwork()


    def generate_predicting_data(self):
        self.pred_data=predicting_data(self.inputpath,self.dtipath_in,
                                  self.dtipath,self.inputpath + '/nodif_brain_mask.nii.gz',
                                  self.tpath,self.H,Nc=16)

        self.X = self.pred_data.X
        self.Xmean = self.pred_data.Xmean
        self.Xstd = self.pred_data.Xstd 
        self.S0X = self.pred_data.S0X
        self.S0Xmean = self.pred_data.S0Xmean
        self.S0Xstd = self.pred_data.S0Xstd
        self.Xflat = self.pred_data.Xflat
        self.Xflatmean = self.pred_data.Xflatmean
        self.Xflatstd = self.pred_data.Xflatstd
        self.interp_matrix = torch.from_numpy(np.asarray(self.pred_data.diff_input.interpolation_matrices)) 

       


    def loadNetwork(self):
        path = self.netpath
        self.modelParams=load_obj(path)
        trnr = training.trainer(self.modelParams,
                                #interp_matrix_ind_valid=interp_matrix_ind_valid,maskvalid=mask_valid,
                                Nscalars=1,Ndir=6,ico=self.ico,
                                B=1,Nc=16,Ncore=100,core=self.ico.core_basis,
                                core_inv=self.ico.core_basis_inv,
                                zeros=self.ico.zeros,
                                I=self.pI,J=self.pJ)
        trnr.makeNetwork()
        self.net = trnr.net
        self.net.load_state_dict(torch.load(path + 'net'))


    def predict(self,outpath,batch_size=1):
        H=self.H
        h=self.H+1
        w=5*h
        B = self.X.shape[0]
        shp = self.X.shape[-3:]
        pred_S0Y = torch.zeros((B,)+shp)
        pred_Yflat = torch.zeros((B,)+shp + (h,w))
        device = list(self.net.parameters())[0].device.type
        print('Starting prediction')
        for i in range(0,self.X.shape[0],batch_size):
        #for i in range(0,1,batch_size):
            print(i)
            torch.cuda.empty_cache()
            S0Y,Yflat=self.net(self.X[i:i+batch_size].float().to(device),self.interp_matrix.float().to(device))
            print(S0Y.device.type)
            S0Y = S0Y.detach().cpu()
            Yflat = Yflat.detach().cpu()
            pred_S0Y[i:i+batch_size]=S0Y
            pred_Yflat[i:i+batch_size]= self.Xflat[i:i+batch_size] + Yflat
            del S0Y, Yflat
            torch.cuda.empty_cache()


        pred_S0Y = pred_S0Y*self.S0Xstd + self.S0Xmean
        pred_Yflat = pred_Yflat*self.Xflatstd + self.Xflatmean

        out_S0 = np.zeros(self.pred_data.diff_input.vol.shape[0:3])
        out_flat = np.zeros(self.pred_data.diff_input.vol.shape[0:3] + (h,w))

        pred_S0Y = pred_S0Y.view(-1)
        pred_Yflat = pred_Yflat.view(-1,h,w)

        out_S0[self.pred_data.xp,self.pred_data.yp,self.pred_data.zp] = pred_S0Y
        out_flat[self.pred_data.xp,self.pred_data.yp,self.pred_data.zp] = pred_Yflat


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

        x_bvecs[1:]=self.ico.X_in_grid[self.core==1].flatten()[inds[1:]]
        y_bvecs[1:]=self.ico.Y_in_grid[self.core==1].flatten()[inds[1:]]
        z_bvecs[1:]=self.ico.Z_in_grid[self.core==1].flatten()[inds[1:]]

        bvals[1:]=1000

        sz=out_flat.shape
        diff_out=np.zeros([sz[0],sz[1],sz[2],N_random])
        diff_out[:,:,:,0]=self.pred_data.diff_input.vol.get_fdata()[:,:,:,self.pred_data.diff_input.inds[0]].mean(-1)
        i, j, k = np.where(self.pred_data.diff_input.mask.get_fdata() == 1)

        for p in range(0,len(i)):
            signal =out_flat[i[p],j[p],k[p]]
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




