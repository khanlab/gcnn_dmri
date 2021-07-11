from preprocessing import training_data
import torch
import numpy as np
import os


class dataLoaderTest:
    def __init__(self,subjects_path,H,Nc,Nsub,Npersub,bdir=6):
        self.path = subjects_path
        self.Nsub = Nsub
        self.Npersub = Npersub
        self.Nc = Nc
        self.H = H
        self.bdir=str(bdir)

        self.loadData()

    def loadData(self):
        
        subjects = os.listdir(self.path)
        
        N_subjects=self.Nsub
        N_per_sub = self.Npersub
        bdir=str(self.bdir)
        sub_path = self.path
        dti_base = self.path
        H = self.H
        Nc = self.Nc

        if(N_subjects >= len(subjects)):
            print('WARNING: not enough subjects left for validation, there will be overlap in training and validation')

        self.X,self.S0X,self.Xflat,self.Y,self.S0Y,self.mask_train,self.interp_matrix,interp_matrix_ind=self.load_annoying(subjects,
                                                                                        N_subjects, 
                                                                                        N_per_sub, 
                                                                                        bdir, 
                                                                                        sub_path, 
                                                                                        dti_base,
                                                                                        H, 
                                                                                        Nc)  

class dataLoader:
    def __init__(self,subjects_path,H,Nc,Nsub,Npersub,bdir=6):
        self.path = subjects_path
        self.Nsub = Nsub
        self.Npersub = Npersub
        self.Nc = Nc
        self.H = H
        self.bdir=str(bdir)

        self.loadData()

    def loadData(self):
        
        subjects = os.listdir(self.path)
        
        N_subjects=self.Nsub
        N_per_sub = self.Npersub
        bdir=str(self.bdir)
        sub_path = self.path
        dti_base = self.path
        H = self.H
        Nc = self.Nc

        if(N_subjects >= len(subjects)):
            print('WARNING: not enough subjects left for validation, there will be overlap in training and validation')

        self.X,self.S0X,self.Xflat,self.Xflat_dti,self.Y,self.S0Y,self.mask_train,self.interp_matrix,interp_matrix_ind=self.load_annoying(subjects,
                                                                                        N_subjects, 
                                                                                        N_per_sub, 
                                                                                        bdir, 
                                                                                        sub_path, 
                                                                                        dti_base,
                                                                                        H, 
                                                                                        Nc)
    
        self.Xv,self.S0Xv,self.Xflatv,self.Xflat_dtiv,self.Yv,self.S0Yv,self.mask_trainv,self.interp_matrixv,interp_matrix_indv=self.load_annoying([subjects[-1]],
                                                                                        1, 
                                                                                        N_per_sub, 
                                                                                        bdir, 
                                                                                        sub_path, 
                                                                                        dti_base,
                                                                                        H, 
                                                                                        Nc)
    

    def load_annoying(self,subjects,N_subjects, N_per_sub, bdir, sub_path, dti_base,H, Nc):

        X=[]
        Y=[]
        S0Y=[]
        Xflat =[]
        Xflat_dti=[]
        S0X=[]
        mask_train= []
        interp_matrix = []
        interp_matrix_ind = []

        for sub in range(0,N_subjects):
            print(subjects[sub])
            this_path = sub_path + '/' + subjects[sub] + '/' + bdir + '/'
            this_tpath = sub_path + '/' + subjects[sub] + '/' + bdir + '/'
            this_dti_in_path = dti_base + subjects[sub] + '/' + bdir + '/dtifit'
            this_dti_path = dti_base + subjects[sub] + '/'+str(90)+'/dtifit'
            this_dti_mask_path = this_path + '/nodif_brain_mask.nii.gz'
            this_subject=training_data(this_path,this_dti_in_path, this_dti_path,this_dti_mask_path,this_tpath,H,N_per_sub,
                                    Nc=Nc)
            #X[sub]= this_subject.X #X and Y are already standarized on a per subject basis
            #Y[sub]= this_subject.Y
            X.append(this_subject.X)
            S0Y.append(this_subject.Y[0])
            Y.append(this_subject.Y[1])
            mask_train.append(this_subject.mask_train)
            interp_matrix.append(torch.from_numpy(np.asarray(this_subject.diff_input.interpolation_matrices)))
            Xflat.append(this_subject.Xflat)
            Xflat_dti.append(this_subject.Xflat_dti)
            S0X.append(this_subject.S0X)


            #for the interp matrix index we will make a torch array of the same size as patches in each subject
            this_interp_matrix_inds = torch.ones([this_subject.X.shape[0]])
            this_interp_matrix_inds[this_interp_matrix_inds==1]=sub
            interp_matrix_ind.append(this_interp_matrix_inds)


        X = torch.cat(X)
        #Y = torch.cat(Y)
        Y=torch.cat(Y)
        S0Y = torch.cat(S0Y)
        mask_train = torch.cat(mask_train)
        interp_matrix = torch.cat(interp_matrix)
        interp_matrix_ind=torch.cat(interp_matrix_ind).int()
        Xflat = torch.cat(Xflat)
        Xflat_dti = torch.cat(Xflat_dti)
        S0X = torch.cat(S0X)

        return X,S0X,Xflat,Xflat_dti,Y,S0Y,mask_train,interp_matrix,interp_matrix_ind

