import numpy as np
import copy
import diffusion
import icosahedron
import random
import dihedral12 as d12
import nibabel as nib



"""
Functions that allow us to load training and test data from nifti files
"""

#TODO: need something here that loads voxels from many different subjects. For example 5000 voxels from 10 subjects


def loadDownUp(downdatapath,updatapath,dtipath,N_train,H=11,all=None,interp='inverse_distance'):
    """
    Function to create training data for upsampling diffusion signal
    """
    def list_to_array_X(flats):
    # convert the lists to arrays and also normalize the data to make attenutations brighter
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

    # load down diffusion data
    diff_down = diffusion.diffVolume()
    diff_down.getVolume(downdatapath)
    diff_down.shells()
    diff_down.makeBvecMeshes()
    
    #load up down mean
    if updatapath != None:
        S0meanup= nib.load(downdatapath+'/S0mean.nii.gz').get_fdata()
    if all == None:
        S0meandown=nib.load(updatapath+'/S0mean.nii.gz').get_fdata()
    

    # load up diffusion data
    if updatapath != None: #not the most ideal what to do this
        diff_up = diffusion.diffVolume()
        diff_up.getVolume(updatapath)
        diff_up.shells()
        diff_up.makeBvecMeshes()

    # get the dti data
    dti = diffusion.dti()
    dti.load(pathprefix=dtipath)
    
    # get the icosahedron ready
    ico = icosahedron.icomesh(m=H-1)
    ico.get_icomesh()
    ico.vertices_to_matrix()
    diff_down.makeInverseDistInterpMatrix(ico.interpolation_mesh)
    if updatapath != None:
        diff_up.makeInverseDistInterpMatrix(ico.interpolation_mesh)

    # these are all the voxels
    #i, j, k = np.where(diff_down.mask.get_fdata() == 1)
    if all == None:
        i, j, k = np.where((dti.FA.get_fdata() > 0.3) & (S0meanup >0) & (S0meandown>0) )
    else:
        i, j, k = np.where(diff_down.mask.get_fdata() == 1)
    voxels = np.asarray([i, j, k]).T

    if all == True: #pick all?
        training_inds=np.arange(0,len(i))
    else:
        training_inds=random.sample(range(0,len(i)),N_train)
        #training_inds=np.arange(0,N_train)
        #training_inds=random.sample(range(0,40*N_train),N_train)

    train_voxels = np.asarray([i[training_inds], j[training_inds], k[training_inds]]).T


    I, J, T = d12.padding_basis(ico.m + 1)

    if all==None:
        S0_down_train, flat_down_train, signal_down_train = diff_down.makeFlat(train_voxels, ico, interp=interp)
        flat_down_train=list_to_array_X(flat_down_train)
        S0_down_train=np.asarray(S0_down_train)

        if updatapath != None:
            S0_up_train, flat_up_train, signal_up_train = diff_up.makeFlat(train_voxels, ico, interp=interp)
            flat_up_train = list_to_array_X(flat_up_train)
            S0_up_train=np.asarray(S0_up_train)

        return S0_down_train,flat_down_train,S0_up_train,flat_up_train
    else:
        S0_down_train, flat_down_train, signal_down_train = diff_down.makeFlat(train_voxels, ico, interp=interp)
        flat_down_train = list_to_array_X(flat_down_train)
        S0_down_train = np.asarray(S0_down_train)
        return S0_down_train,flat_down_train






def load(datapath,dtipath,N_train,N_test=0,N_valid=0,all=None,interp='inverse_distance'):
    """
    :param path: path of diffusion and dti data
    :param N_train: Number of training voxels
    :param N_test: Number of test voxels
    :param start: Starting index for Y vector
    :param end: Ending index for Y vector
    :return: Xtrain, Ytrain, Xtest, Ytest, ico, diff
    """
    def list_to_array_X(S, flats):
        # convert the lists to arrays and also normalize the data to make attenutations brighter
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

    # put Y in array format
    def dti_to_array_Y(dti,voxels): #this will need to made more general

        out=np.zeros([len(voxels),13])

        for i,p in enumerate(voxels):
            out[i, 0]=dti.FA.get_fdata()[tuple(p)]
            out[i, 1] = dti.L1.get_fdata()[tuple(p)]
            out[i, 2] = dti.L2.get_fdata()[tuple(p)]
            out[i, 3] = dti.L3.get_fdata()[tuple(p)]
            out[i, 4:7] = dti.V1.get_fdata()[tuple(p)][:]
            out[i, 7:10] = dti.V2.get_fdata()[tuple(p)][:]
            out[i, 10:13] = dti.V3.get_fdata()[tuple(p)][:]

        return out

    # load diffusion data
    diff = diffusion.diffVolume()
    diff.getVolume(datapath)
    diff.shells()
    diff.makeBvecMeshes()

    # get the dti data
    dti = diffusion.dti()
    dti.load(pathprefix=dtipath)

    # get the icosahedron ready
    ico = icosahedron.icomesh(m=4)
    ico.get_icomesh()
    ico.vertices_to_matrix()
    diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)

    # these are all the voxels
    i, j, k = np.where(diff.mask.get_fdata() == 1)
    #i, j, k = np.where(dti.FA.get_fdata() > 0.3)
    if all == True:
        i, j, k = np.where(diff.mask.get_fdata() == 1)
    voxels = np.asarray([i, j, k]).T

    # have to pick inds in a manner that avoids overlap

    if all == True: #pick all?
        training_inds=np.arange(0,len(i))
        test_inds=[0]
        valid_inds=[0]
    else:
        max_number_voxels=len(voxels)
        N_total=N_train+N_test+N_valid
        cut_train = int(max_number_voxels*N_train/N_total)-1
        cut_test  = cut_train + int(max_number_voxels * N_test / N_total)-1
        cut_valid = cut_test +  int(max_number_voxels * N_valid / N_total)-1
        print(cut_train, cut_test, cut_valid)
        training_inds = random.sample(range(cut_train), N_train)
        test_inds = random.sample(range(cut_train, cut_test), N_test)
        valid_inds = random.sample(range(cut_test, cut_valid), N_test)


    #pick straight voxels for testing
    #N = N_train + N_test
    #all_inds = np.arange(0, N)
    # training_inds = all_inds[0:N_train]
    # test_inds = all_inds[N_train:N]

    train_voxels = np.asarray([i[training_inds], j[training_inds], k[training_inds]]).T
    S0_train, flat_train, signal_train = diff.makeFlat(train_voxels, ico,interp=interp)
    if N_test>0:
        test_voxels = np.asarray([i[test_inds], j[test_inds], k[test_inds]]).T
        S0_test, flat_test, signal_test = diff.makeFlat(test_voxels, ico,interp=interp)
    if N_valid>0:    
        valid_voxels = np.asarray([i[valid_inds], j[valid_inds], k[valid_inds]]).T
        S0_valid, flat_valid, signal_valid = diff.makeFlat(valid_voxels, ico,interp=interp)




    # this is for X_train and X_test
    
    I, J, T = d12.padding_basis(ico.m + 1)


    # this should be in correct format
    X_trainp = np.copy(list_to_array_X(S0_train, flat_train))
    if N_test>0:
        X_testp = np.copy(list_to_array_X(S0_test, flat_test))
    if N_valid>0:
        X_validp = np.copy(list_to_array_X(S0_valid, flat_valid))

    #might have to remove this step later on
    X_trainp[np.isinf(X_trainp)] = 0
    X_trainp[np.isnan(X_trainp)] = 0
    
    if N_test>0:
        X_testp[np.isinf(X_testp)] = 0
        X_testp[np.isnan(X_testp)] = 0
    
    if N_valid>0:
        X_validp[np.isinf(X_validp)] = 0
        X_validp[np.isnan(X_validp)] = 0

    # get all of dti and then select required later on during training

    Y_trainp = dti_to_array_Y(dti,train_voxels)
    if N_test>0:
        Y_testp = dti_to_array_Y(dti,test_voxels)

    if N_valid>0:
        Y_validp = dti_to_array_Y(dti, valid_voxels)

    if N_test>0 and N_valid>0:
        return X_trainp,Y_trainp,X_testp,Y_testp,X_validp,Y_validp, ico,diff
    elif N_test >0:
        return X_trainp,Y_trainp,X_testp,Y_testp, ico,diff
    elif N_valid >0:
        return X_trainp,Y_trainp,X_validp,Y_validp, ico,diff
    else:
        return X_trainp,Y_trainp, ico,diff
