import numpy as np
import copy
import diffusion
import icosahedron
import random
import dihedral12 as d12
from nibabel import load



"""
Functions that allow us to load training and test data from nifti files
"""

def load(datapath,dtipath,N_train,N_test,N_valid,interp='inverse_distance'):
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
    voxels = np.asarray([i, j, k]).T

    # have to pick inds in a manner that avoids overlap
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
    test_voxels = np.asarray([i[test_inds], j[test_inds], k[test_inds]]).T
    valid_voxels = np.asarray([i[valid_inds], j[valid_inds], k[valid_inds]]).T

    # this is for X_train and X_test
    S0_train, flat_train, signal_train = diff.makeFlat(train_voxels, ico,interp=interp)
    S0_test, flat_test, signal_test = diff.makeFlat(test_voxels, ico,interp=interp)
    S0_valid, flat_valid, signal_valid = diff.makeFlat(valid_voxels, ico,interp=interp)

    I, J, T = d12.padding_basis(ico.m + 1)


    # this should be in correct format
    X_trainp = np.copy(list_to_array_X(S0_train, flat_train))
    X_testp = np.copy(list_to_array_X(S0_test, flat_test))
    X_validp = np.copy(list_to_array_X(S0_valid, flat_valid))

    #might have to remove this step later on
    X_trainp[np.isinf(X_trainp)] = 0
    X_testp[np.isinf(X_testp)] = 0
    X_validp[np.isinf(X_validp)] = 0
    X_trainp[np.isnan(X_trainp)] = 0
    X_testp[np.isnan(X_testp)] = 0
    X_validp[np.isnan(X_validp)] = 0

    # get all of dti and then select required later on during training


    Y_trainp = dti_to_array_Y(dti,train_voxels)
    Y_testp = dti_to_array_Y(dti,test_voxels)
    Y_validp = dti_to_array_Y(dti, valid_voxels)

    return X_trainp,Y_trainp,X_testp,Y_testp,X_validp,Y_validp, ico,diff
