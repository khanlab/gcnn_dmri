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

def load(basepath,N_train,N_test,start=None,end=None):
    """
    :param path: path of diffusion and dti data
    :param N_train: Number of training voxels
    :param N_test: Number of test voxels
    :param start: Starting index for Y vector
    :param end: Ending index for Y vector
    :return: Xtrain, Ytrain, Xtest, Ytest, ico, diff
    """
    def list_to_array_X(S, flats):
        # convert the lists to arrays and also renormalize the data to make attenutations brighter
        N = len(flats)
        shells = len(flats[0])
        h = len(flats[0][0])
        w = len(flats[0][0][0])
        out = np.zeros([N, shells, h, w])
        for p in range(0, N):
            #S_mean = S[p].mean()
            #if (np.isnan(S_mean) == 1) | (S_mean == 0):
            #    print('Nan! or zero')
            for s in range(0, shells):
                temp = copy.deepcopy(flats[p][s][I[0,:,:],J[0,:,:]])
                #temp= 1 - temp
                #temp[temp==1]=0
                out[p, s, :, :]=temp
                # out[p, s, :, :] = 1000/flats[p][s][I[:, :, 0], J[:, :, 0]]
                # out[p, s, :, :] = (flats[p][s][I[0, :, :], J[0, :, :]]/S_mean)
                #out[p, s, :, :] = (flats[p][s][I[0, :, :], J[0, :, :]])
                # for chart in range(0,5):
                #    top=chart*w
                #    bottom=top+w-1
                #    out[p,s,top,0]=0
                #    out[p, s, bottom, 0] = 0
                # out[out==S_mean]=0
        return out

    # put Y in array format
    def list_to_array_Y(L1, L2, L3):
        N = len(L1)
        out = np.zeros([N, 3])
        scale = 1
        for p in range(0, N):
            out[p, 0] = scale * L1[p]
            out[p, 1] = scale * L2[p]
            out[p, 2] = scale * L3[p]
        return out

    # load diffusion data
    diff = diffusion.diffVolume()
    diff.getVolume(basepath)
    diff.shells()
    diff.makeBvecMeshes()

    # get the dti data
    dti = diffusion.dti()
    dti.load(pathprefix=basepath + '/dti')

    # get the icosahedron ready
    ico = icosahedron.icomesh(m=4)
    ico.get_icomesh()
    ico.vertices_to_matrix()
    diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)

    # mask is available in diff but whatever
    #mask = load(basepath + "/nodif_brain_mask.nii.gz")

    # these are all the voxels
    i, j, k = np.where(diff.mask.get_fdata() == 1)
    voxels = np.asarray([i, j, k]).T

    # pick 50000 random for training from first 500000 inds
    training_inds = random.sample(range(700000), N_train)
    test_inds = random.sample(range(500000, 700000), N_test)

    #pick straight voxels for testing
    N = N_train + N_test
    all_inds = np.arange(0, N)
    # training_inds = all_inds[0:N_train]
    # test_inds = all_inds[N_train:N]

    train_voxels = np.asarray([i[training_inds], j[training_inds], k[training_inds]]).T
    test_voxels = np.asarray([i[test_inds], j[test_inds], k[test_inds]]).T

    # this is for X_train and X_test
    S0_train, flat_train, signal_train = diff.makeFlat(train_voxels, ico)
    S0_test, flat_test, signal_test = diff.makeFlat(test_voxels, ico)

    I, J, T = d12.padding_basis(ico.m + 1)


    # this should be in correct format
    X_trainp = np.copy(list_to_array_X(S0_train, flat_train))
    X_testp = np.copy(list_to_array_X(S0_test, flat_test))

    X_trainp[np.isinf(X_trainp)] = 0
    X_testp[np.isinf(X_testp)] = 0
    X_trainp[np.isnan(X_trainp)] = 0
    X_testp[np.isnan(X_testp)] = 0

    # get L1,L2 and L3 (training)
    L1_train = []
    L2_train = []
    L3_train = []
    for p in train_voxels:
        L1_train.append(dti.V1.get_fdata()[tuple(p)][0])
        L2_train.append(dti.V1.get_fdata()[tuple(p)][1])
        L3_train.append(dti.V1.get_fdata()[tuple(p)][2])
    L1_test = []
    L2_test = []
    L3_test = []
    for p in test_voxels:
        L1_test.append(dti.V1.get_fdata()[tuple(p)][0])
        L2_test.append(dti.V1.get_fdata()[tuple(p)][1])
        L3_test.append(dti.V1.get_fdata()[tuple(p)][2])

    Y_trainp = list_to_array_Y(L1_train, L2_train, L3_train)
    Y_testp = list_to_array_Y(L1_test, L2_test, L3_test)

    return X_trainp,Y_trainp,X_testp,Y_testp,ico,diff
