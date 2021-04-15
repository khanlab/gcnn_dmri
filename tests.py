import icosahedron
import nibabel as nib
import os
import numpy as np
import diffusion
import matplotlib.pyplot as plt
import stripy
from mayavi import mlab
import gPyTorch

#we need to perform some test to confirm that each stage of the pipeline is working correctly

#1) Spherical signal to flat icosahedron
#   a) the best way to do this is to genrate diffusion signal on the icosahedron and then see if flat signal is the same

#get bvecs for icosahedron
#generate signal (can be anistropic gaussian
#vertices will match with flat matrix

def makeFakeVol():
    ico=icosahedron.icomesh()
    ico.get_icomesh()
    ico.vertices_to_matrix()

    #make bvec and bval files
    testdir='./data/test/'
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    bvals=1000*np.ones(len(ico.interpolation_mesh.x))
    fbvals=open(testdir+'bvals','w')
    fbvecs=open(testdir+'bvecs','w')

    #write the b=0 line
    fbvals.write(str(0.0)+" ")
    fbvecs.write(str(0.0) + " ")

    for idx,bval in enumerate(bvals):
        #print(bvals[idx])
        fbvals.write(str(bvals[idx])+" ")
        fbvecs.write(str(ico.interpolation_mesh.x[idx])+" ")
    fbvecs.write('\n')
    fbvecs.write(str(0.0) + " ")
    for idx,bval in enumerate(bvals):
        fbvecs.write(str(ico.interpolation_mesh.y[idx])+" ")
    fbvecs.write('\n')
    fbvecs.write(str(0.0) + " ")
    for idx, bval in enumerate(bvals):
        fbvecs.write(str(ico.interpolation_mesh.z[idx])+ " ")

    @np.vectorize
    def gauss(lat,long):
        x,y,z=stripy.spherical.lonlat2xyz(long,lat)
        vec=x[0],y[0],z[0]-1
        dot=np.dot(vec,vec)
        #return np.exp( -dot/2 )
        return np.sin(long)

    signal=np.zeros(len(ico.interpolation_mesh.lats)+1)

    signal[1:]=gauss(ico.interpolation_mesh.lats,ico.interpolation_mesh.lons)

    s=1
    top_faces = [[1, 2], [5, 6], [9, 10], [13, 14], [17, 18]]
    face_groups=[[ 1,  2,  3,  4],
                 [ 5,  6,  7,  8],
                 [ 9, 10, 11, 12],
                 [13, 14, 15, 16],
                 [17, 18, 19, 20]
                 ]
    signal[:]=0
    for c in range(0,5):
        face_group=face_groups[c]
        for f in face_group:
            for p in range(0, len(ico.face_list[f])):
                i = ico.i_list[f][p]
                j = ico.j_list[f][p]
                if np.isnan(i) == 0 & np.isnan(j) == 0:
                    signal[s]=100*c+i*10+j
                s=s+1

    signal=signal.reshape([1,1,1,len(signal)])
    signal_nii=nib.Nifti1Image(signal,affine=np.eye(4))
    nib.save(signal_nii,testdir+'/data.nii.gz')
    mask_nii=nib.Nifti1Image(np.ones([1,1,1]),affine=np.eye(4))
    nib.save(mask_nii,testdir+'/nodif_brain_mask.nii.gz')

    return signal,ico

signal,ico=makeFakeVol()


#######################load fake vol and check#####################
basepath="./data/test"
#load diffusion data
diff=diffusion.diffVolume()
diff.getVolume(basepath)
diff.shells()
diff.makeBvecMeshes()

#get the icosahedron ready
ico1=icosahedron.icomesh()
ico1.get_icomesh()
ico1.vertices_to_matrix()
diff.makeInverseDistInterpMatrix(ico1.interpolation_mesh)

#mask is available in diff but whatever
#mask =load("/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion/nodif_brain_mask.nii.gz")
mask =nib.load(basepath+"/nodif_brain_mask.nii.gz")



#this is for X_train and X_test
S0_train,flat_train,signal_train=diff.makeFlat([[0,0,0]],ico1)

plt.imshow(np.flip(flat_train[0][0],0))

#plt.imshow((flat_train[0][0]))


#confirm that we have spherical guassian
#ax = plt.subplot(projection='3d')
#ax.scatter(ico.interpolation_mesh.x,ico.interpolation_mesh.y,ico.interpolation_mesh.z,s=100,c=signal[0,0,0,1:])
#


#devise something to check the padding
#conv1=gPyTorch.gConv2d()
