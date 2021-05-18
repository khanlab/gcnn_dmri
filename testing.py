import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import diffusion
import icosahedron
import dihedral12 as d12
from dipy.core.sphere import cart2sphere

#matplotlib.use('Agg')

class makeFlatTesting:
    """
    this class is to make a mock signal and then flatten it using the icosahedron to make sure the flattening is
    happening correctly
    """
    def __init__(self,testpath):

        #make the bvectors
        self.N_directions=12*12
        self.theta=np.linspace(0.001,np.pi,12)
        self.phi=np.linspace(0.01,2*np.pi,12)
        self.theta,self.phi=np.meshgrid(self.theta,self.phi)
        self.theta=self.theta.flatten()
        self.phi=self.phi.flatten()
        self.signal = np.zeros([1,1,1,self.N_directions+1])
        self.mask = np.ones([1,1,1])
        self.testpath = testpath

    def makeNSaveSignal(self):
        self.signal[:,:,:,0]=1
        for p in range(1,self.N_directions+1):
            self.signal[:,:,:,p]=self.fxn(self.theta[p-1],self.phi[p-1])

        signalnii=nib.Nifti1Image(self.signal,np.eye(4))
        masknii=nib.Nifti1Image(self.mask,np.eye(4))

        nib.save(signalnii,self.testpath +'data.nii.gz')
        nib.save(masknii,self.testpath+'nodif_brain_mask.nii.gz')

        #save bvecs and bvals
        fbval = open(self.testpath + '/bvals', "w")
        fbval.write(str(0)+" ") #b0 value
        for i in range(0,self.N_directions):
            fbval.write(str(1000) +" " )
        fbval.close()

        fbvec = open(self.testpath + '/bvecs', "w")
        #x bvals
        fbvec.write(str(0) + " ")
        for p in range(0,self.N_directions):
            fbvec.write(str(np.cos(self.phi[p])*np.sin(self.theta[p]))+" " )
        fbvec.write("\n")

        # y bvals
        fbvec.write(str(0) + " ")
        for p in range(0, self.N_directions):
            fbvec.write(str(np.sin(self.phi[p ]) * np.sin(self.theta[p ]))+" " )
        fbvec.write("\n")

        # z bvals
        fbvec.write(str(0) + " ")
        for p in range(0, self.N_directions):
            fbvec.write(str(np.cos(self.theta[p ]))+" " )
        fbvec.write("\n")
        fbvec.close()

    def fxn(self,theta,phi):
        return np.cos(theta)


    def makeFlat(self):
        self.diff=diffusion.diffVolume()
        self.diff.getVolume(self.testpath)
        self.diff.shells()
        self.diff.makeBvecMeshes()

        self.ico=icosahedron.icomesh()
        self.ico.get_icomesh()
        self.ico.vertices_to_matrix()
        self.diff.makeInverseDistInterpMatrix(self.ico.interpolation_mesh)

        self.S, self.flat, self.signal=self.diff.makeFlat([[0,0,0]],self.ico,interp='linear')

        I, J, T = d12.padding_basis(self.ico.m + 1)



        self.flat[0][0]=self.flat[0][0][I[0,:,:],J[0,:,:]]


        #make theta map in flat space

        H=self.ico.m +1
        h=H+1
        w=5*h

        theta = np.zeros([h,w])
        phi=np.zeros([h,w])

        top_faces = [[1, 2], [5, 6], [9, 10], [13, 14], [17, 18]]

        for c in range(0,5):
            face=top_faces[c]
            for top_bottom in face:
                vecs=self.ico.face_list[top_bottom]
                t,p=self.vec2thetaphi(vecs)
                i=self.ico.i_list[top_bottom]
                j=self.ico.j_list[top_bottom]
                i = np.asarray(i).astype(int)
                j = np.asarray(c * h + j + 1).astype(int)
                theta[i,j]=t
                phi[i,j]=p

        self.theta_flat=theta
        self.phi_flate=phi



    def vec2thetaphi(self,vecs):
        theta=[]
        phi=[]
        for v in vecs:
            x=v[0]
            y=v[1]
            z=v[2]

            if z==0:
                th=np.pi/2
            else:
                th=np.arccos( z/np.sqrt(x*x+y*y+z*z))

            if np.isnan(th)==1:
                print(x,y,z)

            if x==0.0 and y>0:
                ph=np.pi/2
            elif x==0.0 and y<0:
                ph=-np.pi/2
            else:
                ph=np.arctan(y/x)
            theta.append(th)
            phi.append(ph)

        return theta, phi

mft=makeFlatTesting('./data/testing/')
mft.makeNSaveSignal()
mft.makeFlat()

#
#
# subnetpath=sys.argv[1]
# subgrndpath=sys.argv[2]
#
#
# print(os.listdir(sys.argv[1]))
#
# V1_grnd=nib.load(subgrndpath + 'dtifit_V1.nii.gz')
# FA_grnd=nib.load(subgrndpath + 'dtifit_FA.nii.gz')
#
#
#
# bdir_mean=[]
# bdir_std=[]
# bdirs=np.asarray(os.listdir(subnetpath))
# bdirs=bdirs.astype(int)
# bdirs.sort()
#
# for bdir in bdirs:
#     mask=nib.load(subnetpath +'/' + str(bdir) +'/' + 'nodif_brain_mask.nii.gz')
#     V1_network=nib.load(subnetpath +'/' + str(bdir) +'/' + 'dtifit_V1.nii.gz')
#     dot = np.abs((V1_network.get_fdata()*V1_grnd.get_fdata()).sum(axis=-1))
#     eps=1e-6
#     dot[np.abs(dot-1)<eps]=1.0
#     dotnii=dot
#     dotnii=1-np.rad2deg(np.arccos(dotnii))/90
#     dotnii=nib.Nifti1Image(dotnii,V1_network.affine)
#     nib.save(dotnii,subnetpath +'/' + str(bdir) +'/' + 'V1_difference_dtifit.nii.gz')
#     #dot=dot[mask.get_fdata()>0]
#     dot=dot[FA_grnd.get_fdata()>0.3]
#     dot= np.rad2deg(np.arccos(dot))
#     bdir_mean.append(dot.mean())
#     bdir_std.append(dot.std())
#
# bdir_mean=np.asarray(bdir_mean)
# bdir_std=np.asarray(bdir_std)
# print(bdir_mean)
# print(bdir_std)
#
#
# plt.figure()
# plt.plot(bdirs,bdir_mean,color='black')
# plt.plot(bdirs,bdir_mean+bdir_std,':',color='black')
# plt.plot(bdirs,bdir_mean-bdir_std,':',color='black')
# plt.ylabel('degrees')
# plt.xlabel('directions')
# plt.savefig('plot')
