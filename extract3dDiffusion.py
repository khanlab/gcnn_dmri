import numpy as np
import copy
import diffusion
import icosahedron
import random
import dihedral12 as d12
from nibabel import load
import nibabel as nib
import torch
import re
import os

def loader_3d(diffpath,H):
    diff=diffusion.diffVolume(diffpath)
    diff.cut_ik()
    ico = icosahedron.icomesh(m=H - 1)
    diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)
    I, J, T = d12.padding_basis(H=H)
    i, j, k = np.where(diff.mask.get_fdata() > 0)
    voxels = np.asarray([i, j, k]).T

    S0,out=diff.makeFlat(voxels,ico)
    out= out[:,:,I[0,:,:],J[0,:,:]]

    S0_flat=np.zeros((diff.mask.get_fdata().shape[0:3]+(S0.shape[-1],)))
    S0_flat[i,j,k]=S0
    S0_flat=torch.from_numpy(S0_flat).contiguous()

    diff_flat=np.zeros((diff.mask.get_fdata().shape[0:3]+out.shape[1:]))
    diff_flat[i,j,k,:,:,:]=out
    diff_flat=torch.from_numpy(diff_flat).contiguous()

    return S0_flat.unfold(0,3,3).unfold(1,3,3).unfold(2,3,3), diff_flat.unfold(0,3,3).unfold(1,3,3).unfold(2,3,3)











#
# def list_to_array_X(flats,I,J):
#     # convert the lists to arrays and also normalize the data to make attenutations brighter
#     N = len(flats)
#     shells = len(flats[0])
#     h = len(flats[0][0])
#     w = len(flats[0][0][0])
#     out = np.zeros([N, shells, h, w])
#     for p in range(0, N):
#         for s in range(0, shells):
#             temp = copy.deepcopy(flats[p][s][I[0, :, :], J[0, :, :]])
#             out[p, s, :, :] = temp  # notice no normalizaton applied for now
#     return out
#
#
# def dti_to_array_Y(dti, voxels):  # this will need to made more general
#
#     out = np.zeros([len(voxels), 13])
#
#     for i, p in enumerate(voxels):
#         out[i, 0] = dti.FA.get_fdata()[tuple(p)]
#         out[i, 1] = dti.L1.get_fdata()[tuple(p)]
#         out[i, 2] = dti.L2.get_fdata()[tuple(p)]
#         out[i, 3] = dti.L3.get_fdata()[tuple(p)]
#         out[i, 4:7] = dti.V1.get_fdata()[tuple(p)][:]
#         out[i, 7:10] = dti.V2.get_fdata()[tuple(p)][:]
#         out[i, 10:13] = dti.V3.get_fdata()[tuple(p)][:]
#
#     return out
#
#
#
#
#
# class chunk_loader:
#     def __init__(self,path):
#         self.path = path
#     def load(self,cut=50):
#         files=os.listdir(self.path+'data_chunks/')
#         files_dti=os.listdir(self.path+'dti_chunks/')
#         X=[] #input
#         Y=[] #output
#         S0=[]
#         for idx,file in enumerate(files):
#             if 'data_flat' in file:
#                 parse=re.split(r'[_.]',file)
#                 percent=int(parse[-3])
#                 if percent >= cut:
#                     print('loading: '+ self.path+'data_chunks/'+file)
#                     X.append(nib.load(self.path+'data_chunks/'+file).get_fdata())
#                     thisstring=''
#                     for par in range(2,len(parse)-2):
#                         thisstring=thisstring+"_"+parse[par]
#                     print('loading: ' +self.path+'dti_chunks/'+'dti_all'+thisstring)
#                     Y.append(nib.load(self.path+'dti_chunks/'+'dti_all'+thisstring+'.nii.gz').get_fdata())
#                     S0.append(nib.load(self.path+'data_chunks/data_S0'+thisstring+'.nii.gz').get_fdata())
#                     print('\n')
#         return np.stack(X,axis=0), np.stack(S0,axis=0), np.stack(Y,axis=0)
#
#
# class extractor3d:
#     def __init__(self,datapath,dtipath,outpath,interp='inverse_distance'):
#         self.datapath = datapath
#         self.dtipath = dtipath
#         self.outpath = outpath
#         self.interp=interp
#
#         #diff stuff
#         self.diff=diffusion.diffVolume()
#         self.diff.getVolume(datapath)
#         self.diff.shells()
#         self.diff.makeBvecMeshes()
#
#         #dti stuff
#         self.dti=diffusion.dti()
#         self.dti.load(pathprefix=self.dtipath)
#
#         #ico stuff
#         self.ico = icosahedron.icomesh(m=10)
#         self.ico.get_icomesh()
#         self.ico.vertices_to_matrix()
#         self.diff.makeInverseDistInterpMatrix(self.ico.interpolation_mesh)
#
#         self.N_split = [] #split volume into [N_split, N_split, N_split]
#
#     def splitNsave(self,chunk_size=9):
#
#         #we want our grid to be
#         i,j,k=np.where(self.diff.mask.get_fdata()>0)
#
#         #dimensions of brain
#         deltai=max(i)-min(i)
#         deltaj=max(j)-min(j)
#         deltak=max(k)-min(k)
#
#         #number of chunks needeed
#         chunksi = -(-deltai//chunk_size)
#         chunksj = -(-deltaj // chunk_size)
#         chunksk = -(-deltak // chunk_size)
#
#         sizestring='i-'+str(min(i)) + '-'+ str(max(i)) + '_j-'+ str(min(j)) + '-'+ str(max(j)) +'_k-'+ str(min(k)) + \
#                    '-'+ str(max(k)) + '_'
#
#         #total chunks
#         H = self.ico.m + 1
#         w = 5 * (H + 1)
#         h = H + 1
#         N_chunks=chunksi*chunksj*chunksk
#         N_flat = chunk_size * chunk_size * chunk_size
#         chunks_S0=np.zeros([N_chunks, chunk_size, chunk_size, chunk_size, len(self.diff.inds[0])])
#         chunks_data = np.zeros([N_chunks, chunk_size, chunk_size, chunk_size, h, w])
#         chunks_dti = np.zeros([N_chunks,chunk_size,chunk_size,chunk_size,13])
#
#         I, J, T = d12.padding_basis(self.ico.m + 1)
#         ip, jp, kp = np.meshgrid(np.arange(0, chunk_size), #these are for the chunk
#                                  np.arange(0, chunk_size),
#                                  np.arange(0, chunk_size))
#
#         #extract chunks
#         if not os.path.exists(self.outpath + '/data_chunks'):
#             os.makedirs(self.outpath + '/data_chunks')
#         if not os.path.exists(self.outpath + '/dti_chunks'):
#             os.makedirs(self.outpath + '/dti_chunks')
#
#         chnk_ind = 0
#         for ii in range(min(i),max(i),chunk_size):
#             for jj in range(min(j), max(j), chunk_size):
#                 for kk in range(min(k), max(k), chunk_size):
#                     flat = np.zeros([N_flat, 3, h, w])
#                     S0 = np.zeros([N_flat,len(self.diff.inds[0])])
#                     print('Working on chunk id:' + str(chnk_ind))
#                     il, jl, kl = np.meshgrid(np.arange(ii, ii + chunk_size), #these are for the larger image
#                                              np.arange(jj, jj + chunk_size),
#                                              np.arange(kk, kk + chunk_size))
#                     voxels=np.asarray([il.flatten(),jl.flatten(),kl.flatten()]).T
#
#                     mask = self.diff.mask.get_fdata()[il.flatten(), jl.flatten(), kl.flatten()]
#                     perc_filled=100*len(mask[mask>0])/len(mask)
#                     fill=mask>0
#                     if any(fill): #only evaluate at non-zero points
#                         S0_temp, flat[fill,:,:,:], signal = self.diff.makeFlat(voxels[fill], self.ico,
#                                                                               interp=self.interp)
#                         S0_temp=np.asarray(S0_temp)
#                         S0[fill,:]=S0_temp[:,0,:]
#
#                     chunks_data[chnk_ind, ip.flatten(), jp.flatten(), kp.flatten(), :, :] = flat[:, 0, :, :]
#                     chunk_nii=nib.Nifti1Image(chunks_data[chnk_ind,:,:,:,:,:],self.diff.vol.affine)
#                     nib.save(chunk_nii,self.outpath+'data_chunks/data_flat_'+sizestring+str(chnk_ind)+'_'+str(int(
#                         perc_filled))+'.nii.gz')
#
#                     chunks_S0[chnk_ind, ip.flatten(), jp.flatten(), kp.flatten(), :] = S0[:, :]
#                     chunk_nii=nib.Nifti1Image(chunks_S0[chnk_ind,:,:,:,:],self.diff.vol.affine)
#                     nib.save(chunk_nii,self.outpath+'data_chunks/data_S0_'+sizestring+str(chnk_ind)+'_'+str(int(
#                         perc_filled))+'.nii.gz')
#
#                     #chunks_dti[chnk_ind,il.flatten(),jl.flatten(),kl.flatten(),:]=dti_to_array_Y(self.dti,voxels)
#                     chunks_dti[chnk_ind,ip.flatten(), jp.flatten(), kp.flatten(), :] = dti_to_array_Y(self.dti, voxels)
#                     chunk_nii = nib.Nifti1Image(chunks_dti[chnk_ind, :, :, :, :], self.diff.vol.affine)
#                     nib.save(chunk_nii, self.outpath+'dti_chunks/dti_all_' +sizestring+ str(chnk_ind)+'_'+str(int(
#                         perc_filled))
#                              +'.nii.gz')
#                     chnk_ind+=1
#
#     # def splitNsave(self,chunk_size=29):
#     #     sz=self.diff.vol.shape
#     #     print(sz)
#     #     H=self.ico.m + 1
#     #     w = 5 * (H + 1)
#     #     h = H + 1
#     #     N_chunks=int(sz[0]/chunk_size)*int(sz[1]/chunk_size)*int(sz[2]/chunk_size)
#     #     chunks=np.zeros([N_chunks,chunk_size,chunk_size,chunk_size,h,w])
#     #     chunks_dti=np.zeros([N_chunks,chunk_size,chunk_size,chunk_size,13])
#     #     chnk_ind=0
#     #     I, J, T = d12.padding_basis(self.ico.m + 1)
#     #     ii, jj, kk = np.meshgrid(np.arange(0, chunk_size),
#     #                              np.arange(0, chunk_size),
#     #                              np.arange(0, chunk_size))
#     #     N_flat=chunk_size*chunk_size*chunk_size
#     #     for i in range(0,sz[0],chunk_size): #extract the chunks and convert them
#     #         for j in range(0,sz[1],chunk_size):
#     #             for k in range(0,sz[2],chunk_size):
#     #                 flat = np.zeros([N_flat, 3, h, w])
#     #                 print('Working on chunk id:'+str(chnk_ind))
#     #                 il,jl,kl=np.meshgrid(np.arange(i,i+chunk_size),
#     #                                      np.arange(j,j+chunk_size),
#     #                                      np.arange(k,k+chunk_size))
#     #                 print(np.arange(i,i+chunk_size))
#     #                 print(np.arange(i,i+chunk_size))
#     #                 print(np.arange(i,i+chunk_size))
#     #                 voxels=np.asarray([il.flatten(),jl.flatten(),kl.flatten()]).T
#     #                 mask=self.diff.mask.get_fdata()[il.flatten(),jl.flatten(),kl.flatten()]
#     #                 perc_filled=100*len(mask[mask>0])/len(mask)
#     #                 fill=mask>0
#     #                 if any(fill):
#     #                     S0, flat[fill,:,:,:], signal = self.diff.makeFlat(voxels[fill], self.ico, interp=self.interp)
#     #                 flat=list_to_array_X(flat,I,J)
#     #                 #chunks[chnk_ind,il.flatten(),jl.flatten(),kl.flatten(),:,:]=flat[:,0,:,:]
#     #                 chunks[chnk_ind, ii.flatten(), jj.flatten(), kk.flatten(), :, :] = flat[:, 0, :, :]
#     #                 chunk_nii=nib.Nifti1Image(chunks[chnk_ind,:,:,:,:,:],self.diff.vol.affine)
#     #                 #np.save(self.outpath+'data_'+str(chnk_ind)+'.npy')
#     #                 nib.save(chunk_nii,self.outpath+'/data_flat_'+str(chnk_ind)+'_'+str(int(perc_filled))+'.nii.gz')
#     #                 #chunks_dti[chnk_ind,il.flatten(),jl.flatten(),kl.flatten(),:]=dti_to_array_Y(self.dti,voxels)
#     #                 chunks_dti[chnk_ind,ii.flatten(), jj.flatten(), kk.flatten(), :] = dti_to_array_Y(self.dti, voxels)
#     #                 chunk_nii = nib.Nifti1Image(chunks_dti[chnk_ind, :, :, :, :], self.diff.vol.affine)
#     #                 nib.save(chunk_nii, self.outpath+'/dti_all_' + str(chnk_ind)+'_'+str(int(perc_filled)) +'.nii.gz')
#     #                 chnk_ind+=1
#
