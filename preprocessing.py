import nibabel as nib
import diffusion
import icosahedron
import dihedral12 as d12
import torch
import numpy as np

def makeFlat(diffpath,outpath,H):
    """
    This will make a nifti in icosahedron representation of first shell
    :param diffpath: path for diffusion data
    :param outpath: path for output
    :return: S0 nifti and nifti with shape [H,W,D,h,w] where h,w are dimensions of icosahedron space. H and D are cut by one for
    divisibility by 3.
    """

    #initialize everything
    print('loading everything')
    diff = diffusion.diffVolume(diffpath)
    diff.cut_ijk()
    ico = icosahedron.icomesh(m=H - 1)
    diff.makeInverseDistInterpMatrix(ico.interpolation_mesh)
    I, J, T = d12.padding_basis(H=H)
    i, j, k = np.where(diff.mask.get_fdata() > 0)
    voxels = np.asarray([i, j, k]).T

    #get icosahedron output
    print('pushing to icosahedron')
    S0, out = diff.makeFlat(voxels, ico)
    print('padding')
    out = out[:, :, I[0, :, :], J[0, :, :]]

    print('saving')
    S0_flat = np.zeros((diff.mask.get_fdata().shape[0:3] + (S0.shape[-1],)))
    S0_flat[i, j, k] = S0

    diff_flat = np.zeros((diff.mask.get_fdata().shape[0:3] + out.shape[1:]))
    diff_flat[i, j, k, :, :, :] = out

    #save everthing (even the cut diffusion)
    nib.save(diff.vol,diffpath + '/data_cut.nii.gz')
    nib.save(diff.mask,diffpath + '/mask_cut.nii.gz')
    diff_flat = nib.Nifti1Image(diff_flat,diff.vol.affine)
    nib.save(diff_flat,outpath+ '/data_cut_flat.nii.gz')
    S0_flat = nib.Nifti1Image(S0_flat, diff.vol.affine)
    nib.save(S0_flat, outpath + '/S0_cut_flat.nii.gz')

    return S0_flat, diff_flat

def flatten_for_2dconv(input):
    """
    Flatten for 2d convolutions
    :param input: tensor of shape [batch, Nc, Nc, Nc, shells, h, w]
    :return: tensor of shape [batch,Nc^3*shells,h,w]
    """
    Nc=input.shape[1]
    h=input.shape[-2]
    w=input.shape[-1]








