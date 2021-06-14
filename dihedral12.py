import numpy as np
import torch


"""
Some useful functions for the dihedral group
"""

def unproject(weights):
    """
    Goes from 1d configuration to 3x3 configuration
    :param weights in 1d configuration with shape [7,N], center weight at end
    :return: [3,3,N] weights
    """
    N=weights.shape[-1]
    kernel=torch.zeros([3,3,N],requires_grad=False)
    for i in range(0,N):
        kernel[1, 2,i] = weights[0,i]
        kernel[0, 2,i] = weights[1,i]
        kernel[0, 1,i] = weights[2,i]
        kernel[1, 0,i] = weights[3,i]
        kernel[2, 0,i] = weights[4,i]
        kernel[2, 1,i] = weights[5,i]
        kernel[1, 1,i] = weights[6,i]
    kernel[0, 0, :]=  7
    kernel[2, 2, :] = 7
    return kernel

def rotate(weights, angle):
    """
    Rotates weights by angle in 1d configuration
    :param weights: weights in 1d configuration with shape [7,N] (center weight at end)
    :param angle: integer representing how many multiples of theta to rotate by
    :return: rotated weights by angle
    """
    if angle is None:
        angle = 1
    if int(angle) is False:
        raise ValueError("angles need to be ints")
    else:
        angle = angle % 6

    weights_n = weights.clone()
    weights_n = weights_n.view([7,-1])
    for i in range(0,weights_n.shape[-1]):
        weights_n[0:6,i] = torch.roll(weights_n[0:6,i], angle)
    return weights_n

def reflect(weights, axis):
    """
    Reflects along an axis in 1d configuration
    :param weights: weights in 1d configuration (center weight at end)
    :param axis: axis of reflection, represented as integer, 0 is x-axis, 1 is middle of first edge, 2 is first
    vertex etc..
    :return: reflected weights along axis
    """
    if axis is None:
        axis = 0
    if int(axis) is False:
        raise ValueError("axis need to be ints")
    else:
        axis = axis % 6

    # first reflect on x-axis and then rotate
    weights_n=weights.clone()
    weights_n = weights_n.view([7, -1])
    for i in range(0,weights_n.shape[-1]):
        temp_weights = weights_n[1:3,i].clone()
        weights_n[1:3,i] = torch.roll(weights_n[4:6,i], 1)
        weights_n[4:6,i] = torch.roll(temp_weights, 1)
    return rotate(weights_n,axis)

def expand_scalar(weights):
    """
    Takes weights with shape [7,N] (center weight at end) and returns rotated and reflected versions of same weight
    in shape [3,3,N*12].
    :param weights with shape [7,N] (center weight at end)
    :return: rotated and reflected weights [3,3,N*12], first 12 are rotated/reflected versions of first weight and so
    on...,
    """
    if len(weights.shape) == 1:
        weights = weights.view(7, 1)

    N=weights.shape[-1]
    weights_e=torch.zeros([7,N,12])
    for angle in range(0,6):
        weights_e[:,:,angle]=rotate(weights,angle)
    for axis in range (0,6):
        weights_e[:,:,axis+6] = reflect(weights, axis)
    return unproject(weights_e.view(7,N*12))

def rotate_deep(weights,angle):
    """
    Takes weights with shape [7,12,N] (center weight at end) and returns composition of existing group action (the
    dimension with size 12) and an inverse rotation by angle
    :param weights: weights with shape [7,12,N] (center weight at end)
    :param angle: integer representing how many multiples of theta to rotate by
    :return: rotated weights with shape [7,12,N]
    """

    N=weights.shape[-1]
    weights_n=weights.clone()

    idx_rot=np.arange(6)
    idx_rot = (idx_rot - angle)%6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_rot,idx_ref))
    weights_n[:,:,:]=weights[:,idx,:]
    weights_n=weights_n.view(7,12*N)
    weights_n= rotate(weights_n,angle)
    weights_n=weights_n.view(7,12,N)
    return weights_n

def reflect_deep(weights, axis):
    """
    Takes weights with shape [7,12,N] (center weight at end) and returns composition of existing group action (the
    dimension with size 12) and an inverse reflection along axis
    :param weights: weights with shape [7,12,N] (center weight at end)
    :param axis: axis of reflection, represented as integer, 0 is x-axis, 1 is middle of first edge, 2 is first
    vertex etc..
    :return: reflected weights with shape [7,12,N]
    """
    N=weights.shape[-1]
    weights_n=weights.clone()
    idx_rot = np.arange(6)
    idx_rot = (axis-idx_rot) % 6
    idx_ref = idx_rot + 6
    idx = np.concatenate((idx_ref, idx_rot)) #note how reflections and rotations get swapped, this is intentional
    weights_n[:,:,:] = weights[:,idx,:]
    weights_n=weights_n.view(7,12*N)
    weights_n= reflect(weights_n,axis)
    return weights_n.view([7,12,N])

def expand_regular(weights):
    """
    Takes weights with shape [7,12,N] (center weight at end) and returns composition of existing group action (the
    dimension with size 12) and all other 12 group actions. Note that the 12 group actions are inverses.
    :param weights: weights with shape [7,12,N] (center weight at end)
    :return:  rotated and reflected weights [3,3,12*N*12]
    """
    if len(weights.shape)==2:
        weights=weights.view(7,12,1)

    N=weights.shape[-1]
    weights_e = torch.zeros([7,12,N,12])
    for angle in range(0,6):
        weights_e[:,:,:,angle]= rotate_deep(weights,angle)
    for axis in range(0,6):
        weights_e[:, :, :,axis+6] = reflect_deep(weights, axis)
    return unproject(weights_e.view(7,12*N*12))

def padding_basis(H):
    """
    Creates bases (2 or 3 dimensional index arrays) to pad initial and deep layers
    :param input: input image of size [5*(H+1), H+1, .] the last dimensions exists and is 12 if deep=1
    :param deep: initial layer deep = 0, deep layer deep=1
    :return: index arrays to be used for padding
    """

    strip_xy=np.arange(0,H-1)
    h = H + 1
    w = 5 * (H + 1)

    #I, J, T = np.meshgrid(np.arange(0, h), np.arange(0, w),np.arange(0, 12), indexing='ij')
    #I_out, J_out, T_out = np.meshgrid(np.arange(0, h), np.arange(0, w),np.arange(0, 12), indexing='ij')

    T, I, J = np.meshgrid(np.arange(0, 12),np.arange(0, h), np.arange(0, w), indexing='ij')
    T_out ,I_out, J_out = np.meshgrid(np.arange(0, 12),np.arange(0, h), np.arange(0, w), indexing='ij')

    for t in range(0,12):
        if t<=5:
            t_left=t
            t_right_left=(t+1) % 6
            t_right_bottom=(2-t) % 6
            t_right_right=(2-t) % 6
            t_right_top= (t-1) % 6

        if t>5:
            t_left = t
            t_right_left = (((t-6) + 1) % 6) + 6
            t_right_bottom = ((2 - (t-6)) % 6) + 6
            t_right_right = ((2 - (t-6)) % 6) + 6
            t_right_top = (((t -6) - 1) % 6) + 6

        for c in range(0, 5):
            #bottom padding
            c_left = c
            x_left = strip_xy
            y_left = -1
            i_left, j_left = xy2ind(H, c_left, x_left, y_left)
            #print(i_left,j_left)

            c_right = (c + 2) % 5
            x_right = H - 2
            y_right = strip_xy
            i_right, j_right = xy2ind(H, c_right, x_right, y_right)
            #print(i_right,j_right)


            I_out[t_left, i_left, j_left] = I[ t_right_bottom, i_right, j_right]
            J_out[t_left, i_left, j_left] = J[ t_right_bottom, i_right, j_right]
            T_out[t_left, i_left, j_left] = T[ t_right_bottom, i_right, j_right]

            # top padding
            strip_xy_top = np.arange(0, H)
            c_left = c
            x_left = np.arange(0, H)
            y_left = H - 1
            i_left, j_left = xy2ind(H, c_left, x_left, y_left)
            #print(i_left,j_left)

            c_right = (c + 1) % 5
            x_right = 0
            y_right = H-1-strip_xy_top
            i_right, j_right = xy2ind(H, c_right, x_right, y_right)
            #print(i_right, j_right)
            #print('---------------------------------')

            I_out[t_left,i_left, j_left] = I[t_right_top,i_right, j_right ]
            J_out[t_left,i_left, j_left] = J[t_right_top,i_right, j_right ]
            T_out[t_left,i_left, j_left] = T[t_right_top,i_right, j_right ]

        #right padding
        strip_xy_right=np.arange(0,H)
        c_left = 4
        x_left = H-1
        y_left = strip_xy_right
        i_left, j_left = xy2ind(H, c_left, x_left, y_left)
        #print(i_left,j_left)

        c_right = (c_left + 3) % 5
        x_right = strip_xy_right
        y_right = 0
        i_right, j_right = xy2ind(H, c_right, x_right, y_right)
        #print(i_right,j_right)


        I_out[t_left, i_left, j_left] = I[t_right_right, i_right, j_right]
        J_out[t_left, i_left, j_left] = J[t_right_right, i_right, j_right]
        T_out[t_left, i_left, j_left] = T[t_right_right, i_right, j_right]

        # left padding #IMPORTANT: do inner ones need to be padded in each convolution?
        c_left = 0
        x_left = -1
        y_left = strip_xy
        i_left, j_left = xy2ind(H, c_left, x_left, y_left)
        # print(i_left,j_left)

        c_right = (c_left - 1) % 5
        x_right = H - 2 - strip_xy
        y_right = H - 2
        i_right, j_right = xy2ind(H, c_right, x_right, y_right)
        # print(i_right, j_right)
        # print('---------------------------------')

        I_out[t_left, i_left, j_left] = I[t_right_left, i_right, j_right]
        J_out[t_left, i_left, j_left] = J[t_right_left, i_right, j_right]
        T_out[t_left, i_left, j_left] = T[t_right_left, i_right, j_right]

    return I_out, J_out, T_out

def pad(out,I,J,T):
    #if deep==0:
    #    out = out[:,:,I[:,:,0],J[:,:,0]] #last dimensions is theta
    #    return out
    #if deep==1:
        #have to figure out how to reshape things here
    shape=list(out.shape) #keep orig shape
    input_dim=int(shape[1]/12) #get input dimension
    newshape = [shape[0],input_dim,12]+ shape[-2:]
    out_n=out.view(newshape)
    out_n=out_n[:,:,T,I,J]
    return out_n.view(shape)


def basis_expansion(deep):
    """
    Returns the expanded filter basis
    :param deep: whether it is a scalar (deep=0) or a regular layer (deep=1)
    :return: expanded filter basis shape [12,3,3] (deep=0) and two bases both with shape [12,12,3,3] (deep=1)
    """

    if deep==0:
        basis = torch.arange(0, 7)
        basis_e= expand_scalar(basis).permute([-1, 0, 1]).view(12, 3, 3)
        return basis_e.type(torch.long)

    if deep==1:
        basis=torch.arange(0,7)
        basis_t=torch.arange(0,12)
        basis,basis_t=torch.meshgrid(basis,basis_t)

        #expand_regular returns the phi dimension (inverse group action) as last, we want the theta dimension to be
        # last this is why the permute is there
        basis=expand_regular(basis).permute([-1,0,1]).view(12,12,3,3).permute([1,0,-2,-1])
        basis_t=expand_regular(basis_t).permute([-1,0,1]).view(12,12,3,3).permute([1,0,-2,-1])

        return basis.type(torch.long), basis_t.type(torch.long)

def apply_weight_basis(weights,basis=None,basis_t=None):
    """
    Returns expanded weights, i.e., expanded basis applied to the weights
    :param weights: weights
    :param basis: basis for first layer
    :param basis_t: basis for deep layer
    :return: expanded weights (with orientation layers stacked
    """
    Cout=weights.shape[0]
    Cin=weights.shape[1]
    if basis_t is None:
        weights_e=torch.cat([weights,torch.zeros(weights.shape[0:2]+(1,)).cuda(weights.device.index)],dim=-1)[:,:,basis]
        #we have to stack the orientation channels (!)
        weights_e=weights_e.permute(0,2,1,3,4).reshape(Cout*12,Cin,3,3)
        return weights_e
    else:
        weights_e = torch.cat([weights, torch.zeros(weights.shape[0:2] + (12,1)).cuda(weights.device.index)], dim=-1)[:,:, basis_t, basis]
        #we have to stack the orientation channels (!)
        weights_e = weights_e.permute(0,3,1,2,4,5).reshape(Cout*12,Cin*12,3,3)
        return weights_e


def xy2ind(H,c,x,y):
    """
    We define a local x,y basis for each chart which are placed next to each other so the coloumns point along the
    x-axis and the row points opposite to the y-axis. This function converts between each chart, c, coordinates, x,
    y to image inds.
    :param H: width of image is 5*(H+1), height is (H+1)
    :param c: chart index, between 1 to 5 (inclusive)
    :param x: x coordinate in chart basis, between -1 and H-1 (inclusive)
    :param y: y coordinate in chart basis, between -1 and H-1 (inclusive)
    :return:
    """
    i=H-y-1
    j=c*(H+1)+1+x
    return i,j

def bias_basis(out_channels):
    """
    Simple function that creates a basis to copy bias for output channels 12 times each for orientation channels
    :param out_channels: number of output channels
    :return: basis for expanding basis
    """
    N = out_channels
    basis_e = torch.zeros(N * 12, dtype=torch.long)
    for i in range(0, N):
        for j in range(0, 12):
            basis_e[12 * i + j] = i
    return basis_e
