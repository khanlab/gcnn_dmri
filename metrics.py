import os
import nibabel as nib
import numpy as np
import sys
import matplotlib.pyplot as plt


def save_dot(Anii,Bnii,outpath=None):
    A=Anii.get_fdata()
    B=Bnii.get_fdata()
    A = A.reshape([-1, 3])
    B = B.reshape([-1, 3])
    dot = A*B
    dot = np.sum(dot,-1)
    dot = np.rad2deg(np.arccos(dot))
    for i in range(0,dot.shape[0]):
        if dot[i] > 90:
            dot[i] = 180 - dot[i]
    dot = dot.reshape(Anii.shape[0:3])
    dot = nib.Nifti1Image(dot,Anii.affine)
    return dot
    #nib.save(dot,outpath)

def save_diff(Anii,Bnii,outpath):
    A = Anii.get_fdata()
    B = Bnii.get_fdata()
    diff = np.abs(A-B)
    diff = nib.Nifti1Image(diff,Anii.affine)
    diff.save(diff,outpath)



sub_path = '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/'
subjects = os.listdir(sub_path)
bdirs = os.listdir(sub_path + subjects[0] +'/')
#compute angle difference for all angles
diff_directions=[]
# for bdir in bdirs:
#     mean_sub = []
#     for sub in subjects:
#         mask_nii = nib.load(sub_path + sub + '/' + '6' + '/mask1_cut_pad.nii.gz' )
#         V1_nii = nib.load(sub_path + sub + '/'+ bdir + '/dtifit_V1.nii.gz')
#         V1_gt_nii = nib.load(sub_path + sub + '/'+ '90' + '/dtifit_V1.nii.gz')
#         out = save_dot(V1_nii,V1_gt_nii)
#         out = out.get_fdata()
#         mask = mask_nii.get_fdata()
#         out = out[mask==1]
#         mean_sub.append(np.nanmean(out))
#     diff_directions.append(np.mean(mean_sub))


mean_net_v1=[]
mean_dti6_v1=[]
for sub in subjects:

    mask = nib.load(sub_path + sub + '/6/mask2_cut_pad.nii.gz').get_fdata()

    V1_6_nii = nib.load(sub_path + sub +'/6/dtifit_V1.nii.gz')
    V1_net_nii = nib.load(sub_path + sub +'/6/dtifit_network_V1.nii.gz')
    V1_gt_nii = nib.load(sub_path + sub +'/90/dtifit_V1.nii.gz')

    FA_6_nii = nib.load(sub_path + sub +'/6/dtifit_FA.nii.gz')
    FA_net_nii = nib.load(sub_path + sub + '/6/dtifit_network_FA.nii.gz')
    FA_gt_nii = nib.load(sub_path + sub +'/6/dtifit_FA.nii.gz')

    save_dot(V1_6_nii,V1_gt_nii,sub_path + sub +'/6/V1_6_gt_diff.nii.gz')
    save_dot(V1_net_nii,V1_gt_nii,sub_path + sub +'/6/V1_net_gt_diff.nii.gz')

    net = nib.load(sub_path + sub +'/6/V1_net_gt_diff.nii.gz')
    dti6 = nib.load(sub_path + sub +'/6/V1_6_gt_diff.nii.gz')

    FA_gt = FA_gt_nii.get_fdata()
    net = net.get_fdata()
    dti6 = dti6.get_fdata()
    
    net=net[mask==1]
    dti6=dti6[mask==1]
    mean_net_v1.append(np.nanmean(net))
    mean_dti6_v1.append(np.nanmean(dti6))
#dti6=dti6[FA_gt>0.2]

#plt.hist(net,100,histtype='step',color='orange')
#plt.hist(dti6,100,histtype='step',color='blue')
#plt.hist(ne,100,histtype='step',color='black')

# #FA_6 = FA_6_nii.get_fdata()
# FA_net = FA_net_nii.get_fdata()

# #FA_diff_6=abs(FA_6 - FA_gt)
# FA_diff_net=abs(FA_net - FA_gt)

# #FA_diff_6=FA_diff_6[FA_gt > 0.2]
# FA_diff_net=FA_diff_net[mask== 1]


#subnetpath=sys.argv[1]
#subgrndpath=sys.argv[2]


#print(os.listdir(sys.argv[1]))

#V1_grnd=nib.load(subgrndpath + 'dtifit_V1.nii.gz')
#FA_grnd=nib.load(subgrndpath + 'dtifit_FA.nii.gz')





# V1_grnd=nib.load('/home/uzair/Desktop/6/dtifit_V1.nii.gz')
# FA_grnd=nib.load('/home/uzair/Desktop/6/dtifit_FA.nii.gz')
#
#
#
# #bdir_mean=[]
# #bdir_std=[]
# #bdirs=np.asarray(os.listdir(subnetpath))
# #bdirs=bdirs.astype(int)
# #bdirs.sort()
#
# #for bdir in bdirs:
#     #mask=nib.load(subnetpath +'/' + str(bdir) +'/' + 'nodif_brain_mask.nii.gz')
#     #V1_network=nib.load(subnetpath +'/' + str(bdir) +'/' + 'dtifit_V1.nii.gz')
#
# V1_network=nib.load('/home/uzair/Desktop/6/dtifit6_V1.nii.gz')
#
#
# dot = np.abs((V1_network.get_fdata()*V1_grnd.get_fdata()).sum(axis=-1))
# eps=1e-6
# dot[np.abs(dot-1)<eps]=1.0
# dotnii=dot
# dotnii=1-np.rad2deg(np.arccos(dotnii))/90
# dotnii=nib.Nifti1Image(dotnii,V1_network.affine)
# nib.save(dotnii,'/home/uzair/Desktop/6/dtifit6_difference.nii.gz')
# #dot=dot[mask.get_fdata()>0]
# dot=dot[FA_grnd.get_fdata()>0.3]
# dot= np.rad2deg(np.arccos(dot))
# bdir_mean.append(dot.mean())
# bdir_std.append(dot.std())
#
# bdir_mean=np.asarray(bdir_mean)
# bdir_std=np.asarray(bdir_std)
# print(bdir_mean)
# print(bdir_std)


# plt.figure()
# plt.plot(bdirs,bdir_mean,color='black')
# plt.plot(bdirs,bdir_mean+bdir_std,':',color='black')
# plt.plot(bdirs,bdir_mean-bdir_std,':',color='black')
# plt.ylabel('degrees')
# plt.xlabel('directions')
# plt.savefig('plot')