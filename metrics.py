import os
import nibabel as nib
import numpy as np
import sys
import matplotlib.pyplot as plt


def save_dot(Anii,Bnii,outpath=None):
    """
    Returns the angle difference between two vectors niis as another nii
    """
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
    #nib.save(dot,outpath)
    return dot
    

def save_diff(Anii,Bnii,outpath):
    """
    Returns the scalar difference between two scalar niis as a nii
    """
    A = Anii.get_fdata()
    B = Bnii.get_fdata()
    diff = np.abs(A-B)
    diff = nib.Nifti1Image(diff,Anii.affine)
    #nib.save(diff,outpath)
    return diff




nsubs=[5,10,15]
sub_path = '/home/u2hussai/projects/ctb-akhanf/u2hussai/predictions_%d/' % nsubs[0]
subjects = os.listdir(sub_path)
subjects_table = np.empty([len(nsubs),len(subjects)])
FA_table=np.empty([len(nsubs),len(subjects),2]) # Nsubs d6 dnet
V1_table=np.empty([len(nsubs),len(subjects),2]) # Nsubs d6 dnet
FA_table[:]=np.nan
V1_table[:]=np.nan


source_path='/home/u2hussai/project/u2hussai/niceData/testing/'

for nsub_index,nsub in enumerate(nsubs):
    mean_net_v1=[]
    mean_dti6_v1=[]
    sub_path = '/home/u2hussai/projects/ctb-akhanf/u2hussai/predictions_%d/' % nsub
    subjects = os.listdir(sub_path)
    for sub_index,sub in enumerate(subjects):
        print(nsub,sub)
        try:
            print('trying')
            mask = nib.load(source_path + sub + '/masks/mask.nii.gz' ).get_fdata()

            #net_path = '/home/u2hussai/scratch/network_predictions_'+str(nsub)+'_subjects/'
            #net_path ='/home/u2hussai/scratch/network_predictions_'+str(nsub)+'_subjects_right-padding_no-corner-zero_correct-theta-padding/'

            gt_path=source_path+sub+'/diffusion/90/dtifit/'
            pred_path=sub_path+sub+'/'
            down_path=source_path+sub+'/diffusion/6/dtifit/'

            V1_6_nii = nib.load(down_path+'/dtifit_V1.nii.gz')
            V1_net_nii = nib.load(pred_path+'/dtifit_network_V1.nii.gz')
            V1_gt_nii = nib.load(gt_path+'/dtifit_V1.nii.gz')

            FA_6_nii = nib.load(down_path+'/dtifit_FA.nii.gz')
            FA_net_nii = nib.load(pred_path+'/dtifit_network_FA.nii.gz')
            FA_gt_nii = nib.load(gt_path+'/dtifit_FA.nii.gz')

            dti6=save_dot(V1_6_nii,V1_gt_nii,sub_path + sub +'/6/V1_6_gt_diff'+str(nsub)+'.nii.gz')
            dtinet=save_dot(V1_net_nii,V1_gt_nii,sub_path + sub +'/6/V1_net_gt_diff'+str(nsub)+'.nii.gz')

            FA6=save_diff(FA_6_nii,FA_gt_nii,sub_path + sub +'/6/FA_6_gt_diff'+str(nsub)+'.nii.gz')
            FAnet=save_diff(FA_net_nii,FA_gt_nii,sub_path + sub +'/6/FA_6_net_diff'+str(nsub)+'.nii.gz')

            FA6=FA6.get_fdata()[mask==1]
            FAnet=FAnet.get_fdata()[mask==1]
            dti6=dti6.get_fdata()[mask==1]
            dtinet=dtinet.get_fdata()[mask==1]
            # print('nsub is '+str(nsub))
            # # print('angle diff mean of net is '+str(np.nanmean(dtinet)))
            # # print('angle diff mean of 6 is '+ str(np.nanmean(dti6)))
            # # print('FA diff mean of net is '+str(np.nanmean(FAnet)))
            # # print('FA doff mean of 6 is '+ str(np.nanmean(FA6)))
            # #mean_net_v1.append(np.nanmean(dtinet))
            # #mean_dti6_v1.append(np.nanmean(dti6))

            FA_table[nsub_index, sub_index,0]=np.nanmean(FA6)
            FA_table[nsub_index,sub_index,1]=np.nanmean(FAnet)
            
            V1_table[nsub_index,sub_index,0]=np.nanmean(dti6)
            V1_table[nsub_index,sub_index,1]=np.nanmean(dtinet)

            subjects_table[nsub_index,sub_index]=sub

        except:
           print('There is some missing data')


    print(FA_table)
    print(V1_table)            

np.save('/home/u2hussai/project/u2hussai/no_of_subjects/subject_list.npy',subjects)
np.save('/home/u2hussai/project/u2hussai/no_of_subjects/FA_table.npy',FA_table)
np.save('/home/u2hussai/project/u2hussai/no_of_subjects/V1_table.npy',V1_table)
        
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