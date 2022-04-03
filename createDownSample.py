import diffusion
import icosahedron
import os
import sys
import nibabel as nib
import numpy as np
import os
 
subjects_path = "/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/"    
#training_path = '/home/u2hussai/scratch/dtitraining/downsample_cut_pad/'
#prediction_path = '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/'

training_path = '/home/u2hussai/project/u2hussai/scratch_14Sept21/dtitraining/downsample_cut_pad/'
prediction_path = '/home/u2hussai/project/u2hussai/scratch_14Sept21/dtitraining/prediction_cut_pad/'
prediction_path_other='/home/u2hussai/scratch/prediction_other/'


Ntrain = int(sys.argv[1])
Ntest = int(sys.argv[2])

sub_list = os.listdir(subjects_path) #get subject list
for s,sub in enumerate(sub_list): #add the sub- prefix
    sub_list[s] = 'sub-' + sub

#check for overlap
current_training_subs = os.listdir(training_path)
current_prediction_subs = os.listdir(prediction_path) + os.listdir(prediction_path_other)


exclude_subjects = current_prediction_subs + current_prediction_subs
final_subjects = np.asarray([s for s in sub_list if s not in exclude_subjects])

#select train and test random subjects
inds=np.random.randint(0,len(final_subjects),Ntrain+Ntest)
final_subjects = final_subjects[inds]

final_subjects=['sub-192439']
print(final_subjects)
#save
for s in range(0,Ntrain): #downsample train subjects
    sub = final_subjects[s]
    try:
        diff=diffusion.diffVolume(subjects_path + sub[4:] + "/T1w/Diffusion/")
        diff.shells()
        diff.downSample(training_path,sub)
    except:
        print('Error encountered, likely missing data')
for s in range(Ntrain,Ntrain+Ntest): #downsample test subjects
    # try:
    sub = final_subjects[s]
    diff=diffusion.diffVolume(subjects_path + sub[4:] + "/T1w/Diffusion/")
    diff.shells()
    diff.downSample(prediction_path_other,sub)
    #     diff.downSample('/home/u2hussai/scratch/',sub)
    # except:
    #     print('Error encountered, likely missing data')


# diffpath = sys.argv[1]
# outpath = sys.argv[2]
# subjectid = sys.argv[3]


# #diffpath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
# #dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"
# #outpath='/home/u2hussai/scratch/'

# if not os.path.exists(outpath):
#     os.makedirs(outpath)
  
# diff=diffusion.diffVolume()
# diff.getVolume(diffpath)
# diff.shells()
# diff.downSample(outpath,subjectid)
