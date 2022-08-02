#Here we look at many subjects in input_path and randomly
#select N_subs subjects which are not present in out_path
#a list of subjects is created in subjects_lists folder 
#in same directory as this script


import os
import nibabel as nib

#get all subjects from here
input_path='/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/'

#check here for existing subjects or set as None (this is crucial to prevent data leakage!)
#out_path=None#'/home/u2hussai/project/u2hussai/scratch_14Sept21/dtitraining/downsample_cut_pad/'
out_path = '/home/u2hussai/dgcnn/dataHandling/subjects_lists/training.txt'

N_subs=20 #number of subjects to choose

list_filename='/home/u2hussai/dgcnn/dataHandling/subjects_lists/testing.txt' #name of output list

#convert to lists
all_subs=os.listdir(input_path)
exist_subs=[]
if out_path!=None:
    try:
        exist_subs=os.listdir(out_path)
    except:
        print('Path to check is not a dir, trying as text file')
        with open(out_path,'r') as f:
            exist_subs=f.read().splitlines()
    exist_subs_strip=[]
    for sub in exist_subs:
        if 'sub-' in exist_subs[0]:
            exist_subs_strip.append(sub.split('-')[1])
        else:
            exist_subs_strip.append(sub)
    exist_subs=exist_subs_strip
    

#loop through all subjects and stop when enough subjects are found or report if less than N_subs 
N_so_far=0
final_out_subs=[]
for sub in all_subs:
    print('\n')
    print('Loading subject: ',sub)
    if sub not in exist_subs:
        try:        
            diff_nii = nib.load(input_path + sub + '/T1w/Diffusion/data.nii.gz')
            if diff_nii.shape[-1] ==288: #check if all the directions are present
                print('Number of volumes in diffusion file:', diff_nii.shape[-1])
                final_out_subs.append(sub)
                N_so_far +=1
            else: print('Not enough diffusion directions')
        except:
            print('Could not load file')

        if N_so_far >= N_subs:
            break
    else:
        print('subject already exists')

#output text file
if len(final_out_subs) == N_subs:
    print('Enough subjects found, making text file')
    textfile=open(list_filename,'w')
    for sub in final_out_subs:
        textfile.write(sub+'\n')
    textfile.close()
else:
    print('Sorry not enough subjects. Please request fewer or get more.')

