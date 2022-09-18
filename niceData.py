import subprocess
import diffusion
import os
import nibabel as nib
from nibabel import processing
from cutnifti import cuts_and_pad


#this is the path where your HCP data comes from
in_path='/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/'

#this is the output path
out_path='/home/u2hussai/project/u2hussai/niceData/testing/'

#path to subjects txt
subs_list='/home/u2hussai/dgcnn/dataHandling/subjects_lists/testing.txt'
with open(subs_list,'r') as f:
    subs=f.read().splitlines()

#some functions
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_freesurfer_masks(in_path,sub,sub_dir):
    print('Making freesurfer masks for subject:', str(sub))
    aseg_path = in_path + sub + '/T1w/aparc+aseg.nii.gz'
    freesurfer_mask_path = sub_dir+ '/freesurfer_mask/' 
    make_dir(freesurfer_mask_path)

    path_to_freesurfer_script='/home/u2hussai/dgcnn/dataHandling/freesurfer_masks.sh'
    subprocess.call(path_to_freesurfer_script+' '+aseg_path+' '+freesurfer_mask_path,
                    shell=True)
    print('Finished making freesurfer masks for subject: ', str(sub))

def make_diffusion(in_path,sub,sub_dir):
    print('Preparing diffusion data for subject:', str(sub))
    try:
        diff=diffusion.diffVolume(in_path + sub + '/T1w/Diffusion/')
        diff.shells()
        diff.downSample(sub_dir+'/diffusion/') #has cutpad flat default to true
    except:
        print('Error encountered while loading diffusion data, likely missing directions!')
    print('Done with diffusion data for subject:', str(sub))

def make_loss_mask_and_structural(in_path,sub,sub_dir):
    print('Making mask for loss computation for subject:', str(sub))
    ref=nib.load(in_path + sub + '/T1w/Diffusion/nodif_brain_mask.nii.gz') #reference for resampling
    
    
    #all the mask stuff: we have to add two masks and then also bring to diffusion space
    mask_all_wm = nib.load(sub_dir + '/freesurfer_mask' + '/mask_all_wm.nii.gz')
    mask_gm = nib.load(sub_dir + '/freesurfer_mask' + '/mask_gm.nii.gz')
    masknii = nib.Nifti1Image(mask_all_wm.get_fdata() + mask_gm.get_fdata(),mask_gm.affine)
    masknii = processing.resample_from_to(masknii,ref,order=0)
    masknii = cuts_and_pad(masknii)

    mask_all_wm_cut_padnii = cuts_and_pad(processing.resample_from_to(mask_all_wm,ref,order=0))
    mask_gm_cut_padnii = cuts_and_pad(processing.resample_from_to(mask_gm,ref,order=0))  

    make_dir(sub_dir+'/masks/')
    nib.save(masknii,sub_dir+'/masks/mask.nii.gz')
    nib.save(mask_all_wm_cut_padnii,sub_dir+'/masks/mask_all_wm.nii.gz')
    nib.save(mask_gm_cut_padnii,sub_dir+'/masks/mask_gm.nii.gz')
    print('Done with masks for subject:', str(sub))

    #structural same treatment as masks
    print('Working on structural for subject:',str(sub))
    T1nii = nib.load(in_path + sub + '/T1w/T1w_acpc_dc_restore_brain.nii.gz')
    T2nii = nib.load(in_path + sub + '/T1w/T2w_acpc_dc_restore_brain.nii.gz')
    T1_cut_padnii = cuts_and_pad(processing.resample_from_to(T1nii,ref))
    T2_cut_padnii = cuts_and_pad(processing.resample_from_to(T2nii,ref))
    make_dir(sub_dir+'/structural/')
    nib.save(T1_cut_padnii,sub_dir+'/structural/T1.nii.gz')
    nib.save(T2_cut_padnii,sub_dir+'/structural/T2.nii.gz')
    print('Done with structural for subject:',str(sub))

def dtifit_on_directions(sub,sub_dir,directions):
    path_to_dtifit_script='/home/u2hussai/dgcnn/dataHandling/dtifit_on_subjects.sh'
    for direction in directions:
        print('Performing dtifit on subject %s for direction %d' % (sub,direction))
        inpath=sub_dir + '/diffusion/' + str(direction) + '/diffusion/'
        outpath = sub_dir + '/diffusion/' + str(direction) + '/dtifit/'
        make_dir(outpath)
        subprocess.call(path_to_dtifit_script+' '+inpath+' '+outpath,
                    shell=True)
        print('Done dtifit on subject %s for direction %d' % (sub,direction))

#make the subject dir
for i,sub in enumerate(subs):
    
    #if i>0: break #for debugging limit to one subject

    #make dir for subject if not there
    make_dir(out_path + sub )

    #make the freesurfer masks
    make_freesurfer_masks(in_path,sub,out_path+sub)

    #prepare masks and structural
    make_loss_mask_and_structural(in_path,sub,out_path+sub)

    #bring the 90 diffusion data but cut pad it to proper size before
    make_diffusion(in_path,sub,out_path+sub)

    #perform dtifit
    dtifit_on_directions(sub,out_path+sub,[6,90])