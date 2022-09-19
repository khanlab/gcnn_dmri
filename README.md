# Gauge equivariant CNNs for Diffusion MRI (dgcnn)

dgcnn incorporates gauge equivariance into cnns designed to process diffusion MRI (dMRI) data. The dmri signal is realized on an antipodally identified sphere, i.e the real projective space . Inspired by Cohen et al. we model this 'half-sphere' as the top of an icosahedron. Interestingly, invoking the correct padding naturally leads us to use the full dihedral group, , to include reflections in addition to rotations of the hexagon, as shown in the image on the right. Here we show the application of such gauge equivariant layers to de-noising Human Connectome Project dMRI data limited to six gradient directions, a problem similar to the work of Tian et al.

### Data preparation
- Select random subject id's for training and testing, one approach is shown in dataHandling/subject_list_generator.py.
- Similar to Tian et al. we use a mask that avoids CSF. For this we need a grey matter and a white matter mask, which can be made from mri_binarize with the flags --all-wm and --gm respectively.
- Further steps are shown in niceData.py:
  -make_freesurfer_masks runs the shell script to make the mask mentioned above.
  -make_loss_mask_and_structural finalizes the mask, T1 and T2 images with the correct padding and resolution.
  -make_diffusion creates diffusion volumes with fewer gradient directions, directions are choosen in the sequence of the aquisition and then cut off at desired number.
  -dtifit_on_directions runs dtifit on the new diffusion volumes with fewer directions.
  -We obtain the following folder structure:
  ```
    ── <training/testing>
        ├── <subject_id>
        │   ├── diffusion
        │   │   └── <# of gradient directions>
        │   │       ├── diffusion
        │   │       │   ├── bvals
        │   │       │   ├── bvecs
        │   │       │   ├── data.nii.gz
        │   │       │   ├── nodif_brain_mask.nii.gz
        │   │       │   └── S0mean.nii.gz
        │   │       └── dtifit
        │   │           ├── dtifit_< >.nii.gz
        │   ├── freesurfer_mask
        │   │   ├── mask_all_wm.nii.gz
        │   │   └── mask_gm.nii.gz
        │   ├── masks
        │   │   ├── mask_all_wm.nii.gz
        │   │   ├── mask_gm.nii.gz
        │   │   └── mask.nii.gz
        │   └── structural
        │       ├── T1.nii.gz
        │       └── T2.nii.gz
  ```
### Training
Similar to Tian et al. (and references therein) we use a residual network architecture but with the addition of gauge equivariant convolutions on the icosahedron. The training script with the parameters used is training_script.py. Note that structural mri images (T1.nii.gz and T2.nii.gz) are also used as inputs.
