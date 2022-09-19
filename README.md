# Gauge equivariant CNNs for Diffusion MRI (dgcnn)

<img src='https://github.com/uhussai7/images/blob/main/rectangle.svg' align='right' width='240'>

dgcnn incorporates gauge equivariance into cnns designed to process diffusion MRI (dMRI) data. The dmri signal is realized on an antipodally identified sphere, i.e the real projective space <img src='https://latex.codecogs.com/svg.image?\mathbb{R}P^2'>. Inspired by <a href=https://arxiv.org/pdf/1902.04615.pdf>Cohen et al.</a> we model this 'half-sphere' as the top of an icosahedron. Interestingly, invoking the correct padding naturally leads us to use the full dihedral group, <img src='https://latex.codecogs.com/svg.image?D_6'> , to include reflections in addition to rotations of the hexagon, as shown in the image on the right. Here we show the application of such gauge equivariant layers to de-noising <a href ='https://www.humanconnectome.org/'>Human Connectome Project</a> dMRI data limited to six gradient directions, a problem similar to the work of <a href='https://www.sciencedirect.com/science/article/pii/S1053811920305036'>Tian et al. </a>

### Data preparation
- Select random subject id's for training and testing, one approach is shown in [`dataHandling/subject_list_generator.py`](dataHandling/subject_list_generator.py).
- Similar to <a href='https://www.sciencedirect.com/science/article/pii/S1053811920305036'>Tian et al. </a> we use a mask that avoids CSF. For this we need a grey matter and a white matter mask, which can be made from <a href='https://surfer.nmr.mgh.harvard.edu/fswiki/mri_binarize'> `mri_binarize` </a> with the flags `--all-wm` and `--gm` respectively.
- Further steps are shown in [niceData.py](niceData.py):
    - `make_freesurfer_masks` runs the shell script to make the mask mentioned above.
    - `make_loss_mask_and_structural` finalizes the mask, T1 and T2 images with the correct padding and resolution.
    - `make_diffusion` creates diffusion volumes with fewer gradient directions, directions are choosen in the sequence of the aquisition and then cut off at desired number.
    - `dtifit_on_directions` runs dtifit on the new diffusion volumes with fewer directions.
    - We obtain the following folder structure:
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
