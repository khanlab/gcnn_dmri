# Gauge equivariant CNNs for Diffusion MRI (gcnn_dmri)

[![PyPI](https://img.shields.io/pypi/v/gcnn_dmri?style=flat-square)](https://pypi.python.org/pypi/gcnn_dmri/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gcnn_dmri?style=flat-square)](https://pypi.python.org/pypi/gcnn_dmri/)
[![PyPI - License](https://img.shields.io/pypi/l/gcnn_dmri?style=flat-square)](https://pypi.python.org/pypi/gcnn_dmri/)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)

Authors: 
 - @uhussai7 (primary developer)
 - @akhanf 
 
---

<img src='https://github.com/uhussai7/images/blob/main/rectangle.svg' align='right' width='240'>

gcnn_dmri incorporates gauge equivariance into cnns designed to process diffusion MRI (dMRI) data. The dmri signal is realized on an antipodally identified sphere, i.e the real projective space <img src='https://latex.codecogs.com/svg.image?\mathbb{R}P^2'>. Inspired by <a href=https://arxiv.org/pdf/1902.04615.pdf>Cohen et al.</a> we model this 'half-sphere' as the top of an icosahedron. Interestingly, invoking the correct padding naturally leads us to use the full dihedral group, <img src='https://latex.codecogs.com/svg.image?D_6'> , to include reflections in addition to rotations of the hexagon, as shown in the image on the right. Here we show the application of such gauge equivariant layers to de-noising <a href ='https://www.humanconnectome.org/'>Human Connectome Project</a> dMRI data limited to six gradient directions, a problem similar to the work of <a href='https://www.sciencedirect.com/science/article/pii/S1053811920305036'>Tian et al. </a>

### Data preparation
- Select random subject id's for training and testing, one approach is shown in [`dataHandling/subject_list_generator.py`](dataHandling/subject_list_generator.py).
- Similar to <a href='https://www.sciencedirect.com/science/article/pii/S1053811920305036'>Tian et al. </a> we use a mask that avoids CSF. For this we need a grey matter and a white matter mask, which can be made from <a href='https://surfer.nmr.mgh.harvard.edu/fswiki/mri_binarize'> `mri_binarize` </a> with the flags `--all-wm` and `--gm` respectively.
- Further steps are shown in [niceData.py](niceData.py):
    - `make_freesurfer_masks` runs the shell script to make the mask mentioned above.
    - `make_loss_mask_and_structural` finalizes the mask, T1 and T2 images with the correct padding and resolution.
    - `make_diffusion` creates diffusion volumes with fewer gradient directions, directions are choosen in the sequence of the aquisition and then cut off at desired number.
    - `dtifit_on_directions` runs <a href='https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide'> `dtifit` </a> on the new diffusion volumes with fewer directions.
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
Similar to <a href='https://www.sciencedirect.com/science/article/pii/S1053811920305036'>Tian et al. </a> (and references therein) we use a residual network architecture but with the addition of gauge equivariant convolutions on the half icosahedron. The training script with the parameters used is [`training_script.py`](training_script.py). Note that structural mri images (`T1.nii.gz` and `T2.nii.gz`) are also used as inputs.

### Predictions
Predictions can be performed with the script [`predicting_script.py`](predicting_script.py). This will create a diffusion volume file, `data_network.nii.gz` along with `bvecs_network` and `bvals_network`, upon which one may perform `dtifit`. Following are some results of the denoising, the left grey images are fractional anistropy and right colored images are the `V1` vector:

<p align="center" width="100%">
    <img src='https://github.com/uhussai7/images/blob/main/dgcnn.png' width='960'>
</p>

---

**Documentation**: [https://akhanf.github.io/gcnn_dmri](https://akhanf.github.io/gcnn_dmri)

**Source Code**: [https://github.com/akhanf/gcnn_dmri](https://github.com/akhanf/gcnn_dmri)

**PyPI**: [https://pypi.org/project/gcnn_dmri/](https://pypi.org/project/gcnn_dmri/)

---

Graph-equivariant CNNs for diffusion MRI

## Installation

```sh
pip install gcnn_dmri
```

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.7+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/akhanf/gcnn_dmri/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/akhanf/gcnn_dmri/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/akhanf/gcnn_dmri/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
