#!/bin/bash

inpath=$1
outdir=$2
net=$3
sing=/home/u2hussai/projects/ctb-akhanf/akhanf/singularity/neuroglia_0.0.1.img 
dir=$inpath
echo directory is $dir
singularity exec --bind /home/u2hussai $sing dtifit -k $dir/data${net}.nii.gz -r $dir/bvecs${net} -b $dir/bvals${net} -m $dir/nodif_brain_mask.nii.gz -o $outdir/dtifit${net}
