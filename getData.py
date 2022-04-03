import os

#we need to get some more subj ects for predictions that not allready present in training or testing

base1='/home/u2hussai/project/u2hussai/scratch_14Sept21/dtitraining/'
subs_training = os.listdir(base1 + 'downsample_cut_pad/')
subs_testing = os.listdir(base1 + 'prediction_cut_pad/')
subs_all = subs_training + subs_testing

source_path = '/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/'
subs_source=os.listdir(source_path)

subs_not_taken = list(set(subs_source)-set(subs_all))

