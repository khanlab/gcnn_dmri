import predicting

#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-125_Nepochs-50_patience-10_factor-0.5_lr-0.01_batch_size-1_interp-inverse_distance_glayers-8-8-8-1_gactivation0-relu_residual5d'
netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-600_Nepochs-50_patience-10_factor-0.5_lr-0.01_batch_size-1_interp-inverse_distance_glayers-16-16-16-16-16-1_gactivation0-relu_residual5d'
datapath= '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/sub-518746/6/'


predictor = predicting.residual5dPredictor(datapath,netpath,multigpu=False)
predictor.net=predictor.net.cuda()
predictor.predict('/home/u2hussai/scratch/')